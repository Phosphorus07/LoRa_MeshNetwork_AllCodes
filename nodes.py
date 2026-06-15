"""
LoRa Mesh Sandbox — Node Abstraction (Modules 2–4)
==================================================
A single Python class that represents a Heltec LoRa module.

All Heltecs run IDENTICAL code — the role (CONTROL, ROUTER, or SUBSCRIBER) is
configured at startup.  This mirrors what we'll do on the real hardware:
flash the same firmware to every module, then a config flag (compile-time or
runtime) decides what role each plays.

State machine for SUBSCRIBER (heartbeat-only):
    BOOT -> IDLE -> READ_SENSOR -> BUILD_PACKET -> TRANSMIT -> IDLE -> ...

State machine for CONTROL (root):
    BOOT -> LISTEN -> PARSE -> HANDLE -> LISTEN -> ...
    (also emits BEACON every BEACON_INTERVAL_S so children learn the tree)

State machine for ROUTER (relay):
    BOOT -> LISTEN -> PARSE -> HANDLE -> FORWARD -> LISTEN -> ...
    (also emits BEACON carrying its own hop_count once it has a parent)

Why three roles?  Module 4 introduces multi-hop routing.  Subscribers far from
CONTROL can't be heard directly so an intermediate ROUTER decodes their DATA
packets and re-transmits them toward CONTROL.  Both CONTROL and ROUTER
periodically broadcast BEACONs that carry their hop_count; downstream nodes
use those beacons to populate their routing tables (see routing.py).
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

from packets import (
    Packet,
    PacketType,
    BROADCAST_ID,
    encode_temperature_payload,
    decode_temperature_payload,
    encode_ack_payload,
    decode_ack_payload,
    encode_beacon_payload,
    decode_beacon_payload,
    encode_e2e_ack_payload,
    decode_e2e_ack_payload,
)


# =============================================================================
# MODULE 5 — SX1278 ENERGY CONSTANTS (datasheet, +14 dBm setting)
# =============================================================================
# These are the radio-only currents at the supply rail.  MCU sleep/run currents
# are intentionally lumped in with the "sleep" baseline; the goal is a
# defensible *radio-dominated* battery model, not a full MCU energy budget.
SX1278_TX_CURRENT_A    = 0.120     # 120 mA @ +14 dBm transmit
SX1278_RX_CURRENT_A    = 0.0115    # 11.5 mA in continuous receive
SX1278_SLEEP_CURRENT_A = 1e-6      # 1 µA quiescent (Heltec datasheet)
V_SUPPLY_V             = 3.3       # battery → buck → MCU/radio rail


# =============================================================================
# ROLE / STATE ENUMS
# =============================================================================
class NodeRole(Enum):
    """Logical role decided at boot — drives which behaviour loop runs.

    CONTROL and SUBSCRIBER were the original Module 2 set; ROUTER is added in
    Module 4 to act as a relay between distant subscribers and the root.
    """
    CONTROL    = auto()
    ROUTER     = auto()
    SUBSCRIBER = auto()


class NodeState(Enum):
    """Possible states a node can be in.  Used for logging and debug only —
    the simulator doesn't gate behaviour on these (the role and external
    scheduler do that)."""
    BOOT         = auto()
    IDLE         = auto()
    LISTEN       = auto()
    READ_SENSOR  = auto()
    BUILD_PACKET = auto()
    TRANSMIT     = auto()
    PARSE        = auto()
    HANDLE       = auto()
    FORWARD      = auto()   # NEW in Module 4 — ROUTER is relaying a DATA pkt


# =============================================================================
# SIMULATED TEMPERATURE SENSOR (unchanged from Module 2)
# =============================================================================
class TemperatureSensor:
    """
    Mock temperature sensor — replaces the real DS18B20 / DHT22 / etc.
    until the field hardware is wired up.

    Models a sensor sitting in a Karoo pump shed:
        - baseline temperature (config'd, e.g., 22°C)
        - slow diurnal drift (sinusoidal, ±5°C over 24 hours)
        - small per-reading noise (±0.3°C)

    The diurnal drift makes the data 'live' — you'll see the temperature
    drift over the course of a long simulation, which makes the dashboard
    feel real.  Interface stays identical once the real sensor is wired up,
    so we won't have to touch the rest of the codebase.
    """

    def __init__(
        self,
        baseline_c: float = 22.0,
        diurnal_amplitude_c: float = 5.0,
        noise_std_c: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.baseline  = baseline_c
        self.amplitude = diurnal_amplitude_c
        self.noise     = noise_std_c
        self.rng       = random.Random(seed)

    def read(self, sim_time_s: float) -> float:
        """Return a temperature reading at the given simulated time."""
        # 24-hour cycle = 86400 seconds; phase so peak is at noon
        phase   = (sim_time_s / 86400.0) * 2 * math.pi
        diurnal = self.amplitude * math.sin(phase - math.pi / 2)
        noise   = self.rng.gauss(0, self.noise)
        return self.baseline + diurnal + noise


# =============================================================================
# STATS DATACLASS
# =============================================================================
@dataclass
class NodeStats:
    """Running counters — a node's own understanding of its performance.

    NOTE: PDR for Module 4 is computed in run_module4.py, not from these
    counters — the orchestrator owns the authoritative delivery count
    because it has the topology information needed to credit correctly.
    """
    packets_sent:     int = 0
    packets_received: int = 0
    packets_decoded:  int = 0
    packets_rejected: int = 0   # parsed but failed validation (wrong dst, etc.)
    last_rssi:        int = 0
    last_seq_seen:    dict = field(default_factory=dict)  # src_id -> last seq


# =============================================================================
# DEFAULT CONFIG CONSTANTS
# =============================================================================
# These are sensible defaults so module-2 callers (which don't pass network_id
# or beacon_interval) still work.  Module 4 explicitly overrides each one.
DEFAULT_NETWORK_ID        = 0x01    # one farm = one network ID
DEFAULT_BEACON_INTERVAL_S = 10.0    # CONTROL/ROUTER beacon every 10 s
DEFAULT_HEARTBEAT_S       = 5.0     # subscriber heartbeat period
BATTERY_MIN_MV            = 3300    # don't drain below LiFePO4 cliff voltage
BATTERY_DRAIN_PER_TX_MV   = 1       # cosmetic — real drain handled in M5


# =============================================================================
# NODE
# =============================================================================
class LoRaNode:
    """
    A simulated Heltec LoRa node.

    Identical code, identical capabilities — what differs is the role
    config and the behaviour loop that role triggers.

    Constructor accepts a superset of every module's kwargs so we never have
    to fork the file per module.  Each kwarg is optional (with sane defaults
    or None) and the role determines which subset is actually used.
    """

    def __init__(
        self,
        node_id: int,
        role: NodeRole,
        position_m: tuple = (0.0, 0.0),
        beacon_interval_s: float = DEFAULT_BEACON_INTERVAL_S,
        heartbeat_interval_s: float = DEFAULT_HEARTBEAT_S,
        peer_id: Optional[int] = None,
        battery_mv_initial: int = 4150,
        sensor: Optional[TemperatureSensor] = None,
        routing_table=None,                        # routing.RoutingTable
        network_id: int = DEFAULT_NETWORK_ID,
        log_callback: Optional[Callable] = None,
    ):
        # ---- Identity & topology ----
        self.id        = node_id
        self.role      = role
        self.position  = position_m
        self.peer_id   = peer_id          # legacy 1:1 fallback (Module 2/3)
        self.network_id = network_id

        # ---- Timing config ----
        self.beacon_interval    = beacon_interval_s
        self.heartbeat_interval = heartbeat_interval_s

        # ---- Hardware-ish state ----
        self.battery_mv = battery_mv_initial
        self.sensor     = sensor
        self.log        = log_callback or (lambda *a, **kw: None)

        # ---- Mesh state ----
        # Stored as a plain attribute so the orchestrator can also rebind it
        # after construction (Module 4 builds the table first and passes it in;
        # Module 2 leaves it None entirely).
        self.routing_table = routing_table

        # ---- Internal state machine + counters ----
        self.state              = NodeState.BOOT
        self.next_heartbeat_at  = 0.0       # set by _boot for SUBSCRIBER
        self.next_beacon_at     = 0.0       # set externally by orchestrator
        self.tx_seq             = 0         # outgoing seq_num counter
        self.stats              = NodeStats()

        # ---- Module 5: fail-injection + reverse-route + energy accounting ---
        # `failed` is a hard kill-switch the orchestrator flips when it wants
        # to simulate a ROUTER outage (Section 6).  Every entry point that
        # builds or forwards a packet must check this and bail out silently.
        self.failed = False

        # Reverse route: when a DATA packet flows up through us we remember
        # the previous hop (downstream neighbour) keyed by the *originator*
        # subscriber ID, so that when CONTROL's E2E ACK comes back addressed
        # to that originator, we know which neighbour to forward it down to.
        # Map: originator_id -> previous_hop_id
        self.downstream_to: Dict[int, int] = {}

        # State-time integrators.  We charge in time slices: when the
        # orchestrator hands a packet to the radio we call account_tx(ToA),
        # and between events we call account_until(now) which assigns the
        # gap to TX/RX/SLEEP depending on the idle mode the node is in.
        self.time_in_tx_s    = 0.0
        self.time_in_rx_s    = 0.0
        self.time_in_sleep_s = 0.0
        self.last_accounted_at_s = 0.0

        self._boot()

    # -------------------------------------------------------------------------
    # BOOT
    # -------------------------------------------------------------------------
    def _boot(self) -> None:
        """Initial setup — equivalent to Arduino-style setup().

        IMPORTANT: SUBSCRIBER's first heartbeat offset is drawn from the
        *global* random module — Module 4's orchestrator deliberately depends
        on the RNG draw order (with seed=42) to make the SUB_NEAR / SUB_FAR
        timing reproducible.  Do not change this call without also fixing the
        seed-derived comments in run_module4.py.
        """
        self.log(
            f"Node {self.id} ({self.role.name}) boot at position {self.position}"
        )

        if self.role == NodeRole.SUBSCRIBER:
            self.state = NodeState.IDLE
            # Stagger first heartbeat slightly so multiple nodes don't sync up.
            # This was the Module 2 design; Module 4 relies on the same draw.
            self.next_heartbeat_at = random.uniform(0.5, 1.5)
        else:
            # CONTROL and ROUTER both listen by default; the orchestrator will
            # set next_beacon_at and arm the first beacon TIMER_FIRE explicitly.
            self.state = NodeState.LISTEN

    # =========================================================================
    # SUBSCRIBER BEHAVIOUR
    # =========================================================================
    def tick_subscriber(self, sim_time_s: float) -> Optional[Packet]:
        """
        Called by the simulation scheduler when this subscriber's heartbeat
        timer fires.

        Returns a Packet ready to transmit, or None if it isn't yet time
        (legacy poll-style call from Module 2).  Module 4's orchestrator only
        calls this when the timer has already fired so the None branch is
        effectively defensive.

        Side effects:
          * advances the state machine BOOT → READ_SENSOR → BUILD_PACKET → TRANSMIT → IDLE
          * drains battery by BATTERY_DRAIN_PER_TX_MV (cosmetic)
          * bumps tx_seq
          * advances next_heartbeat_at (left in place so Module 2 polling still works;
            Module 4's DES re-arms the timer itself via sched.schedule)
        """
        # Defensive — Module 2 poll-style caller might call us before our time.
        if sim_time_s < self.next_heartbeat_at:
            return None

        # ---- Read the sensor ----
        self.state = NodeState.READ_SENSOR
        # If no sensor wired, use 0.0 — useful for collision tests that don't
        # care about payload contents.
        temp_c = self.sensor.read(sim_time_s) if self.sensor else 0.0

        # Cosmetic battery drain.  Real per-state energy accounting lives in
        # Module 5; this just gives Module 2's dashboard something to display.
        self.battery_mv = max(BATTERY_MIN_MV,
                              self.battery_mv - BATTERY_DRAIN_PER_TX_MV)

        # ---- Build the DATA packet ----
        self.state = NodeState.BUILD_PACKET
        payload = encode_temperature_payload(
            temperature_c=temp_c,
            battery_mv=self.battery_mv,
            last_rssi=self.stats.last_rssi,
        )
        # dst_id = peer_id is a legacy fallback.  Module 4 overrides dst_id
        # after this call once it knows the next-hop from the routing table.
        # peer_id must therefore be SET to something sensible (CONTROL or
        # ROUTER) by the orchestrator even though it may be overridden.
        pkt = Packet(
            packet_type=PacketType.DATA,
            src_id=self.id,
            dst_id=self.peer_id if self.peer_id is not None else BROADCAST_ID,
            seq_num=self.tx_seq,
            payload=payload,
        )
        self.tx_seq = (self.tx_seq + 1) & 0xFFFF

        self.state = NodeState.TRANSMIT
        self.stats.packets_sent += 1
        self.log(
            f"[t={sim_time_s:6.2f}s] Node {self.id} TX  seq={pkt.seq_num} "
            f"temp={temp_c:.2f}°C  batt={self.battery_mv}mV"
        )

        # Module-2 poll-mode advances its own "next at" cursor.  Module 4's
        # DES re-arms the timer through the scheduler so this value is
        # informational only for it.
        self.next_heartbeat_at = sim_time_s + self.heartbeat_interval
        self.state = NodeState.IDLE
        return pkt

    # =========================================================================
    # BEACON BEHAVIOUR (CONTROL + ROUTER)
    # =========================================================================
    def tick_beacon(self, sim_time_s: float) -> Optional[Packet]:
        """
        Build a BEACON packet for broadcast.  Called by the orchestrator
        when this node's beacon timer fires.

        Returns:
            Packet with broadcast dst_id and a 2-byte payload (hop_count,
            network_id), or None if this node can't yet emit a beacon
            (ROUTER without a known hop count).

        hop_count semantics:
            CONTROL: 0   (it IS the root)
            ROUTER:  routing_table.my_hop_count(now) — typically 1 once it
                     has heard CONTROL.  If None (no parent yet) we return
                     None so the orchestrator can defer.
            SUBSCRIBER: subscribers do not beacon.  We raise loudly because
                     calling tick_beacon on a subscriber is a programming
                     error — better to fail in test than to silently mis-emit.
        """
        if self.role == NodeRole.SUBSCRIBER:
            raise RuntimeError(
                f"tick_beacon() called on SUBSCRIBER node {self.id} — "
                "only CONTROL and ROUTER may beacon."
            )

        if self.role == NodeRole.CONTROL:
            hop_count = 0
        else:   # ROUTER
            if self.routing_table is None:
                # Logic error — ROUTER must have a routing table to compute its
                # hop_count.  Fail loud rather than fudge a value.
                raise RuntimeError(
                    f"ROUTER node {self.id} has no routing_table — "
                    "cannot emit beacon."
                )
            hop_count = self.routing_table.my_hop_count(sim_time_s)
            if hop_count is None:
                # Haven't heard CONTROL yet; the orchestrator will retry on
                # the next beacon tick.  Return None silently — this is the
                # normal startup-transient case, not a bug.
                self.log(
                    f"[t={sim_time_s:6.2f}s] Node {self.id} (ROUTER) "
                    "beacon deferred — no parent yet"
                )
                return None

        payload = encode_beacon_payload(
            hop_count=hop_count,
            network_id=self.network_id,
        )
        pkt = Packet(
            packet_type=PacketType.BEACON,
            src_id=self.id,
            dst_id=BROADCAST_ID,
            seq_num=self.tx_seq,
            payload=payload,
        )
        self.tx_seq = (self.tx_seq + 1) & 0xFFFF
        self.stats.packets_sent += 1
        self.log(
            f"[t={sim_time_s:6.2f}s] Node {self.id} ({self.role.name}) "
            f"BEACON hop={hop_count} net={self.network_id}"
        )
        # Don't update next_beacon_at here — the orchestrator (DES) owns
        # beacon scheduling in Module 4.  We leave the attribute on the node
        # purely for inspection / logging.
        return pkt

    def receive_beacon(self, pkt: Packet, rssi_dbm: float, sim_time_s: float) -> None:
        """
        Process a successfully-decoded BEACON arriving at this node.

        Updates the routing table with one fresh neighbour entry.  Drops
        beacons from foreign networks (different network_id) so two farms'
        nodes won't accidentally merge their trees.

        CONTROL has no routing_table (it IS the root); for it this is a
        no-op — useful only for logging.
        """
        decoded = decode_beacon_payload(pkt.payload)
        if decoded is None:
            self.stats.packets_rejected += 1
            self.log(
                f"[t={sim_time_s:6.2f}s] Node {self.id} BEACON malformed "
                f"from {pkt.src_id}"
            )
            return

        # Foreign-network guard.  Without this two adjacent farms running the
        # same firmware would merge meshes the moment they came in range.
        if decoded["network_id"] != self.network_id:
            self.stats.packets_rejected += 1
            self.log(
                f"[t={sim_time_s:6.2f}s] Node {self.id} BEACON foreign net="
                f"{decoded['network_id']} (ours={self.network_id}) — dropped"
            )
            return

        self.stats.packets_received += 1
        self.stats.packets_decoded  += 1
        self.stats.last_rssi = int(rssi_dbm)

        self.log(
            f"[t={sim_time_s:6.2f}s] Node {self.id} BEACON from {pkt.src_id} "
            f"hop={decoded['hop_count']} RSSI={rssi_dbm:.1f}dBm"
        )

        # CONTROL has no routing table — it never selects a parent.
        if self.routing_table is None:
            return

        self.routing_table.update(
            neighbour_id = pkt.src_id,
            rssi_dbm     = rssi_dbm,
            hop_count    = decoded["hop_count"],
            now          = sim_time_s,
        )

    # =========================================================================
    # ACK GENERATION (Module 4)
    # =========================================================================
    def generate_ack(self, ack_seq: int, acked_src: int, dst_id: int) -> Packet:
        """
        Build an ACK packet that confirms reception of seq=ack_seq from
        original source acked_src to next-hop dst_id.

        The on-wire ACK payload (encode_ack_payload) carries only the
        sequence number — Module 4's ACK-matching key is (next_hop, seq) which
        is enough to identify the pending retry token uniquely.  We keep
        acked_src in the *interface* for future-proofing (Module 5's E2E ACK
        will need to identify the original sender, and we don't want two
        slightly-different ACK constructors).

        Note: this advances tx_seq as a normal packet would.  Callers that
        only want to *probe* the ToA without actually transmitting must
        decrement tx_seq themselves (run_module4.py does this when it needs
        to measure the ACK's air time before scheduling a forward).
        """
        payload = encode_ack_payload(ack_seq)
        pkt = Packet(
            packet_type=PacketType.ACK,
            src_id=self.id,
            dst_id=dst_id,
            seq_num=self.tx_seq,
            payload=payload,
        )
        self.tx_seq = (self.tx_seq + 1) & 0xFFFF
        self.log(
            f"Node {self.id} ACK->{dst_id} for src={acked_src} seq={ack_seq}"
        )
        return pkt

    # =========================================================================
    # MODULE 5 — END-TO-END ACK + REVERSE-ROUTE LEARNING
    # =========================================================================
    def learn_downstream(self, originator_id: int, previous_hop_id: int) -> None:
        """
        Record the previous-hop neighbour we received a DATA packet from,
        keyed by the *originator* subscriber ID (which may be several hops
        below us).  Used later when an E2E ACK for that originator comes
        back through us — we need to forward it down to the same neighbour.

        We deliberately key on originator_id, not prev_hop_id directly,
        because in a deeper tree several subscribers might funnel through the
        same downstream neighbour and we want a clean originator → neighbour
        map regardless of who relayed it last.
        """
        # CONTROL doesn't need a downstream table; it's the apex.  Storing
        # anyway is harmless but we early-return for clarity.
        if self.role == NodeRole.CONTROL:
            return
        self.downstream_to[originator_id] = previous_hop_id

    def next_downstream_hop(self, originator_id: int) -> Optional[int]:
        """
        Look up the next downstream neighbour to forward an E2E ACK toward
        the given originator, or None if we never saw a DATA packet from them.
        Returning None is a soft failure — the orchestrator logs and drops.
        """
        return self.downstream_to.get(originator_id)

    def generate_e2e_ack(
        self,
        ack_seq_num:   int,
        originator_id: int,
        dst_id:        int,
    ) -> Packet:
        """
        Build an end-to-end ACK_E2E packet.

        Called on CONTROL when it has successfully decoded an originator's
        DATA packet — CONTROL emits the ACK addressed to the first downstream
        hop on the reverse route.  ROUTERs do NOT call this constructor;
        instead they receive an ACK_E2E and re-emit it with a rewritten dst_id
        (handled inline in run_module5.py for visibility).

        Payload carries (ack_seq_num, originator_id) so every relay can read
        the originator without parsing the header dst_id (which is the next
        hop, not the final destination).
        """
        payload = encode_e2e_ack_payload(ack_seq_num, originator_id)
        pkt = Packet(
            packet_type=PacketType.ACK_E2E,
            src_id=self.id,
            dst_id=dst_id,
            seq_num=self.tx_seq,
            payload=payload,
        )
        self.tx_seq = (self.tx_seq + 1) & 0xFFFF
        self.log(
            f"Node {self.id} ACK_E2E->{dst_id} for orig={originator_id} "
            f"seq={ack_seq_num}"
        )
        return pkt

    # =========================================================================
    # MODULE 5 — ENERGY ACCOUNTING
    # =========================================================================
    # The model: a node is always in exactly one of {TX, RX, SLEEP} states.
    # The orchestrator drives the transitions:
    #   * account_tx(toa) is called the moment a packet is handed to the
    #     radio — it adds `toa` seconds to the TX integrator AND advances
    #     last_accounted_at_s by `toa`.  Time spent transmitting is *not*
    #     also accounted as RX/SLEEP, hence the explicit cursor advance.
    #   * account_until(now, idle_mode) is called whenever the orchestrator
    #     wants to "catch up" the integrators to the current sim time.  The
    #     gap (now - last_accounted_at_s) is assigned to whichever idle
    #     bucket the node is in between bursts (RX during a listen window,
    #     SLEEP otherwise).
    #
    # We charge once per real time slice — never re-charge a heartbeat's
    # listen window when retries happen, otherwise battery life is wildly
    # under-counted.  Run_module5.py is careful to call account_until only
    # at the boundaries of distinct events, not per retry.

    def account_tx(self, toa_s: float) -> None:
        """Add `toa_s` seconds to the TX integrator and advance the cursor."""
        if toa_s <= 0:
            return
        self.time_in_tx_s         += toa_s
        self.last_accounted_at_s  += toa_s

    def account_until(self, now_s: float, idle_mode: str = "sleep") -> None:
        """
        Catch up the idle integrators to `now_s`.

        Args:
            now_s:     simulation time we're advancing to
            idle_mode: "rx"    -> the gap was spent listening (subscriber
                                  ACK window, or always-on ROUTER/CONTROL)
                       "sleep" -> the gap was spent in deep sleep

        Silently skips if the cursor is already ahead (can happen when a TX
        burst overlapped this call) or if the elapsed time is non-positive.
        """
        gap = now_s - self.last_accounted_at_s
        if gap <= 0:
            return
        if idle_mode == "rx":
            self.time_in_rx_s    += gap
        else:
            self.time_in_sleep_s += gap
        self.last_accounted_at_s = now_s

    def energy_joules(self) -> float:
        """
        Total radio energy consumed across the simulation so far, in joules.
        Sum of (current * voltage * time) for each state.
        """
        e_tx    = SX1278_TX_CURRENT_A    * V_SUPPLY_V * self.time_in_tx_s
        e_rx    = SX1278_RX_CURRENT_A    * V_SUPPLY_V * self.time_in_rx_s
        e_sleep = SX1278_SLEEP_CURRENT_A * V_SUPPLY_V * self.time_in_sleep_s
        return e_tx + e_rx + e_sleep

    def average_current_ma(self, total_time_s: float) -> float:
        """
        Time-weighted average current draw over `total_time_s`, in mA.
        Used to project battery life from a finite simulation window.
        """
        if total_time_s <= 0:
            return 0.0
        i_avg_a = (
            SX1278_TX_CURRENT_A    * self.time_in_tx_s    +
            SX1278_RX_CURRENT_A    * self.time_in_rx_s    +
            SX1278_SLEEP_CURRENT_A * self.time_in_sleep_s
        ) / total_time_s
        return i_avg_a * 1000.0

    # =========================================================================
    # CONTROL DATA RECEPTION (preserved from Module 2 — unchanged behaviour)
    # =========================================================================
    def receive(
        self,
        raw_bytes: bytes,
        sim_time_s: float,
        rssi: float,
        snr: float,
    ) -> None:
        """
        Called when the channel delivers a unicast DATA packet to this node.

        Module 4's orchestrator only invokes this on CONTROL (it handles
        ROUTER-side reception inline so it can ACK + forward atomically).
        Module 2 used this on the single CONTROL node too, so the behaviour
        is identical to the original Module 2c implementation.
        """
        self.stats.packets_received += 1
        self.state = NodeState.PARSE

        pkt = Packet.from_bytes(raw_bytes)
        if pkt is None:
            self.stats.packets_rejected += 1
            self.log(
                f"[t={sim_time_s:6.2f}s] Node {self.id} RX "
                "CORRUPT (CRC fail or bad header)"
            )
            self.state = NodeState.LISTEN
            return

        # Filter — packet not for us?  (Broadcast 0xFF is for everyone.)
        if pkt.dst_id != self.id and pkt.dst_id != BROADCAST_ID:
            self.stats.packets_rejected += 1
            self.log(
                f"[t={sim_time_s:6.2f}s] Node {self.id} RX  "
                f"not for us (dst={pkt.dst_id})"
            )
            self.state = NodeState.LISTEN
            return

        # Duplicate detection — last-seq-per-src.  In Module 4 the orchestrator
        # owns the authoritative dedup; this is just so the node's own stats
        # don't over-count repeats.
        last_seq = self.stats.last_seq_seen.get(pkt.src_id, -1)
        if pkt.seq_num == last_seq:
            self.log(
                f"[t={sim_time_s:6.2f}s] Node {self.id} RX  "
                f"DUPLICATE seq={pkt.seq_num}"
            )
            self.state = NodeState.LISTEN
            return
        self.stats.last_seq_seen[pkt.src_id] = pkt.seq_num

        # Process
        self.state = NodeState.HANDLE
        self.stats.packets_decoded += 1
        self.stats.last_rssi = int(rssi)

        if pkt.packet_type == PacketType.DATA:
            decoded = decode_temperature_payload(pkt.payload)
            if decoded:
                self.log(
                    f"[t={sim_time_s:6.2f}s] Node {self.id} RX  "
                    f"seq={pkt.seq_num} src={pkt.src_id} "
                    f"temp={decoded['temperature_c']:.2f}°C  "
                    f"batt={decoded['battery_mv']}mV  "
                    f"RSSI={rssi:.1f}dBm  SNR={snr:.1f}dB"
                )
            else:
                self.log(
                    f"[t={sim_time_s:6.2f}s] Node {self.id} RX  unknown payload"
                )

        self.state = NodeState.LISTEN
