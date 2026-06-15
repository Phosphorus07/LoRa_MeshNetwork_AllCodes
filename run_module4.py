"""
LoRa Mesh Sandbox — Module 4: Multi-Hop Routing + Reliability
=============================================================
Author: Simon Henson (with Claude)
Project: LoRa Mesh Network Water Pump Monitoring & Control System

Adds a four-node mesh topology on top of the Module 3 DES engine:

    CONTROL (node 1) — root, at origin.  Emits beacons (hop_count=0).
    ROUTER  (node 2) — relay, 8 km east.  Hears CONTROL, emits beacons
                       (hop_count=1), forwards SUB_FAR's data toward CONTROL.
    SUB_NEAR(node 3) — 2 km east of CONTROL.  Direct one-hop link to CONTROL.
    SUB_FAR (node 4) — 8 km east of ROUTER (= 16 km east of CONTROL).
                       A ridge between CONTROL and SUB_FAR blocks direct RF
                       contact.  SUB_FAR's only path to CONTROL is via ROUTER
                       (two hops).

Routing:
    Each non-CONTROL node selects the neighbour with the strongest received
    beacon RSSI as its parent (simplified RPL OF0).  The routing table is
    updated every time a beacon arrives.

Reliability:
    DATA packets are acknowledged hop-by-hop.  If ACK is missing within
    ACK_TIMEOUT_S the sender retransmits up to MAX_RETRIES times.

Seven validation sections:
    1. Beacon propagation   — SUB_FAR selects ROUTER as parent, hop_count=2
    2. Multi-hop delivery   — SUB_FAR heartbeats reach CONTROL via ROUTER (2 hops)
    3. Single-hop sanity    — SUB_NEAR reaches CONTROL directly (1 hop)
    4. ACK/retry under loss — 12 km lossy link; retransmissions and PDR improvement
    5. PDR comparison       — 120 s, ACK OFF vs ACK ON, side-by-side figures
    6. Collision regression — Module 3 detector still fires with 4-node topology
    7. Router duty cycle    — ROUTER airtime reported; flagged if ≥10%

Run:
    cd ~/Documents/LoRa_Sandbox
    source .venv/bin/activate
    python run_module4.py
"""

import math
import random
from typing import Dict, List, Optional, Tuple

# ---- Module 1 ----
from physics import time_on_air_ms

# ---- Module 2 ----
from packets import (
    Packet, PacketType, BROADCAST_ID,
    encode_ack_payload, decode_ack_payload,
    decode_beacon_payload,
)
from nodes import LoRaNode, NodeRole, TemperatureSensor

# ---- Module 3 ----
from scheduler import Scheduler, Event, TX_START, TX_END, TIMER_FIRE, LOG_VERBOSE, LOG_SUMMARY
from channel_v3 import RadioChannelV3

# ---- Module 4 ----
from routing import RoutingTable


# =============================================================================
# DESIGN CONSTANTS — all locked from Module 1; change nothing here
# =============================================================================

# Node IDs
NODE_CONTROL_ID  = 1
NODE_ROUTER_ID   = 2
NODE_SUB_NEAR_ID = 3
NODE_SUB_FAR_ID  = 4

# Topology — positions in metres (x, y)
# CONTROL is at origin; everything is east of it.
# The ridge between CONTROL and SUB_FAR is modelled as a topology barrier
# (see REACHABLE_FROM below) rather than a physics tweak, because the
# log-distance model doesn't include terrain features.  This mirrors what
# the actual Excelsior Farm deployment will face: a kopje separates the
# far pump from the control box.
CONTROL_POS      = (0.0,    0.0)
ROUTER_X_M       = 8_000.0   # ROUTER — 8 km east of CONTROL
SUB_NEAR_X_M     = 2_000.0   # SUB_NEAR — 2 km east of CONTROL (reliable 1-hop)
SUB_FAR_ROUTER_DIST_M   = 8_000.0   # default: SUB_FAR 8 km east of ROUTER
SUB_FAR_MARGINAL_DIST_M = 12_000.0  # marginal link test (Sections 4 & 5 variant)

# LoRa RF parameters (must match all other modules)
SF               = 9
BW_HZ            = 125_000
CR               = 1       # 1 = CR 4/5
TX_POWER_DBM     = 14.0
ANTENNA_GAIN_DBI = 2.0
CABLE_LOSS_DB    = 1.0
PATH_LOSS_EXP    = 2.7
SHADOW_STD_DB    = 6.0
FREQUENCY_MHZ    = 433.0
NOISE_FLOOR_DBM  = -124.0
BATTERY_MV       = 4150

# Timing
HEARTBEAT_INTERVAL_S  = 5.0
BEACON_INTERVAL_S     = 10.0  # one beacon every 10 s from CONTROL and ROUTER
FIRST_BEACON_DELAY_S  = 0.10  # CONTROL fires its first beacon at t=0.10 s
BEACON_STAGGER_S      = 0.20  # ROUTER fires its first beacon 0.20 s AFTER CONTROL
                               # This guarantees CONTROL's beacon arrives at ROUTER
                               # (t≈0.10+0.165=0.265 s) before ROUTER emits (t=0.30 s),
                               # so ROUTER's first beacon already carries hop_count=1.

# ACK/retry parameters
# Round-trip time estimate: DATA ToA (≈0.165 s) + ACK ToA (≈0.165 s) ≈ 0.33 s.
# We set the timeout to 0.60 s — nearly double — to tolerate scheduling jitter.
ACK_TIMEOUT_S   = 0.60
MAX_RETRIES     = 3      # 3 retries after the initial attempt = 4 total attempts
RETRY_BACKOFF_S = 0.05   # small pause before retry — reduces re-collision risk

# Network ID — prevents accidentally joining a neighbouring farm's mesh
NETWORK_ID = 0x01

# RNG seed
RNG_SEED = 42

# New event type strings for Module 4 (reuse scheduler's constant syntax)
ACK_TIMEOUT_EVT = "ACK_TIMEOUT"  # fires when waiting for ACK times out

# REACHABLE_FROM[tx_id] = list of rx_ids that CAN receive from tx_id.
# The CONTROL ↔ SUB_FAR pair is blocked (terrain barrier).
# All other pairs are limited by the normal path-loss model.
REACHABLE_FROM: Dict[int, List[int]] = {
    NODE_CONTROL_ID:   [NODE_ROUTER_ID, NODE_SUB_NEAR_ID],
    NODE_ROUTER_ID:    [NODE_CONTROL_ID, NODE_SUB_NEAR_ID, NODE_SUB_FAR_ID],
    NODE_SUB_NEAR_ID:  [NODE_CONTROL_ID, NODE_ROUTER_ID],
    NODE_SUB_FAR_ID:   [NODE_ROUTER_ID],
}


# =============================================================================
# HELPERS
# =============================================================================

def euclidean_m(pos_a: tuple, pos_b: tuple) -> float:
    """Euclidean distance in metres between two (x,y) positions."""
    return math.sqrt((pos_b[0]-pos_a[0])**2 + (pos_b[1]-pos_a[1])**2)


def toa_s(pkt: Packet) -> float:
    """Time-on-Air for a packet in seconds (SF9, BW 125 kHz, CR 4/5)."""
    result = time_on_air_ms(pkt.size_bytes(), sf=SF, bw_hz=BW_HZ, cr=CR)
    return result["t_total_ms"] / 1000.0


# =============================================================================
# CORE SIMULATION FUNCTION
# =============================================================================

def run_mesh_simulation(
    sim_duration_s: float,
    rng_seed: int,
    ack_enabled: bool,
    log_fn=None,
    sub_far_to_router_m: float = SUB_FAR_ROUTER_DIST_M,
) -> dict:
    """
    Run the four-node mesh for sim_duration_s seconds and return metrics.

    Args:
        sim_duration_s      — how long to simulate
        rng_seed            — seeds both global random (node boot jitter) and
                              the channel's private RNG (shadow fading)
        ack_enabled         — whether hop-by-hop ACK+retry is active
        log_fn              — print or None; verbose event log if provided
        sub_far_to_router_m — distance between SUB_FAR and ROUTER in metres.
                              Use SUB_FAR_MARGINAL_DIST_M for lossy-link tests.

    Returns dict with keys:
        sent_sub_near, sent_sub_far
        delivered_sub_near, delivered_sub_far
        pdr_sub_near, pdr_sub_far   (floats, 0–100)
        collisions, retransmissions, retry_exhausted
        router_airtime_s, router_duty_cycle_pct
        rt_sub_far_parent, rt_sub_far_hops   (routing-table snapshot at end)
        rt_sub_near_parent, rt_sub_near_hops
    """

    # ---- RNG ----
    # Global seed controls boot-time jitter (random.uniform in LoRaNode._boot).
    # Channel gets its own private Random(seed) for shadow fading.
    random.seed(rng_seed)

    # ---- Channel ----
    channel = RadioChannelV3(
        path_loss_exponent  = PATH_LOSS_EXP,
        shadow_fading_std_db= SHADOW_STD_DB,
        sf                  = SF,
        tx_power_dbm        = TX_POWER_DBM,
        tx_gain_dbi         = ANTENNA_GAIN_DBI,
        rx_gain_dbi         = ANTENNA_GAIN_DBI,
        cable_loss_db       = CABLE_LOSS_DB,
        noise_floor_dbm     = NOISE_FLOOR_DBM,
        seed                = rng_seed,
    )

    # ---- Routing tables (one per non-CONTROL node) ----
    rt_router   = RoutingTable(NODE_ROUTER_ID,   NODE_CONTROL_ID)
    rt_sub_near = RoutingTable(NODE_SUB_NEAR_ID, NODE_CONTROL_ID)
    rt_sub_far  = RoutingTable(NODE_SUB_FAR_ID,  NODE_CONTROL_ID)

    # ---- Positions ----
    sub_far_x = ROUTER_X_M + sub_far_to_router_m
    positions = {
        NODE_CONTROL_ID:   CONTROL_POS,
        NODE_ROUTER_ID:    (ROUTER_X_M,   0.0),
        NODE_SUB_NEAR_ID:  (SUB_NEAR_X_M, 0.0),
        NODE_SUB_FAR_ID:   (sub_far_x,    0.0),
    }

    def dist(a_id: int, b_id: int) -> float:
        """Distance in metres between two nodes by ID."""
        return euclidean_m(positions[a_id], positions[b_id])

    # ---- Nodes ----
    # Node construction order matters: SUBSCRIBER._boot() calls random.uniform,
    # consuming the global RNG.  We build CONTROL and ROUTER first (they don't
    # consume from global RNG in _boot), then the two subscribers.
    # With seed=42 this gives SUB_NEAR.next_heartbeat_at ≈ 1.139 s (first
    # random.uniform call) and SUB_FAR.next_heartbeat_at from the second call.
    _log = log_fn or (lambda *_: None)

    control = LoRaNode(
        node_id           = NODE_CONTROL_ID,
        role              = NodeRole.CONTROL,
        position_m        = positions[NODE_CONTROL_ID],
        beacon_interval_s = BEACON_INTERVAL_S,
        network_id        = NETWORK_ID,
        log_callback      = _log,
    )

    router = LoRaNode(
        node_id           = NODE_ROUTER_ID,
        role              = NodeRole.ROUTER,
        position_m        = positions[NODE_ROUTER_ID],
        beacon_interval_s = BEACON_INTERVAL_S,
        network_id        = NETWORK_ID,
        routing_table     = rt_router,
        log_callback      = _log,
    )

    sub_near = LoRaNode(
        node_id              = NODE_SUB_NEAR_ID,
        role                 = NodeRole.SUBSCRIBER,
        position_m           = positions[NODE_SUB_NEAR_ID],
        heartbeat_interval_s = HEARTBEAT_INTERVAL_S,
        peer_id              = NODE_CONTROL_ID,  # default; overridden by routing table
        battery_mv_initial   = BATTERY_MV,
        sensor               = TemperatureSensor(seed=rng_seed + 2),
        routing_table        = rt_sub_near,
        network_id           = NETWORK_ID,
        log_callback         = _log,
    )

    sub_far = LoRaNode(
        node_id              = NODE_SUB_FAR_ID,
        role                 = NodeRole.SUBSCRIBER,
        position_m           = positions[NODE_SUB_FAR_ID],
        heartbeat_interval_s = HEARTBEAT_INTERVAL_S,
        peer_id              = NODE_ROUTER_ID,   # default; overridden by routing table
        battery_mv_initial   = BATTERY_MV,
        sensor               = TemperatureSensor(seed=rng_seed + 3),
        routing_table        = rt_sub_far,
        network_id           = NETWORK_ID,
        log_callback         = _log,
    )

    all_nodes = {
        NODE_CONTROL_ID:   control,
        NODE_ROUTER_ID:    router,
        NODE_SUB_NEAR_ID:  sub_near,
        NODE_SUB_FAR_ID:   sub_far,
    }

    # ---- Scheduler ----
    log_level = LOG_VERBOSE if log_fn else LOG_SUMMARY
    sched = Scheduler(log_level=log_level)

    # ---- Shared mutable state (closures capture these) ----
    results = {
        "sent_sub_near":        0,
        "sent_sub_far":         0,
        "delivered_sub_near":   0,
        "delivered_sub_far":    0,
        "collisions":           0,
        "retransmissions":      0,
        "retry_exhausted":      0,
        "router_airtime_s":     0.0,
    }

    # pending_acks: set of (node_id, seq_num, attempt) tokens.
    # When an ACK arrives, the matching tokens are removed.
    # When ACK_TIMEOUT fires, if the token is still present, retry.
    pending_acks: set = set()

    # ---- Beacon timing setup ----
    # CONTROL fires first at FIRST_BEACON_DELAY_S = 0.10 s.
    # ROUTER fires first at 0.10 + BEACON_STAGGER_S = 0.30 s.
    # By t=0.265 s CONTROL's first beacon has arrived at ROUTER, so ROUTER's
    # routing table has CONTROL as parent before ROUTER emits its own beacon.
    control.next_beacon_at = FIRST_BEACON_DELAY_S
    router.next_beacon_at  = FIRST_BEACON_DELAY_S + BEACON_STAGGER_S

    # ---- Seed initial events ----
    sched.schedule(FIRST_BEACON_DELAY_S,
                   TIMER_FIRE, {"timer_type": "beacon", "node": control})

    sched.schedule(FIRST_BEACON_DELAY_S + BEACON_STAGGER_S,
                   TIMER_FIRE, {"timer_type": "beacon", "node": router})

    sched.schedule(sub_near.next_heartbeat_at,
                   TIMER_FIRE, {"timer_type": "heartbeat", "node": sub_near})

    sched.schedule(sub_far.next_heartbeat_at,
                   TIMER_FIRE, {"timer_type": "heartbeat", "node": sub_far})

    # =========================================================================
    # INNER HELPERS (closures over sched, channel, results, pending_acks)
    # =========================================================================

    def node_dist(sender_id: int, receiver_id: int) -> float:
        return dist(sender_id, receiver_id)

    def _schedule_tx(
        sender_id: int,
        receiver,           # int (unicast) or list[(int,float)] (broadcast)
        pkt: Packet,
        pkt_role: str,
        backoff_s: float = 0.0,
    ) -> None:
        """
        Push a TX_START event (which will itself schedule TX_END).

        For unicast:  receiver is an int node ID.
        For broadcast: receiver is a list of (rx_id, distance_m) tuples.

        The backoff_s delays TX_START — used for retries to reduce re-collision.
        """
        raw       = pkt.to_bytes()
        pkt_toa   = toa_s(pkt)
        tx_start_t = sched.now() + backoff_s
        tx_end_t   = tx_start_t + pkt_toa

        # Track ROUTER airtime (forward + ACK + beacon transmissions)
        if sender_id == NODE_ROUTER_ID:
            results["router_airtime_s"] += pkt_toa

        is_broadcast = isinstance(receiver, list)

        if is_broadcast:
            # receiver = [(rx_id, dist_m), ...]
            # Use minimum distance as the distance_m stored in register_tx —
            # this value is unused by pop_collision_flag but must be > 0.
            min_d = min((d for _, d in receiver), default=1.0)
            payload_d = {
                "pkt_role":   pkt_role,
                "sender_id":  sender_id,
                "receivers":  receiver,          # list[(rx_id, dist_m)]
                "pkt":        pkt,
                "raw_bytes":  raw,
                "tx_start":   tx_start_t,
                "tx_end":     tx_end_t,
                "distance_m": min_d,
            }
        else:
            dist_m = node_dist(sender_id, receiver)
            payload_d = {
                "pkt_role":   pkt_role,
                "sender_id":  sender_id,
                "receiver_id": receiver,
                "pkt":        pkt,
                "raw_bytes":  raw,
                "tx_start":   tx_start_t,
                "tx_end":     tx_end_t,
                "distance_m": dist_m,
            }

        sched.schedule(backoff_s, TX_START, payload_d)

    def _schedule_ack(sender_id: int, dst_id: int, ack_seq: int, acked_src: int) -> None:
        """Schedule an ACK packet transmission."""
        ack_node = all_nodes[sender_id]
        ack_pkt  = ack_node.generate_ack(ack_seq, acked_src, dst_id)
        _schedule_tx(sender_id, dst_id, ack_pkt, pkt_role="ack")

    def _schedule_forward(original_pkt: Packet, backoff_s: float = 0.0) -> None:
        """
        ROUTER re-sends original_pkt toward CONTROL, preserving src_id and
        seq_num so CONTROL credits the delivery to the original subscriber.

        backoff_s: how long to wait before TX_START.  When ACK is enabled this
        must be at least the ACK ToA so ROUTER finishes the ACK before starting
        the forward (only one TX chain on a real radio).
        """
        fwd = Packet(
            packet_type = PacketType.DATA,
            src_id      = original_pkt.src_id,      # preserve original source
            dst_id      = NODE_CONTROL_ID,
            seq_num     = original_pkt.seq_num,      # preserve seq for dup-detect
            payload     = original_pkt.payload,
        )
        _schedule_tx(NODE_ROUTER_ID, NODE_CONTROL_ID, fwd, pkt_role="forwarded",
                     backoff_s=backoff_s)

    # =========================================================================
    # DISPATCH
    # =========================================================================

    def dispatch(evt: Event) -> None:
        t = sched.now()
        et = evt.event_type

        if et == TIMER_FIRE:
            _handle_timer(evt, t)
        elif et == TX_START:
            _handle_tx_start(evt, t)
        elif et == TX_END:
            _handle_tx_end(evt, t)
        elif et == ACK_TIMEOUT_EVT:
            _handle_ack_timeout(evt, t)

    # ---- TIMER_FIRE ----

    def _handle_timer(evt: Event, t: float) -> None:
        timer_type = evt.payload["timer_type"]
        node       = evt.payload["node"]

        if timer_type == "heartbeat":
            _do_heartbeat(node, t)
        elif timer_type == "beacon":
            _do_beacon(node, t)

    def _do_heartbeat(node: LoRaNode, t: float) -> None:
        """Subscriber tries to send a heartbeat toward its best parent."""

        # If routing table is not yet populated, defer by 1 s and wait for beacon.
        if node.routing_table and not node.routing_table.has_route(t):
            _log(
                f"[t={t:.2f}s] Node {node.id}: no route yet — "
                "deferring heartbeat 1.0 s"
            )
            sched.schedule(1.0, TIMER_FIRE, {"timer_type": "heartbeat", "node": node})
            return

        # tick_subscriber builds the DATA packet and advances the node state machine
        pkt = node.tick_subscriber(t)
        if pkt is None:
            return  # shouldn't happen, but guard anyway

        # Determine next hop from routing table (or fall back to peer_id)
        if node.routing_table:
            next_hop = node.routing_table.best_parent(t)
        else:
            next_hop = node.peer_id

        if next_hop is None:
            _log(f"[t={t:.2f}s] Node {node.id}: no next hop — packet dropped")
            return

        # Override dst to next_hop (tick_subscriber uses peer_id which may be stale)
        if pkt.dst_id != next_hop:
            pkt = Packet(PacketType.DATA, pkt.src_id, next_hop, pkt.seq_num, pkt.payload)

        # Track sent count for PDR calculation
        if node.id == NODE_SUB_NEAR_ID:
            results["sent_sub_near"] += 1
        elif node.id == NODE_SUB_FAR_ID:
            results["sent_sub_far"] += 1

        _schedule_tx(node.id, next_hop, pkt, pkt_role="data")

        # Schedule the next heartbeat (DES controls timing, not next_heartbeat_at)
        sched.schedule(HEARTBEAT_INTERVAL_S, TIMER_FIRE,
                       {"timer_type": "heartbeat", "node": node})

        # If ACK is enabled, set a timeout for this transmission
        if ack_enabled:
            attempt = 0
            token   = (node.id, pkt.seq_num, attempt)
            pending_acks.add(token)
            pkt_toa = toa_s(pkt)
            sched.schedule(
                pkt_toa + ACK_TIMEOUT_S,
                ACK_TIMEOUT_EVT,
                {
                    "node":       node,
                    "seq_num":    pkt.seq_num,
                    "attempt":    attempt,
                    "next_hop":   next_hop,
                    "pkt":        pkt,          # kept for retransmission
                }
            )

    def _do_beacon(node: LoRaNode, t: float) -> None:
        """CONTROL or ROUTER emits a beacon broadcast."""
        pkt = node.tick_beacon(t)
        if pkt is None:
            # Guard: tick_beacon returned None (timing mismatch) — reschedule
            sched.schedule(BEACON_INTERVAL_S, TIMER_FIRE,
                           {"timer_type": "beacon", "node": node})
            return

        # Build broadcast receiver list: (rx_id, distance_m) for each eligible node
        eligible  = REACHABLE_FROM.get(node.id, [])
        receivers = [(rx_id, node_dist(node.id, rx_id)) for rx_id in eligible]

        _schedule_tx(node.id, receivers, pkt, pkt_role="beacon")

        # Schedule next beacon timer
        sched.schedule(BEACON_INTERVAL_S, TIMER_FIRE,
                       {"timer_type": "beacon", "node": node})

    # ---- TX_START ----

    def _handle_tx_start(evt: Event, t: float) -> None:
        """Register the transmission in the channel (collision tracking) and schedule TX_END."""
        p = evt.payload
        channel.register_tx(
            tx_start   = p["tx_start"],
            tx_end     = p["tx_end"],
            node_id    = p["sender_id"],
            freq_mhz   = FREQUENCY_MHZ,
            sf         = SF,
            distance_m = p["distance_m"],
        )
        # Delay until TX_END; forward the entire payload so TX_END handler has context
        delay = max(0.0, p["tx_end"] - t)
        sched.schedule(delay, TX_END, p)

    # ---- TX_END ----

    def _handle_tx_end(evt: Event, t: float) -> None:
        role = evt.payload["pkt_role"]
        if role == "beacon":
            _deliver_beacon(evt, t)
        elif role in ("data", "forwarded"):
            _deliver_data(evt, t)
        elif role == "ack":
            _deliver_ack(evt, t)

    def _deliver_beacon(evt: Event, t: float) -> None:
        """Evaluate a broadcast beacon for all eligible receivers."""
        p         = evt.payload
        sender_id = p["sender_id"]
        receivers = p["receivers"]    # list[(rx_id, dist_m)]
        pkt       = p["pkt"]

        # pop_collision_flag: check + remove from in_flight WITHOUT touching RNG
        collided = channel.pop_collision_flag(
            tx_end_time = p["tx_end"],
            node_id     = sender_id,
            freq_mhz    = FREQUENCY_MHZ,
            sf          = SF,
        )
        if collided:
            results["collisions"] += 1
            _log(f"[t={t:.4f}s] BEACON from={sender_id}: COLLISION — all receivers lost")
            return

        # Evaluate path loss and deliver to each receiver independently
        for rx_id, dist_m in receivers:
            rx = channel.transmit(dist_m)   # draws from channel.rng
            if rx.decoded:
                rx_node = all_nodes[rx_id]
                rx_node.receive_beacon(pkt, rx.rssi_dbm, t)

    def _deliver_data(evt: Event, t: float) -> None:
        """Evaluate unicast DATA delivery; handle forwarding and ACK generation."""
        p           = evt.payload
        sender_id   = p["sender_id"]
        receiver_id = p["receiver_id"]
        pkt         = p["pkt"]
        raw         = p["raw_bytes"]
        role        = p["pkt_role"]

        rx = channel.resolve_rx(
            tx_end_time = p["tx_end"],
            node_id     = sender_id,
            freq_mhz    = FREQUENCY_MHZ,
            sf          = SF,
        )

        if not rx.decoded:
            if rx.failure_reason == "collision":
                results["collisions"] += 1
            _log(
                f"[t={t:.4f}s] DATA {sender_id}→{receiver_id}  "
                f"seq={pkt.seq_num}  LOST  reason={rx.failure_reason}"
            )
            return

        _log(
            f"[t={t:.4f}s] DATA {sender_id}→{receiver_id}  "
            f"seq={pkt.seq_num}  RSSI={rx.rssi_dbm:.1f} dBm  DECODED ✓"
        )

        if receiver_id == NODE_CONTROL_ID:
            # -----------------------------------------------------------------
            # End-to-end delivery at CONTROL
            # -----------------------------------------------------------------
            control.receive(raw, t, rx.rssi_dbm, rx.snr_db)
            orig_src = pkt.src_id
            if orig_src == NODE_SUB_NEAR_ID:
                results["delivered_sub_near"] += 1
            elif orig_src == NODE_SUB_FAR_ID:
                results["delivered_sub_far"] += 1

            if ack_enabled and role == "data":
                # Direct delivery from SUB_NEAR — CONTROL ACKs back to sender
                _schedule_ack(NODE_CONTROL_ID, sender_id, pkt.seq_num, pkt.src_id)

        elif receiver_id == NODE_ROUTER_ID:
            # -----------------------------------------------------------------
            # ROUTER receives from SUB_FAR: ACK the subscriber, then forward.
            # ROUTER cannot transmit ACK and forward simultaneously — a real
            # radio has only one TX chain.  We stagger the forward by the ACK's
            # Time-on-Air (≈165 ms at SF9) plus a small guard margin so the
            # ACK TX_END fires before the forward TX_START.
            # -----------------------------------------------------------------
            if ack_enabled:
                # Hop-by-hop ACK first: ROUTER sends ACK to SUB_FAR.
                _schedule_ack(NODE_ROUTER_ID, sender_id, pkt.seq_num, pkt.src_id)
                # Compute how long the ACK occupies the air so we can delay forward.
                _ack_pkt = all_nodes[NODE_ROUTER_ID].generate_ack(
                    pkt.seq_num, pkt.src_id, sender_id
                )
                fwd_delay = toa_s(_ack_pkt) + 0.02  # 20 ms guard beyond ACK end
            else:
                fwd_delay = 0.0  # no ACK — forward immediately

            router.state = __import__("nodes").NodeState.FORWARD
            _schedule_forward(pkt, backoff_s=fwd_delay)

        # If any other node ID receives a DATA packet (shouldn't happen with our
        # topology, but guard against logic errors):
        elif receiver_id not in all_nodes:
            _log(f"[t={t:.4f}s] WARNING: DATA delivered to unknown node {receiver_id}")

    def _deliver_ack(evt: Event, t: float) -> None:
        """Evaluate unicast ACK delivery; clear pending retry state."""
        p           = evt.payload
        sender_id   = p["sender_id"]
        receiver_id = p["receiver_id"]
        pkt         = p["pkt"]

        rx = channel.resolve_rx(
            tx_end_time = p["tx_end"],
            node_id     = sender_id,
            freq_mhz    = FREQUENCY_MHZ,
            sf          = SF,
        )

        if not rx.decoded:
            _log(
                f"[t={t:.4f}s] ACK {sender_id}→{receiver_id}  "
                f"LOST  reason={rx.failure_reason}"
            )
            return   # subscriber will time out and retry

        ack_data = decode_ack_payload(pkt.payload)
        if ack_data is None:
            return

        ack_seq  = ack_data["ack_seq_num"]

        _log(
            f"[t={t:.4f}s] ACK {sender_id}→{receiver_id}  "
            f"seq={ack_seq}  RECEIVED ✓"
        )

        # Remove ALL pending tokens for (receiver_id, ack_seq) regardless of attempt.
        # This clears any stale timeouts that haven't fired yet.
        # Use difference_update() (in-place, no rebind) rather than -= so that
        # Python's closure rules don't treat pending_acks as a local variable.
        stale = {tok for tok in pending_acks
                 if tok[0] == receiver_id and tok[1] == ack_seq}
        pending_acks.difference_update(stale)

    # ---- ACK_TIMEOUT ----

    def _handle_ack_timeout(evt: Event, t: float) -> None:
        """
        Fired when a subscriber did not receive an ACK within ACK_TIMEOUT_S.

        If the pending_acks token is gone (ACK arrived after schedule but
        before this fires) — do nothing (stale timeout).
        Otherwise retry up to MAX_RETRIES times.
        """
        p       = evt.payload
        node    = p["node"]
        seq_num = p["seq_num"]
        attempt = p["attempt"]
        token   = (node.id, seq_num, attempt)

        if token not in pending_acks:
            return   # ACK already received — this timeout is stale

        pending_acks.discard(token)

        if attempt >= MAX_RETRIES:
            _log(
                f"[t={t:.4f}s] Node {node.id} seq={seq_num}: "
                f"MAX_RETRIES={MAX_RETRIES} exceeded — packet abandoned"
            )
            results["retry_exhausted"] += 1
            return

        # ---- Retry ----
        new_attempt = attempt + 1
        results["retransmissions"] += 1
        pkt      = p["pkt"]
        next_hop = p["next_hop"]

        _log(
            f"[t={t:.4f}s] Node {node.id} seq={seq_num}: "
            f"RETRY {new_attempt}/{MAX_RETRIES}"
        )

        # Retransmit with a small backoff to reduce immediate re-collision risk
        _schedule_tx(node.id, next_hop, pkt, pkt_role="data",
                     backoff_s=RETRY_BACKOFF_S)

        # Schedule the next ACK timeout for this retry attempt
        pkt_toa   = toa_s(pkt)
        new_token = (node.id, seq_num, new_attempt)
        pending_acks.add(new_token)
        sched.schedule(
            RETRY_BACKOFF_S + pkt_toa + ACK_TIMEOUT_S,
            ACK_TIMEOUT_EVT,
            {
                "node":     node,
                "seq_num":  seq_num,
                "attempt":  new_attempt,
                "next_hop": next_hop,
                "pkt":      pkt,
            }
        )

    # =========================================================================
    # RUN
    # =========================================================================
    sched.run_until(sim_duration_s, dispatch)

    # ---- Final metrics ----
    def safe_pdr(delivered: int, sent: int) -> float:
        return 100.0 * delivered / sent if sent > 0 else 0.0

    results["pdr_sub_near"]         = safe_pdr(results["delivered_sub_near"], results["sent_sub_near"])
    results["pdr_sub_far"]          = safe_pdr(results["delivered_sub_far"],  results["sent_sub_far"])
    results["router_duty_cycle_pct"]= 100.0 * results["router_airtime_s"] / sim_duration_s

    # Routing-table snapshots (used by test sections)
    results["rt_sub_far_parent"]  = rt_sub_far.best_parent(sim_duration_s)
    results["rt_sub_far_hops"]    = rt_sub_far.my_hop_count(sim_duration_s)
    results["rt_sub_near_parent"] = rt_sub_near.best_parent(sim_duration_s)
    results["rt_sub_near_hops"]   = rt_sub_near.my_hop_count(sim_duration_s)
    results["rt_router_parent"]   = rt_router.best_parent(sim_duration_s)
    results["rt_router_hops"]     = rt_router.my_hop_count(sim_duration_s)

    return results


# =============================================================================
# SECTION 6 — COLLISION REGRESSION (standalone, no full sim needed)
# =============================================================================

def _section_6_collision_regression() -> bool:
    """
    Verify that Module 3's collision detector fires correctly when three
    Module 4 nodes (CONTROL, ROUTER, SUB_NEAR) all transmit simultaneously.

    Uses RadioChannelV3 directly — no DES needed.
    All three packets overlap completely → all three must be 'collision'.

    This confirms that adding more node roles hasn't broken the collision
    detection logic introduced in Module 3.
    """
    ch = RadioChannelV3(seed=42)

    # All three transmit simultaneously at t=1.0 s
    tx_s  = 1.0
    pkt_bytes = 16   # standard heartbeat size
    toa = time_on_air_ms(pkt_bytes, sf=SF, bw_hz=BW_HZ, cr=CR)["t_total_ms"] / 1000.0
    tx_e = tx_s + toa

    ch.register_tx(tx_s, tx_e, NODE_CONTROL_ID,   FREQUENCY_MHZ, SF,
                   euclidean_m(CONTROL_POS, (ROUTER_X_M, 0.0)))
    ch.register_tx(tx_s, tx_e, NODE_ROUTER_ID,    FREQUENCY_MHZ, SF,
                   euclidean_m((ROUTER_X_M, 0.0), (SUB_NEAR_X_M, 0.0)))
    ch.register_tx(tx_s, tx_e, NODE_SUB_NEAR_ID,  FREQUENCY_MHZ, SF,
                   euclidean_m((SUB_NEAR_X_M, 0.0), CONTROL_POS))

    rx_c  = ch.resolve_rx(tx_e, NODE_CONTROL_ID,  FREQUENCY_MHZ, SF)
    rx_r  = ch.resolve_rx(tx_e, NODE_ROUTER_ID,   FREQUENCY_MHZ, SF)
    rx_sn = ch.resolve_rx(tx_e, NODE_SUB_NEAR_ID, FREQUENCY_MHZ, SF)

    print(f"    CONTROL:  decoded={rx_c.decoded}   reason={rx_c.failure_reason}")
    print(f"    ROUTER:   decoded={rx_r.decoded}   reason={rx_r.failure_reason}")
    print(f"    SUB_NEAR: decoded={rx_sn.decoded}  reason={rx_sn.failure_reason}")

    return (
        rx_c.failure_reason  == "collision"
        and rx_r.failure_reason  == "collision"
        and rx_sn.failure_reason == "collision"
    )


# =============================================================================
# PRINT HELPERS
# =============================================================================

def _sep(char: str = "=", width: int = 70) -> None:
    print(char * width)

def _header(title: str) -> None:
    _sep()
    print(f"  {title}")
    _sep()

def _result(label: str, value, unit: str = "") -> None:
    print(f"    {label:<30} {value}{unit}")


# =============================================================================
# MAIN — seven test sections
# =============================================================================

if __name__ == "__main__":

    print()
    print("LoRa Mesh Sandbox — Module 4")
    print("Multi-hop routing + ACK/retry\n")

    # =========================================================================
    # SECTION 1 — Beacon propagation
    # =========================================================================
    _header("SECTION 1 — Beacon propagation (30 s, seed=42)")
    print(
        "  SUB_FAR is out of direct RF contact with CONTROL (terrain barrier).\n"
        "  CONTROL emits beacon (hop_count=0) → ROUTER hears it → ROUTER emits\n"
        "  beacon (hop_count=1) → SUB_FAR selects ROUTER as parent (hop_count=2).\n"
    )

    r1 = run_mesh_simulation(30.0, RNG_SEED, ack_enabled=False)

    print(f"    SUB_FAR  best_parent={r1['rt_sub_far_parent']}  "
          f"hop_count={r1['rt_sub_far_hops']}")
    print(f"    SUB_NEAR best_parent={r1['rt_sub_near_parent']}  "
          f"hop_count={r1['rt_sub_near_hops']}")
    print(f"    ROUTER   best_parent={r1['rt_router_parent']}  "
          f"hop_count={r1['rt_router_hops']}")

    s1_pass = (
        r1["rt_sub_far_parent"] == NODE_ROUTER_ID    and
        r1["rt_sub_far_hops"]   == 2                 and
        r1["rt_sub_near_parent"] in (NODE_CONTROL_ID, NODE_ROUTER_ID) and
        r1["rt_sub_near_hops"]  <= 2                 and
        r1["rt_router_parent"]  == NODE_CONTROL_ID   and
        r1["rt_router_hops"]    == 1
    )
    print(f"\n  Section 1: {'PASS — routing tree built correctly ✓' if s1_pass else 'FAIL'}")

    # =========================================================================
    # SECTION 2 — Multi-hop delivery
    # =========================================================================
    _header("SECTION 2 — Multi-hop delivery (300 s, 8 km links, ACK OFF)")
    print(
        "  SUB_FAR heartbeats must traverse two hops:\n"
        "    SUB_FAR → ROUTER → CONTROL\n"
    )

    r2 = run_mesh_simulation(300.0, RNG_SEED, ack_enabled=False)

    _result("SUB_FAR sent",        r2["sent_sub_far"],     " pkts")
    _result("SUB_FAR delivered",   r2["delivered_sub_far"], " pkts")
    _result("SUB_FAR PDR",         f"{r2['pdr_sub_far']:.1f}", " %")
    _result("SUB_NEAR PDR",        f"{r2['pdr_sub_near']:.1f}", " %")

    s2_pass = (
        r2["sent_sub_far"] > 0 and
        r2["delivered_sub_far"] > 0 and
        r2["rt_sub_far_hops"] == 2
    )
    print(f"\n  Section 2: {'PASS — SUB_FAR data arrived at CONTROL via 2-hop path ✓' if s2_pass else 'FAIL'}")

    # =========================================================================
    # SECTION 3 — Single-hop sanity
    # =========================================================================
    _header("SECTION 3 — Single-hop sanity (300 s, ACK OFF)")
    print("  SUB_NEAR (2 km from CONTROL) should route directly — 1 hop.\n")

    # Reuse Section 2 results (same run)
    _result("SUB_NEAR sent",       r2["sent_sub_near"],      " pkts")
    _result("SUB_NEAR delivered",  r2["delivered_sub_near"],  " pkts")
    _result("SUB_NEAR PDR",        f"{r2['pdr_sub_near']:.1f}", " %")
    _result("SUB_NEAR hop count",  r2["rt_sub_near_hops"],    " hop(s)")

    s3_pass = (
        r2["sent_sub_near"] > 0 and
        r2["delivered_sub_near"] > 0 and
        r2["rt_sub_near_hops"] == 1
    )
    print(f"\n  Section 3: {'PASS — SUB_NEAR delivers directly to CONTROL in 1 hop ✓' if s3_pass else 'FAIL'}")

    # =========================================================================
    # SECTION 4 — ACK/retry under loss
    # =========================================================================
    _header("SECTION 4 — ACK/retry under loss (300 s, SUB_FAR 12 km from ROUTER)")
    print(
        "  The SUB_FAR→ROUTER link is stretched to 12 km (~93% per-hop PDR).\n"
        "  With ACK OFF, end-to-end loss is ~7% per hop.\n"
        "  With ACK ON (3 retries), effective per-hop PDR → ~99.998%.\n"
    )

    r4_off = run_mesh_simulation(300.0, RNG_SEED, ack_enabled=False,
                                  sub_far_to_router_m=SUB_FAR_MARGINAL_DIST_M)
    r4_on  = run_mesh_simulation(300.0, RNG_SEED, ack_enabled=True,
                                  sub_far_to_router_m=SUB_FAR_MARGINAL_DIST_M)

    print(f"\n    {'Metric':<32} {'ACK OFF':>10}  {'ACK ON':>10}")
    print(f"    {'-'*54}")
    print(f"    {'SUB_FAR sent':<32} {r4_off['sent_sub_far']:>10}  {r4_on['sent_sub_far']:>10}")
    print(f"    {'SUB_FAR delivered':<32} {r4_off['delivered_sub_far']:>10}  {r4_on['delivered_sub_far']:>10}")
    print(f"    {'SUB_FAR PDR':<32} {r4_off['pdr_sub_far']:>9.1f}%  {r4_on['pdr_sub_far']:>9.1f}%")
    print(f"    {'Retransmissions':<32} {r4_off['retransmissions']:>10}  {r4_on['retransmissions']:>10}")
    print(f"    {'Retry exhausted':<32} {r4_off['retry_exhausted']:>10}  {r4_on['retry_exhausted']:>10}")

    s4_pass = (
        r4_on["retransmissions"] > 0           and  # retries actually occurred
        r4_on["pdr_sub_far"] >= r4_off["pdr_sub_far"]  # ACK ON ≥ ACK OFF PDR
    )
    print(f"\n  Section 4: {'PASS — retransmissions occurred and PDR improved with ACK ✓' if s4_pass else 'FAIL'}")

    # =========================================================================
    # SECTION 5 — PDR comparison (viva graph data)
    # =========================================================================
    _header("SECTION 5 — PDR comparison, 120 s full scenario (8 km links)")
    print(
        "  Reference run: standard 8 km topology, 120 s, ACK OFF vs ACK ON.\n"
        "  These are the numbers for the viva graph.\n"
    )

    r5_off = run_mesh_simulation(120.0, RNG_SEED, ack_enabled=False)
    r5_on  = run_mesh_simulation(120.0, RNG_SEED, ack_enabled=True)

    print(f"\n    {'Metric':<36} {'ACK OFF':>10}  {'ACK ON':>10}")
    print(f"    {'-'*58}")
    print(f"    {'SUB_NEAR sent':<36} {r5_off['sent_sub_near']:>10}  {r5_on['sent_sub_near']:>10}")
    print(f"    {'SUB_NEAR PDR':<36} {r5_off['pdr_sub_near']:>9.1f}%  {r5_on['pdr_sub_near']:>9.1f}%")
    print(f"    {'SUB_FAR sent':<36} {r5_off['sent_sub_far']:>10}  {r5_on['sent_sub_far']:>10}")
    print(f"    {'SUB_FAR PDR':<36} {r5_off['pdr_sub_far']:>9.1f}%  {r5_on['pdr_sub_far']:>9.1f}%")
    print(f"    {'Router duty cycle':<36} {r5_off['router_duty_cycle_pct']:>9.2f}%  {r5_on['router_duty_cycle_pct']:>9.2f}%")

    s5_pass = (
        r5_off["sent_sub_near"] > 0 and
        r5_off["sent_sub_far"]  > 0 and
        r5_on["sent_sub_far"]   > 0
    )
    print(f"\n  Section 5: {'PASS — PDR figures generated ✓' if s5_pass else 'FAIL — no packets sent'}")

    # =========================================================================
    # SECTION 6 — Collision regression
    # =========================================================================
    _header("SECTION 6 — Collision regression")
    print(
        "  Force CONTROL, ROUTER, and SUB_NEAR to all start TX at the same\n"
        "  instant.  Module 3's collision detector must flag all three.\n"
    )

    s6_pass = _section_6_collision_regression()
    print(f"\n  Section 6: {'PASS — all three simultaneous TXs correctly collide ✓' if s6_pass else 'FAIL'}")

    # =========================================================================
    # SECTION 7 — Router duty cycle
    # =========================================================================
    _header("SECTION 7 — Router duty cycle (120 s, ACK ON)")
    print(
        "  ROUTER forwards SUB_FAR data, sends ACKs, and emits beacons.\n"
        "  Total airtime must stay below the 10% ISM-band duty-cycle limit.\n"
    )

    # Reuse Section 5 ACK ON results
    dc = r5_on["router_duty_cycle_pct"]
    at = r5_on["router_airtime_s"]

    _result("Router total airtime",    f"{at:.3f}", " s")
    _result("Router duty cycle",       f"{dc:.2f}", " %")
    _result("ISM-band limit",          "10.00", " %")

    DUTY_CYCLE_LIMIT_PCT = 10.0
    if dc >= DUTY_CYCLE_LIMIT_PCT:
        print(f"\n  *** WARNING: Router duty cycle {dc:.2f}% exceeds 10% limit ***")

    s7_pass = dc < DUTY_CYCLE_LIMIT_PCT
    print(f"\n  Section 7: {'PASS — router airtime within 10% duty-cycle limit ✓' if s7_pass else 'FAIL — duty cycle too high'}")

    # =========================================================================
    # OVERALL
    # =========================================================================
    _sep()
    all_pass = s1_pass and s2_pass and s3_pass and s4_pass and s5_pass and s6_pass and s7_pass
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'SOME SECTIONS FAILED'}")
    _sep()
    print()
    print("Module 4 complete — ready for Module 5 (multi-hop routing enhancements)")
