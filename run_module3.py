"""
LoRa Mesh Sandbox — Module 3 Orchestrator
==========================================
Author: Simon Henson (with Claude)
Project: LoRa Mesh Network Water Pump Monitoring & Control System

Replaces the Module 2 tick loop with a Discrete-Event Simulator (DES).
The DES jumps directly to each event (heartbeat timer, TX_START, TX_END)
instead of iterating through thousands of empty time steps — this makes
distance sweep runs fast enough to be interactive.

Four sections:

    Section 1 — Equivalence check
        Reproduce the Module 2 scenario (60 s, 2 km, seed=42) using the
        DES engine and confirm the output matches the known reference.

    Section 2 — Distance sweep
        Run 300 s simulations at [1, 2, 5, 8, 10, 12, 13, 15] km and print
        a PDR vs distance table. PDR should cliff between 12 and 13 km,
        confirming Module 1's 12.7 km reliable-range prediction.

    Section 3 — Synthetic collision test
        Force two subscribers to transmit simultaneously and verify that
        channel_v3 reports 'collision' for both.

    Section 4 — Clean two-node regression
        Confirm that the standard one-subscriber scenario produces zero
        false-positive collisions.

Run:
    python run_module3.py
"""

import random
from typing import Dict, List, Optional, Tuple

# ---- Module 1 ----
from physics import link_budget, time_on_air_ms

# ---- Module 2 (unchanged) ----
from packets import Packet, PacketType, decode_temperature_payload
from nodes import LoRaNode, NodeRole, TemperatureSensor

# ---- Module 3 (new) ----
from scheduler import Scheduler, Event, TX_START, TX_END, TIMER_FIRE, LOG_VERBOSE, LOG_SUMMARY
from channel_v3 import RadioChannelV3, Reception


# =============================================================================
# SHARED DESIGN CONSTANTS
# One place to change them for the whole file.
# =============================================================================
NODE_CONTROL_ID    = 1
NODE_SUBSCRIBER_ID = 2

SF                   = 9
BW_HZ                = 125_000
CR                   = 1          # CR 4/5
TX_POWER_DBM         = 14.0
ANTENNA_GAIN_DBI     = 2.0
CABLE_LOSS_DB        = 1.0
PATH_LOSS_EXPONENT   = 2.7
SHADOW_FADING_STD_DB = 6.0
FREQUENCY_MHZ        = 433.0
HEARTBEAT_INTERVAL_S = 5.0
BATTERY_INITIAL_MV   = 4150
NOISE_FLOOR_DBM      = -124.0

# RNG seed — used for both global random (node boot offset) and channel private RNG
RNG_SEED = 42

# Temperature sensor seed (different from channel so they don't interfere)
SENSOR_SEED = RNG_SEED + 1   # = 43

# Module 2 reference numbers (from the confirmed run_module2.py output)
M2_EXPECTED_SENT     = 12
M2_EXPECTED_PDR      = 100.0    # percent
M2_EXPECTED_AVG_RSSI = -97.0   # dBm  (tolerance applied in check)
M2_RSSI_TOLERANCE    = 0.5     # ±0.5 dBm is "equal" for float comparison


# =============================================================================
# HELPER: compute Time-on-Air for a given packet size
# Encapsulated here so we don't repeat the physics call everywhere.
# =============================================================================
def toa_seconds(packet_size_bytes: int) -> float:
    """
    Return packet Time-on-Air in seconds for our standard SF9/BW125 config.

    For a 16-byte temperature heartbeat:
        ToA ≈ 164.86 ms = 0.16486 s
    """
    return time_on_air_ms(
        payload_bytes=packet_size_bytes,
        sf=SF,
        bw_hz=BW_HZ,
        cr=CR,
    )["t_total_ms"] / 1000.0


# =============================================================================
# CORE SIMULATION FUNCTION
# Used by all four sections.  Returns summary stats so each section can
# check / display them without duplicating the DES loop.
# =============================================================================
def run_des_simulation(
    distance_m: float,
    sim_duration_s: float,
    rng_seed: int,
    log_fn=None,           # callable(str) for event-level logging; None = silent
    extra_subscribers: Optional[List[LoRaNode]] = None,  # for collision tests
    channel_override: Optional[RadioChannelV3] = None,   # inject pre-built channel
) -> Dict:
    """
    Run a single DES simulation and return summary statistics.

    Args:
        distance_m         — TX-to-RX separation (metres)
        sim_duration_s     — how long to simulate (seconds)
        rng_seed           — seeds both global random and channel private RNG
        log_fn             — if provided, called with each log string
        extra_subscribers  — additional LoRaNode(SUBSCRIBER) instances to include
                             (used by Section 3 collision test)
        channel_override   — use this channel instead of creating a fresh one
                             (useful when the caller needs to inspect in_flight
                             after the run)

    Returns a dict:
        sent         — total packets transmitted by all subscribers
        decoded      — packets successfully received by control node
        pdr          — Packet Delivery Ratio (%)
        collisions   — number of packets lost to collision
        receptions   — list of Reception objects (one per transmitted packet)
        avg_rssi     — mean RSSI of decoded packets (dBm), or 0.0 if none
        avg_snr      — mean SNR of decoded packets (dB), or 0.0 if none
    """
    def emit(msg: str) -> None:
        """Internal log helper — forwards to log_fn if set."""
        if log_fn is not None:
            log_fn(msg)

    # -------------------------------------------------------------------------
    # Seed the global random so the subscriber's boot offset is reproducible.
    # In Module 2 the global random was NOT seeded (the boot offset came from
    # whatever state Python's random was in). Here we seed it so every run
    # with the same rng_seed gives the same boot offset → reproducible sweeps.
    # The channel's private rng is random.Random(rng_seed), independent of this.
    # -------------------------------------------------------------------------
    random.seed(rng_seed)

    # Build channel (or use the injected one)
    if channel_override is not None:
        channel = channel_override
    else:
        channel = RadioChannelV3(
            path_loss_exponent=PATH_LOSS_EXPONENT,
            shadow_fading_std_db=SHADOW_FADING_STD_DB,
            sf=SF,
            tx_power_dbm=TX_POWER_DBM,
            tx_gain_dbi=ANTENNA_GAIN_DBI,
            rx_gain_dbi=ANTENNA_GAIN_DBI,
            cable_loss_db=CABLE_LOSS_DB,
            noise_floor_dbm=NOISE_FLOOR_DBM,
            seed=rng_seed,
        )

    # Build the temperature sensor for the primary subscriber
    sensor = TemperatureSensor(
        baseline_c=22.0,
        diurnal_amplitude_c=5.0,
        noise_std_c=0.3,
        seed=SENSOR_SEED,
    )

    # Primary subscriber (pump shed) — _boot() draws from global random here
    # to set next_heartbeat_at = random.uniform(0.5, 1.5)
    primary_sub = LoRaNode(
        node_id=NODE_SUBSCRIBER_ID,
        role=NodeRole.SUBSCRIBER,
        position_m=(distance_m, 0.0),
        heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
        peer_id=NODE_CONTROL_ID,
        battery_mv_initial=BATTERY_INITIAL_MV,
        sensor=sensor,
        log_callback=log_fn,   # None silences the node's own logs
    )

    # Control node (control room)
    control = LoRaNode(
        node_id=NODE_CONTROL_ID,
        role=NodeRole.CONTROL,
        position_m=(0.0, 0.0),
        peer_id=NODE_SUBSCRIBER_ID,
        log_callback=log_fn,
    )

    # Collect all subscribers (primary + any injected extras)
    all_subscribers = [primary_sub] + (extra_subscribers or [])

    # -------------------------------------------------------------------------
    # DES log level: verbose if a log function was provided, summary otherwise.
    # The DES will print event dispatch lines only at LOG_VERBOSE.
    # -------------------------------------------------------------------------
    des_log_level = LOG_VERBOSE if log_fn is not None else LOG_SUMMARY
    sched = Scheduler(log_level=des_log_level)

    # Track all Reception objects for stats
    receptions: List[Reception] = []
    collision_count = 0

    # -------------------------------------------------------------------------
    # Dispatch function — this closure is the "brain" of the simulation.
    # It handles all three event types: TIMER_FIRE, TX_START, TX_END.
    # -------------------------------------------------------------------------
    def dispatch(evt: Event) -> None:
        nonlocal collision_count

        # ---- TIMER_FIRE: heartbeat interval has elapsed ----
        if evt.event_type == TIMER_FIRE:
            sub: LoRaNode = evt.payload["subscriber"]
            pkt: Optional[Packet] = sub.tick_subscriber(evt.time)
            if pkt is None:
                # Shouldn't happen if scheduling is correct, but handle gracefully
                return

            raw_bytes = pkt.to_bytes()
            toa_s = toa_seconds(pkt.size_bytes())
            tx_start_t = evt.time
            tx_end_t = tx_start_t + toa_s

            # Schedule TX_START at delay=0 (same simulated time).
            # In a real LoRa system TX begins immediately after the app hands
            # off the packet to the radio driver.
            sched.schedule(0.0, TX_START, {
                "raw_bytes":  raw_bytes,
                "node_id":    sub.id,
                "tx_start":   tx_start_t,
                "tx_end":     tx_end_t,
                "distance_m": distance_m,
            })

            # Schedule next heartbeat — use the interval directly as delay.
            # sub.next_heartbeat_at was set inside tick_subscriber() to
            # evt.time + HEARTBEAT_INTERVAL_S, so the delay is exactly 5.0 s.
            next_delay = sub.next_heartbeat_at - evt.time
            sched.schedule(next_delay, TIMER_FIRE, {"subscriber": sub})

        # ---- TX_START: radio begins transmitting ----
        elif evt.event_type == TX_START:
            # Register this transmission in the channel so the collision
            # detector can check for overlaps with other in-flight packets.
            channel.register_tx(
                tx_start=evt.payload["tx_start"],
                tx_end=evt.payload["tx_end"],
                node_id=evt.payload["node_id"],
                freq_mhz=FREQUENCY_MHZ,
                sf=SF,
                distance_m=evt.payload["distance_m"],
            )
            # Schedule TX_END at the end of the on-air window
            toa_s = evt.payload["tx_end"] - evt.payload["tx_start"]
            sched.schedule(toa_s, TX_END, evt.payload)

        # ---- TX_END: last symbol has left the antenna ----
        elif evt.event_type == TX_END:
            # Ask the channel what happened to this packet.
            # This is where the shadow fade is drawn (one gauss() call per packet),
            # matching the Module 2 behaviour where channel.transmit() was called
            # synchronously with the subscriber's tick.
            reception = channel.resolve_rx(
                tx_end_time=evt.payload["tx_end"],
                node_id=evt.payload["node_id"],
                freq_mhz=FREQUENCY_MHZ,
                sf=SF,
            )
            receptions.append(reception)

            if reception.failure_reason == "collision":
                collision_count += 1
                emit(
                    f"          └─ CHANNEL  "
                    f"→  COLLISION ✗  (two packets on air simultaneously)"
                )
            elif reception.decoded:
                emit(
                    f"          └─ CHANNEL  "
                    f"RSSI={reception.rssi_dbm:+.1f} dBm  "
                    f"SNR={reception.snr_db:+.1f} dB  "
                    f"margin={reception.margin_db:+.1f} dB  "
                    f"→  DECODED ✓"
                )
                control.receive(
                    evt.payload["raw_bytes"],
                    evt.time,
                    reception.rssi_dbm,
                    reception.snr_db,
                )
            else:
                reason_map = {
                    "below_sensitivity": "signal below sensitivity floor",
                    "snr_too_low":       "SNR below demodulator threshold",
                    "random_per":        "near-threshold probabilistic loss",
                }
                reason_str = reason_map.get(
                    reception.failure_reason, reception.failure_reason
                )
                emit(
                    f"          └─ CHANNEL  "
                    f"RSSI={reception.rssi_dbm:+.1f} dBm  "
                    f"SNR={reception.snr_db:+.1f} dB  "
                    f"margin={reception.margin_db:+.1f} dB  "
                    f"→  LOST ✗  ({reason_str})"
                )

    # -------------------------------------------------------------------------
    # Seed the event queue — one TIMER_FIRE per subscriber at their boot time.
    # The boot time was set by _boot() above using random.uniform(0.5, 1.5).
    # -------------------------------------------------------------------------
    for sub in all_subscribers:
        delay_to_first_tx = sub.next_heartbeat_at  # relative to t=0
        sched.schedule(delay_to_first_tx, TIMER_FIRE, {"subscriber": sub})

    # Run!
    sched.run_until(sim_duration_s, dispatch)

    # Compute aggregate stats
    sent    = sum(s.stats.packets_sent for s in all_subscribers)
    decoded = control.stats.packets_decoded
    pdr     = (decoded / sent * 100.0) if sent > 0 else 0.0

    decoded_rx = [r for r in receptions if r.decoded]
    avg_rssi = (
        sum(r.rssi_dbm for r in decoded_rx) / len(decoded_rx)
        if decoded_rx else 0.0
    )
    avg_snr = (
        sum(r.snr_db for r in decoded_rx) / len(decoded_rx)
        if decoded_rx else 0.0
    )

    return {
        "sent":       sent,
        "decoded":    decoded,
        "pdr":        pdr,
        "collisions": collision_count,
        "receptions": receptions,
        "avg_rssi":   avg_rssi,
        "avg_snr":    avg_snr,
    }


# =============================================================================
# SECTION 1 — EQUIVALENCE CHECK
# Reproduce Module 2's 60 s, 2 km scenario via the DES and confirm
# the stats match the known reference output.
# =============================================================================
def section1_equivalence_check() -> bool:
    print("=" * 70)
    print("  SECTION 1 — Equivalence check: DES vs Module 2 tick loop")
    print("=" * 70)
    print(
        f"\n  Scenario: {60}s sim, 2 km, SF{SF}, {HEARTBEAT_INTERVAL_S}s heartbeat, seed={RNG_SEED}"
    )
    print(f"  Expected: {M2_EXPECTED_SENT} packets, {M2_EXPECTED_PDR:.1f}% PDR, "
          f"avg RSSI ≈ {M2_EXPECTED_AVG_RSSI:.1f} dBm")
    print()

    stats = run_des_simulation(
        distance_m=2_000.0,
        sim_duration_s=60.0,
        rng_seed=RNG_SEED,
        log_fn=print,   # verbose: print every event
    )

    sent     = stats["sent"]
    pdr      = stats["pdr"]
    avg_rssi = stats["avg_rssi"]
    avg_snr  = stats["avg_snr"]

    print()
    print("-" * 70)
    print(f"  DES result:  sent={sent}  PDR={pdr:.1f}%  "
          f"avg RSSI={avg_rssi:+.1f} dBm  avg SNR={avg_snr:+.1f} dB")
    print()

    # Compare against Module 2 reference
    passed = True
    issues = []

    if sent != M2_EXPECTED_SENT:
        issues.append(f"sent {sent} ≠ expected {M2_EXPECTED_SENT}")
        passed = False

    if abs(pdr - M2_EXPECTED_PDR) > 0.1:
        issues.append(f"PDR {pdr:.1f}% ≠ expected {M2_EXPECTED_PDR:.1f}%")
        passed = False

    if abs(avg_rssi - M2_EXPECTED_AVG_RSSI) > M2_RSSI_TOLERANCE:
        issues.append(
            f"avg RSSI {avg_rssi:.1f} dBm outside tolerance "
            f"({M2_EXPECTED_AVG_RSSI:.1f} ± {M2_RSSI_TOLERANCE})"
        )
        passed = False

    if passed:
        print("  Section 1: PASS — DES output matches Module 2 reference ✓")
    else:
        print("  Section 1: FAIL")
        for issue in issues:
            print(f"    DIFF: {issue}")
    print()
    return passed


# =============================================================================
# SECTION 2 — DISTANCE SWEEP
# Run 300 s simulations across a range of distances and print a PDR table.
# Expected: near-100% PDR up to ~12 km, then a sharp cliff to near-0%.
# =============================================================================
def section2_distance_sweep() -> None:
    print("=" * 70)
    print("  SECTION 2 — Distance sweep (300 s, SF9, seed=42)")
    print("=" * 70)
    print(
        f"\n  {'Distance':>10}  {'Packets':>8}  {'PDR':>7}  "
        f"{'Avg RSSI':>10}  {'Avg SNR':>9}"
    )
    print(f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*9}")

    distances_km = [1, 2, 5, 8, 10, 12, 13, 15]

    for d_km in distances_km:
        stats = run_des_simulation(
            distance_m=d_km * 1000.0,
            sim_duration_s=300.0,
            rng_seed=RNG_SEED,
            log_fn=None,   # silent — batch run
        )

        sent     = stats["sent"]
        pdr      = stats["pdr"]
        avg_rssi = stats["avg_rssi"]
        avg_snr  = stats["avg_snr"]

        # Flag the cliff visually
        cliff_marker = "  ← cliff" if d_km == 13 and pdr < 50.0 else ""

        print(
            f"  {d_km:>7} km  {sent:>8}  {pdr:>6.1f}%  "
            f"{avg_rssi:>+9.1f}  {avg_snr:>+8.1f}{cliff_marker}"
        )

    print()
    print(
        "  Module 1 predicted reliable range ≈ 12.7 km.\n"
        "  PDR should cliff between 12 km and 13 km in the table above."
    )
    print()


# =============================================================================
# SECTION 3 — SYNTHETIC COLLISION TEST
# Force two subscribers to transmit at exactly the same time.
# Verify that channel_v3 marks both as 'collision'.
# =============================================================================
def section3_collision_test() -> bool:
    print("=" * 70)
    print("  SECTION 3 — Synthetic collision test")
    print("=" * 70)
    print(
        "\n  Two subscribers at 2 km both fire TX_START at t=1.15 s.\n"
        "  Packets overlap completely (identical timing). Both must be\n"
        "  flagged as 'collision' by channel_v3.resolve_rx().\n"
    )

    COLLISION_TIME = 1.15   # force both subscribers to fire at this time

    random.seed(RNG_SEED)

    # Build a fresh channel so it has a clean in_flight list
    channel = RadioChannelV3(seed=RNG_SEED)

    # Node A — primary subscriber (id=2)
    sub_a = LoRaNode(
        node_id=2,
        role=NodeRole.SUBSCRIBER,
        position_m=(2000.0, 0.0),
        heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
        peer_id=NODE_CONTROL_ID,
        battery_mv_initial=BATTERY_INITIAL_MV,
        sensor=None,   # temp=0.0 — fine for collision test
        log_callback=print,
    )

    # Node B — second subscriber at same distance (id=3)
    sub_b = LoRaNode(
        node_id=3,
        role=NodeRole.SUBSCRIBER,
        position_m=(2000.0, 0.0),
        heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
        peer_id=NODE_CONTROL_ID,
        battery_mv_initial=BATTERY_INITIAL_MV,
        sensor=None,
        log_callback=print,
    )

    # Override boot offsets so both fire at the collision time
    sub_a.next_heartbeat_at = COLLISION_TIME
    sub_b.next_heartbeat_at = COLLISION_TIME

    # The DES — verbose so you see all four events (2× TIMER_FIRE, 2× TX_START, 2× TX_END)
    sched = Scheduler(log_level=LOG_VERBOSE)
    sched.schedule(COLLISION_TIME, TIMER_FIRE, {"subscriber": sub_a})
    sched.schedule(COLLISION_TIME, TIMER_FIRE, {"subscriber": sub_b})

    # Track per-node resolve results
    resolve_results: Dict[int, Reception] = {}

    def dispatch_s3(evt: Event) -> None:
        if evt.event_type == TIMER_FIRE:
            sub: LoRaNode = evt.payload["subscriber"]
            pkt = sub.tick_subscriber(evt.time)
            if pkt is None:
                return
            raw = pkt.to_bytes()
            toa_s = toa_seconds(pkt.size_bytes())
            tx_start_t = evt.time
            tx_end_t   = tx_start_t + toa_s
            sched.schedule(0.0, TX_START, {
                "raw_bytes":  raw,
                "node_id":    sub.id,
                "tx_start":   tx_start_t,
                "tx_end":     tx_end_t,
                "distance_m": 2000.0,
            })

        elif evt.event_type == TX_START:
            channel.register_tx(
                tx_start=evt.payload["tx_start"],
                tx_end=evt.payload["tx_end"],
                node_id=evt.payload["node_id"],
                freq_mhz=FREQUENCY_MHZ,
                sf=SF,
                distance_m=evt.payload["distance_m"],
            )
            toa_s = evt.payload["tx_end"] - evt.payload["tx_start"]
            sched.schedule(toa_s, TX_END, evt.payload)

        elif evt.event_type == TX_END:
            rx = channel.resolve_rx(
                tx_end_time=evt.payload["tx_end"],
                node_id=evt.payload["node_id"],
                freq_mhz=FREQUENCY_MHZ,
                sf=SF,
            )
            resolve_results[evt.payload["node_id"]] = rx
            print(
                f"          └─ CHANNEL  node={evt.payload['node_id']}  "
                f"decoded={rx.decoded}  reason={rx.failure_reason}"
            )

    sched.run_until(5.0, dispatch_s3)

    # Check both packets were marked as collision
    both_collision = (
        len(resolve_results) == 2
        and all(r.failure_reason == "collision" for r in resolve_results.values())
    )

    print()
    for node_id, rx in sorted(resolve_results.items()):
        status = "✓ collision" if rx.failure_reason == "collision" else f"✗ {rx.failure_reason}"
        print(f"    Node {node_id}: {status}")
    print()
    if both_collision:
        print("  Section 3: PASS — both packets correctly flagged as collision ✓")
    else:
        print("  Section 3: FAIL — expected both packets to be 'collision'")
        for node_id, rx in sorted(resolve_results.items()):
            print(f"    Node {node_id}: decoded={rx.decoded}  reason={rx.failure_reason}")
    print()
    return both_collision


# =============================================================================
# SECTION 4 — CLEAN TWO-NODE REGRESSION
# Standard one-subscriber scenario — confirm zero false-positive collisions.
# =============================================================================
def section4_clean_regression() -> bool:
    print("=" * 70)
    print("  SECTION 4 — Clean two-node regression (no false-positive collisions)")
    print("=" * 70)
    print(
        f"\n  Scenario: {60}s, 2 km, single subscriber, seed={RNG_SEED}.\n"
        "  With only one transmitter there can never be a collision.\n"
        "  Any 'collision' result here would be a bug in channel_v3.\n"
    )

    stats = run_des_simulation(
        distance_m=2_000.0,
        sim_duration_s=60.0,
        rng_seed=RNG_SEED,
        log_fn=None,   # quiet — we only care about the collision count
    )

    sent       = stats["sent"]
    decoded    = stats["decoded"]
    collisions = stats["collisions"]
    pdr        = stats["pdr"]

    print(f"  Sent={sent}  Decoded={decoded}  PDR={pdr:.1f}%  Collisions={collisions}")
    print()

    if collisions == 0:
        print("  Section 4: PASS — zero false-positive collisions ✓")
    else:
        print(f"  Section 4: FAIL — {collisions} unexpected collision(s) detected")
    print()
    return collisions == 0


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print()
    print("LoRa Mesh Sandbox — Module 3")
    print("DES engine + collision detection")
    print()

    r1 = section1_equivalence_check()
    section2_distance_sweep()
    r3 = section3_collision_test()
    r4 = section4_clean_regression()

    overall = all([r1, r3, r4])
    print("=" * 70)
    print(f"  Overall: {'ALL PASS ✓' if overall else 'SOME FAILURES — see above'}")
    print("=" * 70)
    print()
    print("Module 3 complete — ready for Module 4 (multi-hop routing)")
