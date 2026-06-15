"""
LoRa Mesh Sandbox — Module 2 Orchestrator
==========================================
Author: Simon Henson (with Claude)
Project: LoRa Mesh Network Water Pump Monitoring & Control System

This is the top-level script that wires Modules 2a–2c together into a
running simulation.  It owns:

    1. Configuration — channel parameters, node IDs, simulation duration
    2. Pre-flight check — link budget printout so you know what to expect
       before a single simulated packet flies
    3. Tick loop — advances simulated time, asks the subscriber whether it
       needs to transmit, passes the packet through the channel, and delivers
       the bytes (or reports a loss) to the control node
    4. Summary report — PDR, average RSSI/SNR, per-node stats

Run:
    python run_module2.py

Expected output for 2 km, 60 s:  ~12 heartbeats, PDR close to 100%
(2 km is well within the reliable zone; fades are cosmetic at this range).
"""

import math
from typing import List, Tuple

# Our four modules — each in its own file in this directory
from physics import link_budget, time_on_air_ms
from channel import RadioChannel, Reception
from packets import Packet, PacketType, decode_temperature_payload
from nodes import LoRaNode, NodeRole, TemperatureSensor


# =============================================================================
# SIMULATION CONFIGURATION
# All design parameters in ONE place. If you want to experiment (closer,
# further, different SF), change them here and re-run — everything else
# adjusts automatically.
# =============================================================================
NODE_CONTROL_ID    = 1           # The "control room" Heltec
NODE_SUBSCRIBER_ID = 2           # The "pump shed" Heltec

SEPARATION_M           = 2_000.0   # Point-to-point separation (metres)
SIM_DURATION_S         = 60.0      # How long to simulate (seconds)
SIM_TICK_S             = 0.05      # Time step — small enough not to miss events

# LoRa / Channel parameters — must match hardware design
SF                     = 9
BW_HZ                  = 125_000
CR                     = 1         # CR 4/5 = 1 (Semtech convention)
TX_POWER_DBM           = 14.0
ANTENNA_GAIN_DBI       = 2.0
CABLE_LOSS_DB          = 1.0
PATH_LOSS_EXPONENT     = 2.7       # Rural Karoo
SHADOW_FADING_STD_DB   = 6.0       # Typical outdoor sigma
FREQUENCY_MHZ          = 433.0

HEARTBEAT_INTERVAL_S   = 5.0
BATTERY_INITIAL_MV     = 4150

# Seed the RNG for reproducible results.
# Change this to see different random fading sequences.
RNG_SEED = 42


# =============================================================================
# LOGGING SETUP
# All log lines go through one function so the output is consistent.
# We buffer everything so we can print the summary at the very end.
# =============================================================================
_log_lines = []    # type: List[str]  — buffer, every logged line ends up here

def log(msg: str) -> None:
    """Print msg immediately and store it for the end-of-run buffer."""
    print(msg)
    _log_lines.append(msg)


def section(title: str) -> None:
    """Print a clearly visible section header."""
    line = "=" * 70
    log(f"\n{line}")
    log(f"  {title}")
    log(line)


# =============================================================================
# STEP 1: PRE-FLIGHT LINK BUDGET
# Before any simulation runs, show the theoretical numbers for this scenario.
# This is how you'd sanity-check a link plan on a real deployment.
# =============================================================================
def print_preflight_report() -> None:
    section("PRE-FLIGHT LINK BUDGET — 2 km rural, SF9, +14 dBm")

    # --- Time-on-Air for our packet size ---
    # Temperature heartbeat: 8 header + 6 payload + 2 CRC = 16 bytes
    # We compute this upfront so we can quote it in the trace.
    typical_packet_bytes = 16   # 8 header + 6 temperature payload + 2 CRC
    toa = time_on_air_ms(
        payload_bytes=typical_packet_bytes,
        sf=SF,
        bw_hz=BW_HZ,
        cr=CR,
    )
    log(f"\n  Packet size:    {typical_packet_bytes} bytes  "
        f"(header 8 + payload 6 + CRC 2)")
    log(f"  Time-on-Air:    {toa['t_total_ms']:.2f} ms  "
        f"(preamble {toa['t_preamble_ms']:.1f} ms + "
        f"payload {toa['t_payload_ms']:.1f} ms)")
    duty_cycle_pct = (toa['t_total_ms'] / 1000.0) / HEARTBEAT_INTERVAL_S * 100.0
    log(f"  Duty cycle:     {duty_cycle_pct:.2f}%  "
        f"(at {HEARTBEAT_INTERVAL_S}s heartbeat interval, <10% = OK)")

    # --- Link budget at the simulation distance ---
    lb = link_budget(
        distance_m=SEPARATION_M,
        sf=SF,
        tx_power_dbm=TX_POWER_DBM,
        tx_antenna_gain_dbi=ANTENNA_GAIN_DBI,
        rx_antenna_gain_dbi=ANTENNA_GAIN_DBI,
        cable_loss_db=CABLE_LOSS_DB,
        frequency_mhz=FREQUENCY_MHZ,
        path_loss_exponent=PATH_LOSS_EXPONENT,
    )
    log(f"\n  At {SEPARATION_M/1000:.1f} km (deterministic, no fading):")
    log(f"    Path loss:         {lb.path_loss_db:>7.2f} dB")
    log(f"    Received power:    {lb.received_power_dbm:>7.2f} dBm")
    log(f"    Sensitivity:       {lb.sensitivity_dbm:>7.2f} dBm  (SF9, BW125)")
    log(f"    Link margin:       {lb.link_margin_db:>+7.2f} dB  "
        f"({'reliable' if lb.reliable else 'marginal'})")
    log(f"    SNR estimate:      {lb.snr_estimate_db:>7.2f} dB")
    log(f"    SNR threshold:     {lb.snr_threshold_db:>7.2f} dB  (SF9 demod floor)")
    log(f"    SNR margin:        {lb.snr_margin_db:>+7.2f} dB")
    decode_word = "YES — will decode" if lb.decodable else "NO — below threshold"
    log(f"    Decodable?         {decode_word}")

    log(f"\n  Shadow fading sigma = {SHADOW_FADING_STD_DB} dB")
    log(f"  With {SHADOW_FADING_STD_DB} dB sigma and {lb.link_margin_db:.1f} dB margin,")
    log(f"  a {SHADOW_FADING_STD_DB}-sigma fade ({SHADOW_FADING_STD_DB} dB) would still leave "
        f"{lb.link_margin_db - SHADOW_FADING_STD_DB:.1f} dB margin.")
    log(f"  Expect near-100% PDR at 2 km. This is correct — 2 km is well")
    log(f"  inside the ~12.7 km reliable range calculated in Module 1.\n")


# =============================================================================
# STEP 2: BUILD THE CHANNEL AND NODES
# =============================================================================
def build_simulation():
    """
    Construct the channel and both nodes.

    Returns (channel, control_node, subscriber_node).
    We seed everything from RNG_SEED so results are reproducible.
    """
    # RadioChannel owns all the physics: path loss, fading, SNR decision
    channel = RadioChannel(
        path_loss_exponent=PATH_LOSS_EXPONENT,
        shadow_fading_std_db=SHADOW_FADING_STD_DB,
        sf=SF,
        tx_power_dbm=TX_POWER_DBM,
        tx_gain_dbi=ANTENNA_GAIN_DBI,
        rx_gain_dbi=ANTENNA_GAIN_DBI,
        cable_loss_db=CABLE_LOSS_DB,
        noise_floor_dbm=-124.0,
        seed=RNG_SEED,
    )

    # TemperatureSensor is a mock — will be replaced by real GPIO/serial later
    sensor = TemperatureSensor(
        baseline_c=22.0,           # Karoo ambient temperature
        diurnal_amplitude_c=5.0,   # ±5°C over 24h cycle
        noise_std_c=0.3,           # per-reading jitter
        seed=RNG_SEED + 1,         # different seed from channel RNG
    )

    # Subscriber sits at the pump shed, sends DATA heartbeats
    subscriber = LoRaNode(
        node_id=NODE_SUBSCRIBER_ID,
        role=NodeRole.SUBSCRIBER,
        position_m=(SEPARATION_M, 0.0),   # 2 km east of control
        heartbeat_interval_s=HEARTBEAT_INTERVAL_S,
        peer_id=NODE_CONTROL_ID,
        battery_mv_initial=BATTERY_INITIAL_MV,
        sensor=sensor,
        log_callback=log,
    )

    # Control sits at the control room, receives and processes packets
    control = LoRaNode(
        node_id=NODE_CONTROL_ID,
        role=NodeRole.CONTROL,
        position_m=(0.0, 0.0),
        peer_id=NODE_SUBSCRIBER_ID,
        log_callback=log,
    )

    return channel, control, subscriber


# =============================================================================
# STEP 3: TICK LOOP
# The heart of the simulation.  Real LoRa operates asynchronously, but in the
# sandbox we discretize time into SIM_TICK_S steps.  Each step we:
#   a) Ask the subscriber if it wants to transmit
#   b) If yes, push the serialized packet through the channel
#   c) If the channel decoded it, deliver the bytes to control
#   d) Log the channel outcome whether it succeeded or failed
#
# We track a list of Reception objects so we can compute aggregate stats.
# =============================================================================
def run_tick_loop(channel, control, subscriber):
    # type: (RadioChannel, LoRaNode, LoRaNode) -> List[Reception]
    """
    Run the main simulation loop for SIM_DURATION_S seconds.

    Returns a list of Reception objects (one per transmitted packet) so the
    caller can compute stats without re-traversing the log.
    """
    section(
        f"SIMULATION START  —  {SIM_DURATION_S:.0f}s  @  "
        f"{SEPARATION_M/1000:.1f} km  (seed={RNG_SEED})"
    )

    receptions = []   # type: List[Reception]  — one entry per transmitted packet

    # We use a float step loop.  Because floating point addition accumulates
    # error over thousands of steps, we derive sim_time from an integer tick
    # counter multiplied by the step size — this keeps it exact.
    total_ticks = int(SIM_DURATION_S / SIM_TICK_S)

    for tick in range(total_ticks + 1):
        sim_time_s = round(tick * SIM_TICK_S, 6)  # exact, no drift

        # --- Ask subscriber if it's time to transmit ---
        pkt = subscriber.tick_subscriber(sim_time_s)

        if pkt is None:
            # Nothing to do this tick — the subscriber is sleeping
            continue

        # --- Packet is ready: serialize and send through channel ---
        # to_bytes() gives us the exact bytes the radio would put on-air,
        # including the 2-byte CRC-16 trailer.
        raw_bytes = pkt.to_bytes()

        # channel.transmit() runs the link budget + random fading math
        # and returns a Reception describing what the receiver would see.
        reception = channel.transmit(SEPARATION_M)
        receptions.append(reception)

        if reception.decoded:
            # Channel says the packet would have demodulated successfully —
            # hand the raw bytes to the control node, which will:
            #   1. Parse the header and CRC
            #   2. Check it's addressed to itself
            #   3. Decode the temperature payload
            #   4. Log the reading with RSSI/SNR
            log(
                f"          └─ CHANNEL  "
                f"RSSI={reception.rssi_dbm:+.1f} dBm  "
                f"SNR={reception.snr_db:+.1f} dB  "
                f"margin={reception.margin_db:+.1f} dB  "
                f"→  DECODED ✓"
            )
            control.receive(raw_bytes, sim_time_s, reception.rssi_dbm, reception.snr_db)

        else:
            # Channel says the packet was lost — log why but don't deliver.
            # The control node never knows this happened (same as real life —
            # a receiver can't log what it never received).
            reason_map = {
                "below_sensitivity": "signal below sensitivity floor",
                "snr_too_low":       "SNR below demodulator threshold",
                "random_per":        "near-threshold probabilistic loss",
            }
            reason_str = reason_map.get(reception.failure_reason, reception.failure_reason)
            log(
                f"          └─ CHANNEL  "
                f"RSSI={reception.rssi_dbm:+.1f} dBm  "
                f"SNR={reception.snr_db:+.1f} dB  "
                f"margin={reception.margin_db:+.1f} dB  "
                f"→  LOST ✗  ({reason_str})"
            )

    return receptions


# =============================================================================
# STEP 4: SUMMARY REPORT
# Print the numbers you'd put in your logbook.
# =============================================================================
def print_summary(control, subscriber, receptions):
    # type: (LoRaNode, LoRaNode, List[Reception]) -> None
    section("SIMULATION SUMMARY")

    # Pull stats from both nodes
    sent         = subscriber.stats.packets_sent
    rx_at_ctrl   = control.stats.packets_received    # raw deliveries from channel
    decoded_ctrl = control.stats.packets_decoded      # packets that fully validated
    rejected     = control.stats.packets_rejected     # wrong CRC / not-for-us

    # PDR: Packet Delivery Ratio = fraction of sent packets correctly received
    # This is THE number that defines link quality in IoT deployments.
    pdr = (decoded_ctrl / sent * 100.0) if sent > 0 else 0.0

    # Average RSSI and SNR — only over packets that the channel decoded
    decoded_receptions = [r for r in receptions if r.decoded]
    if decoded_receptions:
        avg_rssi = sum(r.rssi_dbm for r in decoded_receptions) / len(decoded_receptions)
        avg_snr  = sum(r.snr_db  for r in decoded_receptions) / len(decoded_receptions)
        min_rssi = min(r.rssi_dbm for r in decoded_receptions)
        max_rssi = max(r.rssi_dbm for r in decoded_receptions)
    else:
        avg_rssi = avg_snr = min_rssi = max_rssi = 0.0

    # Channel-level loss (before CRC / address filtering)
    channel_decoded = len(decoded_receptions)
    channel_lost    = sent - channel_decoded

    log(f"\n  Node {subscriber.id} (SUBSCRIBER):")
    log(f"    Packets sent:              {sent}")
    log(f"    Final battery:             {subscriber.battery_mv} mV  "
        f"(started at {BATTERY_INITIAL_MV} mV, drained {BATTERY_INITIAL_MV - subscriber.battery_mv} mV)")

    log(f"\n  Channel:")
    log(f"    Decoded by channel:        {channel_decoded} / {sent}")
    log(f"    Lost by channel:           {channel_lost}  "
        f"({'none' if channel_lost == 0 else 'see reasons in trace above'})")

    log(f"\n  Node {control.id} (CONTROL):")
    log(f"    Bytes delivered:           {rx_at_ctrl} deliveries")
    log(f"    Packets correctly decoded: {decoded_ctrl}")
    log(f"    Packets rejected:          {rejected}  "
        f"(CRC fail / wrong address / duplicate)")

    log(f"\n  LINK STATISTICS:")
    log(f"    PDR (Packet Delivery Ratio): {pdr:.1f}%")
    if decoded_receptions:
        log(f"    Average RSSI:                {avg_rssi:+.1f} dBm")
        log(f"    Average SNR:                 {avg_snr:+.1f} dB")
        log(f"    RSSI range:                  {min_rssi:+.1f} to {max_rssi:+.1f} dBm  "
            f"(spread = {max_rssi - min_rssi:.1f} dB, caused by shadow fading)")

    # --- Sanity verdict ---
    log(f"\n  VERDICT:")
    if pdr >= 99.0:
        log(f"    EXCELLENT  — {pdr:.1f}% PDR at {SEPARATION_M/1000:.1f} km confirms")
        log(f"    the 2 km link is deep inside the reliable zone.")
    elif pdr >= 90.0:
        log(f"    GOOD  — {pdr:.1f}% PDR, occasional fades but link is solid.")
    elif pdr >= 70.0:
        log(f"    MARGINAL  — {pdr:.1f}% PDR.  Check distance / SF / antenna.")
    else:
        log(f"    POOR  — {pdr:.1f}% PDR.  Link budget has a problem. Investigate.")

    log(f"\n  Module 2 complete — ready for Module 3 (aggregation / relay).")
    log("=" * 70 + "\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    # --- Pre-flight sanity check ---
    # Confirm Python can import all our modules before wasting time
    # in the loop only to crash on a missing import.
    print("LoRa Mesh Sandbox — Module 2 End-to-End Test")
    print(f"Configuration: {SEPARATION_M/1000:.1f} km, {SIM_DURATION_S:.0f}s sim, "
          f"SF{SF}, {HEARTBEAT_INTERVAL_S}s heartbeat")

    # 1. Print the theoretical numbers
    print_preflight_report()

    # 2. Build channel + nodes
    channel, control, subscriber = build_simulation()

    # 3. Run the simulation
    receptions = run_tick_loop(channel, control, subscriber)

    # 4. Print summary
    print_summary(control, subscriber, receptions)
