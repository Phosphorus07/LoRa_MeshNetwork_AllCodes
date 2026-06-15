"""
LoRa Mesh Sandbox — Module 1: Physics & Link Budget Engine
============================================================
Author: Simon Henson (with Claude)
Project: LoRa Mesh Network Water Pump Monitoring & Control System

This module implements the four core equations that govern any LoRa link:
    1. Time-on-Air (Semtech's exact formula from AN1200.13)
    2. Free Space Path Loss (FSPL) — best case, line of sight
    3. Log-Distance Path Loss — realistic with environmental factor n
    4. Link Budget — the master equation that decides if a packet decodes

Every equation here is referenced to a published source so you can cite it
in your report and defend it in your viva.

References:
    - Semtech SX1276/77/78/79 Datasheet, Rev 7, May 2020
    - Semtech AN1200.13: "LoRa Modem Designer's Guide"
    - Rappaport, T.S., "Wireless Communications: Principles & Practice", 2nd ed.
"""

import math
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# CONSTANTS — SX1278 datasheet values for SF7-SF12 at BW 125 kHz
# =============================================================================
# Receiver sensitivity in dBm. These are THE numbers that define LoRa's
# legendary range. Each step up in SF doubles symbol time but adds ~2.5 dB
# of link budget. That's the SF tradeoff in one sentence.
SX1278_SENSITIVITY_BW125 = {
    7: -123.0,
    8: -126.0,
    9: -129.0,   # <-- Your design choice
    10: -132.0,
    11: -133.0,  # below this point, low data rate optimisation kicks in
    12: -136.0,
}

# Demodulator SNR threshold per SF (Semtech datasheet Table 13)
# If received SNR > threshold, packet decodes. Below = lost.
# Near threshold there's a probabilistic transition zone.
SX1278_SNR_THRESHOLD = {
    7: -7.5,
    8: -10.0,
    9: -12.5,    # <-- Your design choice
    10: -15.0,
    11: -17.5,
    12: -20.0,
}


# =============================================================================
# 1. TIME-ON-AIR
# =============================================================================
def symbol_time_ms(sf: int, bw_hz: int) -> float:
    """
    LoRa symbol time: T_sym = 2^SF / BW

    For SF9, BW=125 kHz: 2^9 / 125000 = 4.096 ms

    This is the fundamental time unit of LoRa. Every other timing in the
    protocol is a multiple of this.
    """
    return (2 ** sf) / bw_hz * 1000.0  # convert seconds → ms


def time_on_air_ms(
    payload_bytes: int,
    sf: int = 9,
    bw_hz: int = 125_000,
    cr: int = 1,             # CR 4/5 = 1, CR 4/6 = 2, CR 4/7 = 3, CR 4/8 = 4
    preamble_symbols: int = 8,
    explicit_header: bool = True,
    crc_enabled: bool = True,
    low_data_rate_optimisation: Optional[bool] = None,
) -> dict:
    """
    Calculate Time-on-Air using Semtech's exact formula.

    Returns a dict with:
        t_sym_ms     — single symbol duration
        t_preamble   — preamble duration (header sync)
        t_payload    — payload+CRC duration
        t_total      — total airtime (the number you actually care about)
        n_payload    — number of payload symbols (useful for debugging)

    The formula is from Semtech AN1200.13 §4.1.1.6:

        n_payload = 8 + max(
            ceil( (8*PL - 4*SF + 28 + 16*CRC - 20*H) / (4*(SF - 2*DE)) )
            * (CR + 4),
            0
        )

    Where:
        PL  = payload length in bytes
        SF  = spreading factor
        CRC = 1 if CRC enabled, 0 otherwise
        H   = 0 if explicit header, 1 if implicit
        DE  = 1 if low data rate optimisation enabled, 0 otherwise
        CR  = coding rate setting (1 for 4/5, 2 for 4/6, 3 for 4/7, 4 for 4/8)

    Low data rate optimisation (DE) is mandatory when symbol time > 16 ms,
    which happens at SF11/SF12 with BW 125 kHz. We auto-detect this unless
    the caller forces a value.
    """
    t_sym = symbol_time_ms(sf, bw_hz)

    # Auto-enable LDRO when symbol time exceeds 16 ms (Semtech recommendation)
    if low_data_rate_optimisation is None:
        de = 1 if t_sym > 16.0 else 0
    else:
        de = 1 if low_data_rate_optimisation else 0

    h = 0 if explicit_header else 1
    crc = 1 if crc_enabled else 0
    pl = payload_bytes

    # Preamble: standard formula adds 4.25 symbols for sync
    t_preamble = (preamble_symbols + 4.25) * t_sym

    # Payload symbol count — the ugly equation, broken into pieces for clarity
    numerator = 8 * pl - 4 * sf + 28 + 16 * crc - 20 * h
    denominator = 4 * (sf - 2 * de)
    payload_symbols_raw = math.ceil(numerator / denominator) * (cr + 4)
    n_payload = 8 + max(payload_symbols_raw, 0)

    t_payload = n_payload * t_sym
    t_total = t_preamble + t_payload

    return {
        "t_sym_ms": t_sym,
        "t_preamble_ms": t_preamble,
        "t_payload_ms": t_payload,
        "t_total_ms": t_total,
        "n_payload_symbols": n_payload,
        "low_data_rate_opt": bool(de),
    }


# =============================================================================
# 2 & 3. PATH LOSS MODELS
# =============================================================================
def fspl_db(distance_m: float, frequency_mhz: float = 433.0) -> float:
    """
    Free Space Path Loss — the theoretical best case.

    FSPL(dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.44

    This assumes:
        - perfect line of sight
        - no ground reflection
        - no atmosphere absorption
        - far-field conditions (d >> wavelength)

    Use this as your floor — real path loss is ALWAYS worse.

    Quick sanity numbers for 433 MHz:
        100 m  → 65.2 dB
        1 km   → 85.2 dB
        5 km   → 99.2 dB
        10 km  → 105.2 dB
    """
    if distance_m < 1.0:
        # Avoid log10(0) — at sub-metre distances the far-field
        # assumption breaks anyway, so clamp.
        distance_m = 1.0

    d_km = distance_m / 1000.0
    return 20.0 * math.log10(d_km) + 20.0 * math.log10(frequency_mhz) + 32.44


def log_distance_path_loss_db(
    distance_m: float,
    frequency_mhz: float = 433.0,
    path_loss_exponent: float = 2.7,
    reference_distance_m: float = 1.0,
    shadow_fading_db: float = 0.0,
) -> float:
    """
    Log-Distance Path Loss — realistic for terrestrial links.

    PL(d) = PL(d_0) + 10*n*log10(d/d_0) + X_sigma

    The exponent n captures environment:
        n = 2.0    : free space
        n = 2.7-3.0: rural with sparse vegetation (Karoo!)
        n = 3.0-3.5: suburban
        n = 3.5-5.0: urban / heavy obstruction
        n = 4.0-6.0: in-building

    For your Excelsior deployment, n = 2.7-3.0 is the defensible starting
    point. We can refine this once you have real RSSI measurements from
    the two-node hardware test and back-calculate n from the data.

    X_sigma is a Gaussian random variable representing shadow fading
    (large-scale variation from terrain features, vegetation patches, etc.).
    Pass 0 for deterministic results; the simulator will inject random
    values from a normal distribution for Monte Carlo runs.
    """
    pl_d0 = fspl_db(reference_distance_m, frequency_mhz)
    if distance_m <= reference_distance_m:
        return pl_d0 + shadow_fading_db

    return (
        pl_d0
        + 10.0 * path_loss_exponent * math.log10(distance_m / reference_distance_m)
        + shadow_fading_db
    )


# =============================================================================
# 4. LINK BUDGET — the master equation
# =============================================================================
@dataclass
class LinkBudgetResult:
    """All the numbers you need to defend a link in your report."""
    distance_m: float
    path_loss_db: float
    received_power_dbm: float
    sensitivity_dbm: float
    link_margin_db: float
    snr_estimate_db: float
    snr_threshold_db: float
    snr_margin_db: float
    decodable: bool          # link margin > 0 AND snr margin > 0
    reliable: bool           # margin > 10 dB (industry rule of thumb)


def link_budget(
    distance_m: float,
    sf: int = 9,
    tx_power_dbm: float = 14.0,
    tx_antenna_gain_dbi: float = 2.0,
    rx_antenna_gain_dbi: float = 2.0,
    cable_loss_db: float = 1.0,
    frequency_mhz: float = 433.0,
    path_loss_exponent: float = 2.7,
    use_fspl: bool = False,
    noise_floor_dbm: float = -124.0,  # thermal noise + NF for 125 kHz BW
) -> LinkBudgetResult:
    """
    Compute the complete link budget for a single link.

    The master equation:
        P_rx = P_tx + G_tx + G_rx - PL - L_cable

    Then:
        Link Margin   = P_rx - Sensitivity
        SNR (estimate) = P_rx - Noise Floor
        SNR Margin    = SNR - SNR_threshold(SF)

    A packet decodes if Link Margin > 0 AND SNR Margin > 0.

    Default tx_power = 14 dBm: this is the SX1278's standard +14 dBm setting
    on the RFO pin. The chip can do +20 dBm on the PA_BOOST pin, but at
    433 MHz European/SA regulations on this ISM band typically cap effective
    radiated power. We'll stick with 14 dBm for now and you can revisit.

    Default antenna gain 2 dBi: typical for a small 433 MHz quarter-wave
    whip antenna. If you upgrade to a Yagi at the control room, that
    becomes 8-12 dBi and your link budget improves dramatically — worth
    exploring in the simulator!

    Noise floor at -124 dBm: thermal noise (kTB) for 125 kHz BW at 290 K
    is -120.9 dBm. Add a typical 3 dB receiver noise figure → -117.9 dBm.
    But LoRa decodes BELOW the noise floor (that's the magic), so we use
    the effective floor relative to the chip's specification.
    """
    if use_fspl:
        pl = fspl_db(distance_m, frequency_mhz)
    else:
        pl = log_distance_path_loss_db(
            distance_m, frequency_mhz, path_loss_exponent
        )

    p_rx = (
        tx_power_dbm
        + tx_antenna_gain_dbi
        + rx_antenna_gain_dbi
        - pl
        - cable_loss_db
    )

    sensitivity = SX1278_SENSITIVITY_BW125[sf]
    snr_threshold = SX1278_SNR_THRESHOLD[sf]

    link_margin = p_rx - sensitivity
    snr_estimate = p_rx - noise_floor_dbm
    snr_margin = snr_estimate - snr_threshold

    decodable = (link_margin > 0) and (snr_margin > 0)
    reliable = link_margin > 10.0  # 10 dB margin is the standard fade reserve

    return LinkBudgetResult(
        distance_m=distance_m,
        path_loss_db=round(pl, 2),
        received_power_dbm=round(p_rx, 2),
        sensitivity_dbm=sensitivity,
        link_margin_db=round(link_margin, 2),
        snr_estimate_db=round(snr_estimate, 2),
        snr_threshold_db=snr_threshold,
        snr_margin_db=round(snr_margin, 2),
        decodable=decodable,
        reliable=reliable,
    )


# =============================================================================
# UTILITY: maximum range solver
# =============================================================================
def max_range_m(
    sf: int = 9,
    target_margin_db: float = 10.0,
    path_loss_exponent: float = 2.7,
    **link_budget_kwargs,
) -> float:
    """
    Solve for the maximum distance at which the link maintains target_margin_db.

    Uses binary search rather than algebra so it works for any path loss
    model we add later (terrain, Fresnel, etc.).

    target_margin_db = 0  → absolute maximum theoretical range
    target_margin_db = 10 → reliable range with fade reserve (recommended)
    """
    low, high = 1.0, 100_000.0  # 1 m to 100 km search range

    for _ in range(50):  # binary search converges in ~17 iterations for our precision
        mid = (low + high) / 2.0
        result = link_budget(
            mid, sf=sf, path_loss_exponent=path_loss_exponent, **link_budget_kwargs
        )
        if result.link_margin_db > target_margin_db:
            low = mid
        else:
            high = mid

    return round(low, 1)


# =============================================================================
# DEMO: when run directly, print the key numbers for your design
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("LoRa Mesh Sandbox — Physics Layer Self-Test")
    print("Project settings: SF9, BW 125 kHz, CR 4/5, 433 MHz, +14 dBm")
    print("=" * 70)

    # ---- Time on Air for typical packet sizes ----
    print("\n[1] TIME-ON-AIR for various payload sizes (SF9, BW125, CR4/5):")
    print(f"    {'Payload':>10} {'ToA (ms)':>12} {'Symbols':>10}")
    for pl_size in [10, 20, 50, 100, 200]:
        toa = time_on_air_ms(payload_bytes=pl_size, sf=9)
        print(f"    {pl_size:>7} B   {toa['t_total_ms']:>10.2f}   "
              f"{toa['n_payload_symbols']:>8}")

    # ---- ToA across SFs for fixed 20-byte heartbeat ----
    print("\n[2] TIME-ON-AIR across spreading factors (20 B heartbeat):")
    print(f"    {'SF':>4} {'ToA (ms)':>12} {'Bitrate (bps)':>15}")
    for sf in range(7, 13):
        toa = time_on_air_ms(payload_bytes=20, sf=sf)
        # Effective bitrate ≈ payload bits / total airtime
        bitrate = (20 * 8) / (toa['t_total_ms'] / 1000.0)
        print(f"    SF{sf:<2} {toa['t_total_ms']:>10.2f}   {bitrate:>10.0f}")

    # ---- Path loss reference table ----
    print("\n[3] PATH LOSS at 433 MHz:")
    print(f"    {'Distance':>10} {'FSPL':>10} {'Log-Dist (n=2.7)':>20}")
    for d in [100, 500, 1000, 2500, 5000, 10000]:
        fspl = fspl_db(d)
        ld = log_distance_path_loss_db(d, path_loss_exponent=2.7)
        print(f"    {d:>7} m   {fspl:>7.2f} dB   {ld:>16.2f} dB")

    # ---- Link budget at typical distances for your design ----
    print("\n[4] LINK BUDGET at SF9 (your design choice), n=2.7:")
    print(f"    {'Distance':>10} {'PL':>9} {'P_rx':>9} {'Margin':>9} {'Decode?':>9}")
    for d in [500, 1000, 2000, 5000, 10000]:
        lb = link_budget(d, sf=9, path_loss_exponent=2.7)
        decode_str = "YES" if lb.decodable else "NO"
        reliable_str = "+reliable" if lb.reliable else ""
        print(f"    {d:>7} m   {lb.path_loss_db:>6.1f}   "
              f"{lb.received_power_dbm:>6.1f}   "
              f"{lb.link_margin_db:>+6.1f}   {decode_str:>6} {reliable_str}")

    # ---- Maximum range across SFs ----
    print("\n[5] MAXIMUM RANGE for reliable link (10 dB margin, n=2.7):")
    print(f"    {'SF':>4} {'Max range (km)':>16}")
    for sf in range(7, 13):
        r = max_range_m(sf=sf, target_margin_db=10.0, path_loss_exponent=2.7)
        print(f"    SF{sf:<2} {r/1000:>12.2f}")

    # ---- Sanity check: duty cycle at 5 second heartbeat ----
    print("\n[6] DUTY CYCLE CHECK (your 4-6 sec heartbeat target):")
    toa_20b = time_on_air_ms(payload_bytes=20, sf=9)['t_total_ms']
    for interval in [4.0, 5.0, 6.0]:
        dc = (toa_20b / 1000.0) / interval * 100
        verdict = "OK" if dc < 10.0 else "EXCEEDS 10% LIMIT"
        print(f"    Heartbeat {interval}s → duty cycle {dc:.2f}%  ({verdict})")

    print("\n" + "=" * 70)
    print("Physics module ready. Import and use these functions, or run")
    print("`python physics.py` to regenerate this report.")
    print("=" * 70)
