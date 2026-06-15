"""
LoRa Mesh Sandbox — Module 2b: Simulated Radio Channel
========================================================
The bridge between Module 1 (physics) and Module 2 (nodes).

This is where a packet "travels" from one node to another. The channel:
    1. Asks Module 1 for the link budget at the given distance
    2. Adds random shadow fading (X_sigma in the log-distance model)
    3. Decides whether the packet decodes (link margin + SNR threshold)
    4. Reports the realistic RSSI/SNR the receiver would have measured

This is the layer where "real-world messiness" enters the simulation.
Without the random fading, every packet would either always succeed or
always fail — that's not how RF works.
"""

import random
from dataclasses import dataclass
from typing import Optional

from physics import (
    log_distance_path_loss_db,
    SX1278_SENSITIVITY_BW125,
    SX1278_SNR_THRESHOLD,
)


# =============================================================================
# RECEPTION RESULT
# =============================================================================
@dataclass
class Reception:
    """
    What the receiver "sees" when a packet arrives.

    decoded:        True if the chip successfully demodulated
    rssi_dbm:       Received Signal Strength Indicator
    snr_db:         Signal-to-Noise Ratio
    margin_db:      How far above sensitivity (negative = below)
    failure_reason: If decoded=False, why ('below_sensitivity', 'snr_too_low', 'random_per')
    """
    decoded: bool
    rssi_dbm: float
    snr_db: float
    margin_db: float
    failure_reason: Optional[str] = None


# =============================================================================
# CHANNEL
# =============================================================================
class RadioChannel:
    """
    Simulated LoRa link between two points.

    Configurable parameters:
        path_loss_exponent  — environment factor (2.7 = rural Karoo)
        shadow_fading_std   — std dev of random fading in dB (typical 4-8 dB)
        seed                — RNG seed for reproducible runs

    Constants assumed (matching your design):
        SF=9, BW=125 kHz, CR=4/5, freq=433 MHz
        Tx power: 14 dBm, antenna gain: 2 dBi each end, cable loss: 1 dB
    """

    def __init__(
        self,
        path_loss_exponent: float = 2.7,
        shadow_fading_std_db: float = 6.0,
        sf: int = 9,
        tx_power_dbm: float = 14.0,
        tx_gain_dbi: float = 2.0,
        rx_gain_dbi: float = 2.0,
        cable_loss_db: float = 1.0,
        noise_floor_dbm: float = -124.0,
        seed: Optional[int] = None,
    ):
        self.n = path_loss_exponent
        self.sigma = shadow_fading_std_db
        self.sf = sf
        self.tx_power = tx_power_dbm
        self.tx_gain = tx_gain_dbi
        self.rx_gain = rx_gain_dbi
        self.cable_loss = cable_loss_db
        self.noise_floor = noise_floor_dbm
        self.sensitivity = SX1278_SENSITIVITY_BW125[sf]
        self.snr_threshold = SX1278_SNR_THRESHOLD[sf]
        self.rng = random.Random(seed)

    def transmit(self, distance_m: float) -> Reception:
        """
        Simulate a packet transmission across the channel.

        Returns a Reception object describing what the receiver gets.

        The decision logic mirrors how a real SX1278 chip works:
            1. Compute path loss with random shadow fading
            2. Compute received power
            3. Compute SNR estimate
            4. Packet decodes if BOTH:
                 - received power > sensitivity (otherwise nothing to demod)
                 - SNR > threshold for this SF
            5. Even if both conditions met, near-threshold packets have
               probabilistic loss — we model this with a sigmoid curve.
        """
        # Random shadow fading — Gaussian with given std dev
        shadow = self.rng.gauss(0.0, self.sigma)
        path_loss = log_distance_path_loss_db(
            distance_m, path_loss_exponent=self.n, shadow_fading_db=shadow
        )

        # Master link budget equation
        rssi = (
            self.tx_power
            + self.tx_gain
            + self.rx_gain
            - path_loss
            - self.cable_loss
        )
        margin = rssi - self.sensitivity
        snr = rssi - self.noise_floor
        snr_margin = snr - self.snr_threshold

        # Hard failures
        if margin <= 0:
            return Reception(
                decoded=False, rssi_dbm=round(rssi, 1), snr_db=round(snr, 1),
                margin_db=round(margin, 1), failure_reason="below_sensitivity",
            )
        if snr_margin <= 0:
            return Reception(
                decoded=False, rssi_dbm=round(rssi, 1), snr_db=round(snr, 1),
                margin_db=round(margin, 1), failure_reason="snr_too_low",
            )

        # Probabilistic zone near threshold (sigmoid)
        # PER ≈ 0.5 at snr_margin = 0, drops to ~0.01 by snr_margin = 4 dB
        per = 1.0 / (1.0 + 10 ** (snr_margin / 1.5))
        if self.rng.random() < per:
            return Reception(
                decoded=False, rssi_dbm=round(rssi, 1), snr_db=round(snr, 1),
                margin_db=round(margin, 1), failure_reason="random_per",
            )

        return Reception(
            decoded=True, rssi_dbm=round(rssi, 1), snr_db=round(snr, 1),
            margin_db=round(margin, 1),
        )


# =============================================================================
# SELF-TEST — confirm the channel behaves sensibly across distances
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Channel module — self-test")
    print("Running 1000 transmissions at each distance to check PDR")
    print("=" * 70)

    ch = RadioChannel(path_loss_exponent=2.7, shadow_fading_std_db=6.0, seed=42)

    print(f"\n{'Distance':>10} {'Mean RSSI':>11} {'Mean Margin':>13} {'PDR':>8}")
    print("-" * 50)

    for distance in [500, 1000, 2000, 5000, 8000, 12000, 15000]:
        results = [ch.transmit(distance) for _ in range(1000)]
        decoded_count = sum(1 for r in results if r.decoded)
        mean_rssi = sum(r.rssi_dbm for r in results) / len(results)
        mean_margin = sum(r.margin_db for r in results) / len(results)
        pdr = decoded_count / len(results) * 100
        print(f"{distance:>7} m  {mean_rssi:>8.1f}    {mean_margin:>+8.1f}    "
              f"{pdr:>5.1f}%")

    print("\n" + "=" * 70)
    print("Channel module ready.")
    print("=" * 70)
