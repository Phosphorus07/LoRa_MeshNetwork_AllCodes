"""
LoRa Mesh Sandbox — Module 3b: Extended Radio Channel with Collision Detection
===============================================================================
Author: Simon Henson (with Claude)
Project: LoRa Mesh Network Water Pump Monitoring & Control System

This module extends channel.py (Module 2b) with two new capabilities:

    1. in_flight tracking  — records every transmission currently on air
    2. Collision detection — when two packets overlap on the same SF/freq,
       both are lost (worst-case model, no capture effect)

Why worst-case / no capture effect?
    The SX1278 can sometimes decode the stronger of two overlapping packets
    if the power difference exceeds ~6 dB (the "capture effect"). However:
    - The capture ratio varies with SF, BW, and timing alignment
    - It is not guaranteed by Semtech's spec
    - Modelling it requires knowing both RSSI values simultaneously,
      which complicates the DES flow
    - For a viva/report it is more defensible to say "we assume worst-case;
      the real system will only be better" than to make optimistic assumptions

Collision model:
    Any time-window overlap between two transmissions on the same SF and
    frequency = both packets lost, regardless of relative power.
    The overlap check uses half-open intervals: [tx_start, tx_end).

All original channel.py logic is preserved unchanged.  run_module2.py
continues to import from channel.py; run_module3.py imports from here.

References:
    - Semtech AN1200.13: "LoRa Modem Designer's Guide"
    - Rappaport, T.S., "Wireless Communications: Principles & Practice"
    - Bor, M. et al., "Do LoRa Low-Power Wide-Area Networks Scale?",
      MSWiM 2016 — source of the worst-case collision model used here
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from physics import (
    log_distance_path_loss_db,
    SX1278_SENSITIVITY_BW125,
    SX1278_SNR_THRESHOLD,
)


# =============================================================================
# RECEPTION RESULT  (identical to channel.py — copied verbatim)
# =============================================================================
@dataclass
class Reception:
    """
    What the receiver "sees" when a packet arrives.

    decoded:        True if the chip successfully demodulated the packet
    rssi_dbm:       Received Signal Strength Indicator (dBm)
    snr_db:         Signal-to-Noise Ratio (dB)
    margin_db:      How far above sensitivity (negative = below threshold)
    failure_reason: Why decoded=False:
                        'below_sensitivity' — signal too weak
                        'snr_too_low'       — SNR below SF9 demod floor
                        'random_per'        — near-threshold probabilistic loss
                        'collision'         — overlapping transmission on same SF/freq
    """
    decoded: bool
    rssi_dbm: float
    snr_db: float
    margin_db: float
    failure_reason: Optional[str] = None


# =============================================================================
# RADIO CHANNEL V3
# =============================================================================
class RadioChannelV3:
    """
    Simulated LoRa link with collision detection.

    Extends RadioChannel (channel.py) with:
        in_flight   — list of dicts tracking active transmissions
        register_tx — call at TX_START to log a transmission entering the air
        resolve_rx  — call at TX_END to decide packet fate

    All original physics (path loss, shadow fading, SNR check, sigmoid PER)
    are preserved exactly.  The RNG is a private random.Random instance seeded
    at construction time — identical to channel.py's behaviour.

    Collision detection strategy:
        Eager marking in register_tx:
            When a new TX is registered, we immediately check whether it
            overlaps any existing in-flight TX on the same SF/freq.  If so,
            BOTH are marked 'collided=True'.  This eager approach handles the
            simultaneous-start case correctly: by the time resolve_rx is called
            for the first packet, the second packet's entry already exists and
            both have been flagged.

        Belt-and-suspenders check in resolve_rx:
            resolve_rx also scans in_flight for overlapping entries (spec
            requirement).  This catches staggered-start collisions where TX_B
            starts after TX_A but before TX_A ends — register_tx for TX_A
            can't know about TX_B yet, so the check must also run at resolve
            time.

        Together these two checks cover all collision scenarios.
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
        # ---- Link-budget parameters (same as channel.py) ----
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
        # Private RNG — seeded identically to channel.py so Monte Carlo
        # sequences are bit-for-bit reproducible across both modules.
        self.rng = random.Random(seed)

        # ---- Collision tracking (new in V3) ----
        # Each entry is a dict:
        #   tx_start   : float  — simulated time TX started (s)
        #   tx_end     : float  — simulated time TX ends (s)
        #   node_id    : int    — which node is transmitting
        #   freq_mhz   : float  — carrier frequency
        #   sf         : int    — spreading factor
        #   distance_m : float  — distance from TX to RX (for physics calc)
        #   collided   : bool   — True if overlap with another TX detected
        #   resolved   : bool   — True if resolve_rx has already been called
        self.in_flight: List[Dict] = []

    # -------------------------------------------------------------------------
    # CORE PHYSICS  (verbatim from channel.py — do not alter)
    # -------------------------------------------------------------------------
    def transmit(self, distance_m: float) -> Reception:
        """
        Simulate a packet transmission across the channel.

        Called internally by resolve_rx for non-collision packets.
        Can also be called directly (as in Module 2) for the simple case
        where collision detection is not needed.

        The decision logic mirrors how a real SX1278 chip works:
            1. Draw a random shadow fade (Gaussian, std = sigma dB)
            2. Compute path loss including the fade
            3. Compute received power (RSSI) and SNR
            4. Hard-fail if RSSI < sensitivity or SNR < threshold
            5. Near-threshold: apply sigmoid Packet Error Rate curve
               PER ≈ 0.5 at snr_margin = 0, ~0.01 at snr_margin = 4 dB
        """
        # Step 1: random shadow fading — Gaussian with given std dev
        shadow = self.rng.gauss(0.0, self.sigma)
        path_loss = log_distance_path_loss_db(
            distance_m, path_loss_exponent=self.n, shadow_fading_db=shadow
        )

        # Step 2: master link budget equation
        # P_rx = P_tx + G_tx + G_rx - PL - L_cable
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

        # Step 3: hard floor — signal must reach sensitivity
        if margin <= 0:
            return Reception(
                decoded=False, rssi_dbm=round(rssi, 1), snr_db=round(snr, 1),
                margin_db=round(margin, 1), failure_reason="below_sensitivity",
            )

        # Step 4: SNR floor — chip can't demodulate if SNR < threshold
        if snr_margin <= 0:
            return Reception(
                decoded=False, rssi_dbm=round(rssi, 1), snr_db=round(snr, 1),
                margin_db=round(margin, 1), failure_reason="snr_too_low",
            )

        # Step 5: near-threshold probabilistic loss (sigmoid PER curve)
        # This models the gradual degradation in the link, not a hard cliff.
        # PER ≈ 0.5 at snr_margin = 0 dB, drops to ~0.01 by snr_margin = 4 dB
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

    # -------------------------------------------------------------------------
    # COLLISION TRACKING — NEW IN V3
    # -------------------------------------------------------------------------
    def register_tx(
        self,
        tx_start: float,
        tx_end: float,
        node_id: int,
        freq_mhz: float,
        sf: int,
        distance_m: float,
    ) -> None:
        """
        Register a new transmission entering the air.

        Call this at TX_START time (when the first symbol leaves the antenna).

        Args:
            tx_start   — simulation time when TX begins (s)
            tx_end     — simulation time when TX ends (s = tx_start + ToA)
            node_id    — ID of the transmitting node
            freq_mhz   — carrier frequency (MHz)
            sf         — spreading factor (must match receiver's SF to collide)
            distance_m — TX-to-RX distance (stored for physics calc at TX_END)

        Sanity checks:
            - tx_end must be after tx_start (otherwise ToA calculation is wrong)
            - node_id must not already have an unresolved TX in flight
        """
        if tx_end <= tx_start:
            raise ValueError(
                f"register_tx: tx_end ({tx_end:.6f}s) must be > "
                f"tx_start ({tx_start:.6f}s). "
                "Check that ToA calculation returned a positive value."
            )

        # Warn loudly if a node tries to transmit twice simultaneously
        for existing in self.in_flight:
            if existing["node_id"] == node_id and not existing["resolved"]:
                raise RuntimeError(
                    f"register_tx: node {node_id} already has an unresolved "
                    "TX in flight. A real radio can only transmit one packet "
                    "at a time."
                )

        # Build the new entry
        entry: Dict = {
            "tx_start":   tx_start,
            "tx_end":     tx_end,
            "node_id":    node_id,
            "freq_mhz":   freq_mhz,
            "sf":         sf,
            "distance_m": distance_m,
            "collided":   False,   # will be set True if overlap found
            "resolved":   False,   # will be set True after resolve_rx
        }

        # Eager collision scan: check ALL existing (unresolved) in-flight TXs.
        # If any overlap this new TX on the same SF and freq → mark both.
        # This handles the simultaneous-start case: both packets arrive in
        # register_tx before either is resolved, so both get flagged here.
        for existing in self.in_flight:
            if existing["resolved"]:
                continue  # already processed — can't affect new TX
            if existing["node_id"] == node_id:
                continue  # same transmitter — already caught above
            if existing["freq_mhz"] != freq_mhz or existing["sf"] != sf:
                continue  # different channel — no interaction

            # Time-window overlap check: [A_start, A_end) overlaps [B_start, B_end)
            # iff A_start < B_end AND A_end > B_start
            if existing["tx_start"] < tx_end and existing["tx_end"] > tx_start:
                existing["collided"] = True
                entry["collided"] = True

        self.in_flight.append(entry)

    def resolve_rx(
        self,
        tx_end_time: float,
        node_id: int,
        freq_mhz: float,
        sf: int,
    ) -> Reception:
        """
        Resolve what happened to a packet at the end of its transmission.

        Call this at TX_END time (after the last symbol has been sent).

        Steps:
            1. Find this node's in-flight entry (by node_id + tx_end time).
            2. Check in_flight for any OTHER overlapping TX on same SF/freq.
               (Belt-and-suspenders: catches staggered-start collisions that
               register_tx couldn't detect eagerly.)
            3. If collision detected (either from step 2 or eager marking):
               return Reception(decoded=False, failure_reason='collision')
               and do NOT draw a shadow-fading sample.  This preserves the
               RNG sequence for non-collision packets so the equivalence
               check against Module 2 holds.
            4. If no collision: call transmit(distance_m) which draws a shadow
               fade and runs the normal PER logic.

        Args:
            tx_end_time — simulation time when TX ends (must match register_tx)
            node_id     — transmitting node ID
            freq_mhz    — carrier frequency
            sf          — spreading factor

        Returns:
            Reception with failure_reason='collision', 'below_sensitivity',
            'snr_too_low', 'random_per', or decoded=True.

        Raises:
            RuntimeError if no matching unresolved entry is found.
        """
        # Step 1: find this tx's entry in in_flight
        this_tx: Optional[Dict] = None
        for tx in self.in_flight:
            if (
                tx["node_id"] == node_id
                and abs(tx["tx_end"] - tx_end_time) < 1e-9   # float-safe compare
                and not tx["resolved"]
            ):
                this_tx = tx
                break

        if this_tx is None:
            raise RuntimeError(
                f"resolve_rx: no unresolved in-flight TX found for "
                f"node_id={node_id}, tx_end≈{tx_end_time:.6f}s. "
                "Did you call register_tx before resolve_rx?"
            )

        tx_start = this_tx["tx_start"]
        distance_m = this_tx["distance_m"]

        # Step 2: belt-and-suspenders collision check in resolve_rx.
        # We check ALL entries (resolved=True or False) because a resolved
        # entry was still physically in the air during its window — it counts
        # as a collider regardless of DES processing order.
        collision_found = this_tx["collided"]  # may already be True from register_tx

        if not collision_found:
            for tx in self.in_flight:
                if tx is this_tx:
                    continue
                if tx["node_id"] == node_id:
                    continue  # same transmitter — not a collision
                if tx["freq_mhz"] != freq_mhz or tx["sf"] != sf:
                    continue  # different channel
                # Check overlap against THIS tx's window
                if tx["tx_start"] < tx_end_time and tx["tx_end"] > tx_start:
                    collision_found = True
                    break

        # Step 3: mark resolved and prune old entries.
        # We mark BEFORE pruning so that a concurrent same-time collider
        # (already marked collided=True by register_tx) still sees this entry
        # in the list if it calls resolve_rx in the same DES tick.
        this_tx["resolved"] = True

        # Prune all fully-resolved entries whose tx_end has passed.
        # Keeping the list clean avoids false positives in future collision checks.
        self.in_flight = [tx for tx in self.in_flight if not tx["resolved"]]

        # Step 4: return result
        if collision_found:
            # Collision — return a Reception indicating loss without touching the RNG.
            # Not touching the RNG is critical: it keeps the shadow-fading sequence
            # for non-collision packets identical to Module 2's channel.py sequence.
            return Reception(
                decoded=False,
                rssi_dbm=0.0,
                snr_db=0.0,
                margin_db=0.0,
                failure_reason="collision",
            )

        # No collision — run the normal link-budget + PER calculation.
        # transmit() draws one gauss sample + one rng.random() from self.rng,
        # exactly as channel.py does, preserving the Module 2 equivalence.
        return self.transmit(distance_m)

    # -------------------------------------------------------------------------
    # pop_collision_flag — Module 4 helper for BROADCAST beacons
    # -------------------------------------------------------------------------
    def pop_collision_flag(
        self,
        tx_end_time: float,
        node_id: int,
        freq_mhz: float,
        sf: int,
    ) -> bool:
        """
        Same collision logic as resolve_rx Steps 1–3, but without the
        path-loss draw and without returning a Reception.

        Why we need it
        ---------------
        A broadcast beacon is one TX heard by N receivers, each at a
        different distance — so each receiver must run its OWN
        shadow-fading / link-budget draw via channel.transmit(distance_m).
        If we called resolve_rx() N times we would consume N RNG samples
        for what is physically a single transmission, drifting the seed
        sequence away from the Modules-2/3 reference.

        Module 4's beacon flow becomes:
            collided = channel.pop_collision_flag(...)   # one check, no RNG
            if not collided:
                for rx in receivers:
                    rx_result = channel.transmit(dist_m) # one RNG draw per rx

        Steps:
            1. Find this tx's in_flight entry.
            2. OR the eager collision flag with a fresh overlap scan.
            3. Mark the entry resolved and prune all resolved entries.
            4. Return True if the beacon collided, False otherwise.

        Critical: this method MUST NOT touch self.rng.  Doing so would
        break the seed-equivalence guarantee that the project relies on
        for reproducible Module 2/3/4 PDR figures.
        """
        # Step 1 — find this tx's entry (identical to resolve_rx)
        this_tx: Optional[Dict] = None
        for tx in self.in_flight:
            if (
                tx["node_id"] == node_id
                and abs(tx["tx_end"] - tx_end_time) < 1e-9
                and not tx["resolved"]
            ):
                this_tx = tx
                break

        if this_tx is None:
            raise RuntimeError(
                f"pop_collision_flag: no unresolved in-flight TX for "
                f"node_id={node_id}, tx_end≈{tx_end_time:.6f}s. "
                "Did you call register_tx before pop_collision_flag?"
            )

        tx_start = this_tx["tx_start"]

        # Step 2 — overlap scan, identical to resolve_rx
        collision_found = this_tx["collided"]
        if not collision_found:
            for tx in self.in_flight:
                if tx is this_tx:
                    continue
                if tx["node_id"] == node_id:
                    continue
                if tx["freq_mhz"] != freq_mhz or tx["sf"] != sf:
                    continue
                if tx["tx_start"] < tx_end_time and tx["tx_end"] > tx_start:
                    collision_found = True
                    break

        # Step 3 — mark resolved + prune (same accounting as resolve_rx so the
        # in_flight list stays clean and future collisions don't see a stale
        # entry as a phantom collider)
        this_tx["resolved"] = True
        self.in_flight = [tx for tx in self.in_flight if not tx["resolved"]]

        return collision_found


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("channel_v3 self-test")
    print("=" * 70)

    ch = RadioChannelV3(seed=42)

    # ---- 1. Basic single-packet reception (no collision) ----
    print("\n[1] Single packet at 2 km (no collision expected):")
    ch.register_tx(0.0, 0.165, node_id=2, freq_mhz=433.0, sf=9, distance_m=2000.0)
    rx = ch.resolve_rx(0.165, node_id=2, freq_mhz=433.0, sf=9)
    print(f"    decoded={rx.decoded}  RSSI={rx.rssi_dbm} dBm  "
          f"reason={rx.failure_reason}")
    assert rx.decoded, "Single clean packet at 2 km should decode"
    assert rx.failure_reason != "collision"

    # ---- 2. Simultaneous collision ----
    print("\n[2] Simultaneous collision (two nodes, same time, same SF/freq):")
    ch2 = RadioChannelV3(seed=42)
    ch2.register_tx(1.0, 1.165, node_id=2, freq_mhz=433.0, sf=9, distance_m=2000.0)
    ch2.register_tx(1.0, 1.165, node_id=3, freq_mhz=433.0, sf=9, distance_m=2000.0)
    rx_a = ch2.resolve_rx(1.165, node_id=2, freq_mhz=433.0, sf=9)
    rx_b = ch2.resolve_rx(1.165, node_id=3, freq_mhz=433.0, sf=9)
    print(f"    Node 2: decoded={rx_a.decoded}  reason={rx_a.failure_reason}")
    print(f"    Node 3: decoded={rx_b.decoded}  reason={rx_b.failure_reason}")
    assert rx_a.failure_reason == "collision", "Node 2 should be collision"
    assert rx_b.failure_reason == "collision", "Node 3 should also be collision"

    # ---- 3. Staggered collision (B starts during A) ----
    print("\n[3] Staggered collision (B starts 50 ms into A's transmission):")
    ch3 = RadioChannelV3(seed=42)
    ch3.register_tx(1.0, 1.165, node_id=2, freq_mhz=433.0, sf=9, distance_m=2000.0)
    ch3.register_tx(1.05, 1.215, node_id=3, freq_mhz=433.0, sf=9, distance_m=2000.0)
    rx_a = ch3.resolve_rx(1.165, node_id=2, freq_mhz=433.0, sf=9)
    rx_b = ch3.resolve_rx(1.215, node_id=3, freq_mhz=433.0, sf=9)
    print(f"    Node 2: decoded={rx_a.decoded}  reason={rx_a.failure_reason}")
    print(f"    Node 3: decoded={rx_b.decoded}  reason={rx_b.failure_reason}")
    assert rx_a.failure_reason == "collision"
    assert rx_b.failure_reason == "collision"

    # ---- 4. Different SF — no collision ----
    print("\n[4] Same time but different SF — should NOT collide:")
    ch4 = RadioChannelV3(seed=42)
    ch4.register_tx(1.0, 1.165, node_id=2, freq_mhz=433.0, sf=9,  distance_m=2000.0)
    ch4.register_tx(1.0, 1.165, node_id=3, freq_mhz=433.0, sf=10, distance_m=2000.0)
    # resolve in sf=9 context
    rx_a = ch4.resolve_rx(1.165, node_id=2, freq_mhz=433.0, sf=9)
    rx_b = ch4.resolve_rx(1.165, node_id=3, freq_mhz=433.0, sf=10)
    print(f"    Node 2 (SF9):  decoded={rx_a.decoded}  reason={rx_a.failure_reason}")
    print(f"    Node 3 (SF10): decoded={rx_b.decoded}  reason={rx_b.failure_reason}")
    assert rx_a.failure_reason != "collision", "Different SF should not collide"

    print("\n  All assertions passed.")
    print("  channel_v3 module ready.")
    print("=" * 70)
