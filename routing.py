"""
LoRa Mesh Sandbox — Module 4: Routing Table
============================================
Simplified RPL OF0-style parent selection.
Each non-CONTROL node maintains one entry per heard neighbour.
Parent is selected as the neighbour with the strongest recent RSSI.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

# Entry expires if no beacon heard within this window
ROUTE_EXPIRY_S = 30.0
# Module 5 imports this name; same constant, kept as an alias for back-compat
ROUTE_STALE_TIMEOUT_S = ROUTE_EXPIRY_S


@dataclass
class RouteEntry:
    """One neighbour heard via beacon."""
    node_id:    int
    rssi_dbm:   float
    hop_count:  int       # hop_count advertised in the beacon
    last_seen:  float     # simulation time of last beacon from this neighbour


class RoutingTable:
    """
    Maintains a table of heard neighbours and selects the best parent.

    Parent selection rule (OF0-simplified):
        Choose the neighbour with the lowest advertised hop_count.
        Break ties by strongest RSSI.

    This mirrors what a real RPL implementation does: minimise path length
    first, then signal quality. For our four-node topology the result is
    always unambiguous.
    """

    def __init__(self, owner_id: int, control_id: int):
        """
        Args:
            owner_id:   node ID of the node that owns this table
            control_id: root of the tree — never selected as a forwarding
                        parent by CONTROL itself (it IS the root)
        """
        self.owner_id   = owner_id
        self.control_id = control_id
        self._entries: Dict[int, RouteEntry] = {}

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------

    def update(
        self,
        neighbour_id: int,
        rssi_dbm:     float,
        hop_count:    int,
        now:          float,
    ) -> None:
        """
        Record or refresh a neighbour entry from a received beacon.

        Called by nodes.py whenever a beacon is successfully decoded.
        """
        self._entries[neighbour_id] = RouteEntry(
            node_id   = neighbour_id,
            rssi_dbm  = rssi_dbm,
            hop_count = hop_count,
            last_seen = now,
        )

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def _fresh_entries(self, now: float) -> list:
        """Return entries that have not expired."""
        return [
            e for e in self._entries.values()
            if (now - e.last_seen) <= ROUTE_EXPIRY_S
        ]

    def has_route(self, now: float) -> bool:
        """True if at least one fresh neighbour entry exists."""
        return len(self._fresh_entries(now)) > 0

    def best_parent(self, now: float) -> Optional[int]:
        """
        Return the node ID of the best parent, or None if no route exists.

        Selection: lowest hop_count first, then strongest RSSI to break ties.
        """
        entries = self._fresh_entries(now)
        if not entries:
            return None
        best = min(entries, key=lambda e: (e.hop_count, -e.rssi_dbm))
        return best.node_id

    def my_hop_count(self, now: float) -> Optional[int]:
        """
        Return this node's own hop count = best parent's hop_count + 1.
        Returns None if no route exists.
        """
        entries = self._fresh_entries(now)
        if not entries:
            return None
        best = min(entries, key=lambda e: (e.hop_count, -e.rssi_dbm))
        return best.hop_count + 1

    def __repr__(self) -> str:
        return (
            f"RoutingTable(owner={self.owner_id}, "
            f"entries={list(self._entries.keys())})"
        )