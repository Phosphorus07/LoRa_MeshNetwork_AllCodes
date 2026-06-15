"""
LoRa Mesh Sandbox — Module 2a: Packet Structure
================================================
Defines the byte-level packet format used over the LoRa link.

The format is intentionally simple but production-defensible:
    [marker][version][type][src][dst][seq_hi][seq_lo][payload_len][payload...][crc_hi][crc_lo]

Total minimum packet size: 10 bytes (8 header + 2 CRC, no payload)
Maximum payload size: 247 bytes (255 byte LoRa max - 8 header - 2 CRC = 245
but we round to 247 to keep one digit of safety. SX1278 supports up to 255 B.)

Why these design choices? See the journal document — every byte is justified.
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# =============================================================================
# CONSTANTS
# =============================================================================
PACKET_MARKER = 0xAA          # "is this our protocol?"
PROTOCOL_VERSION = 0x01       # bump if format changes
HEADER_SIZE = 8               # bytes before payload
CRC_SIZE = 2                  # CRC-16 trailer
MIN_PACKET_SIZE = HEADER_SIZE + CRC_SIZE
MAX_PAYLOAD_SIZE = 247
BROADCAST_ID = 0xFF


class PacketType(IntEnum):
    """Packet types our protocol understands.

    Defined as an IntEnum so we can compare with integers and also
    print human-readable names when logging.
    """
    BEACON   = 0x01  # heartbeat / periodic announcement
    DATA     = 0x02  # sensor data going up to control
    ACK      = 0x03  # hop-by-hop acknowledgement (Module 4)
    CMD      = 0x04  # control command going down to subscriber (later)
    JOIN_REQ = 0x05  # node wants to join (later)
    JOIN_ACK = 0x06  # control accepts join (later)
    ACK_E2E  = 0x07  # end-to-end ACK from CONTROL back to originator (Module 5)


class SensorType(IntEnum):
    """Sensor types we can encode in a DATA payload."""
    TEMPERATURE = 0x01
    WATER_LEVEL = 0x02     # for later
    FLOW_RATE = 0x03       # for later
    PRESSURE = 0x04        # for later
    PUMP_STATUS = 0x05     # for later
    VOLTAGE_CURRENT = 0x06 # for later


# =============================================================================
# CRC-16 (CCITT polynomial 0x1021)
# =============================================================================
def crc16(data: bytes) -> int:
    """
    Compute CRC-16-CCITT (polynomial 0x1021, initial value 0xFFFF).

    This is a software implementation. On the real Heltec firmware we'd
    likely use the STM32's hardware CRC peripheral, but the result is
    bit-identical so the sandbox is faithful to what the hardware will do.

    Why CRC-CCITT? It's the standard for embedded comms (used by XMODEM,
    Bluetooth, X.25). 65,535 distinct values means a random corruption has
    only 1-in-65536 chance of slipping through.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# =============================================================================
# PACKET CLASS
# =============================================================================
@dataclass
class Packet:
    """
    Application-layer LoRa packet.

    Build a packet with the constructor, serialize with .to_bytes(),
    deserialize with Packet.from_bytes(). The class handles all the
    byte-twiddling so node code stays clean.
    """
    packet_type: PacketType
    src_id: int
    dst_id: int
    seq_num: int
    payload: bytes = b""

    def __post_init__(self):
        # Sanity checks — fail loud in simulation, never silently corrupt
        if not (0 <= self.src_id <= 255):
            raise ValueError(f"src_id out of range: {self.src_id}")
        if not (0 <= self.dst_id <= 255):
            raise ValueError(f"dst_id out of range: {self.dst_id}")
        if not (0 <= self.seq_num <= 0xFFFF):
            raise ValueError(f"seq_num out of range: {self.seq_num}")
        if len(self.payload) > MAX_PAYLOAD_SIZE:
            raise ValueError(
                f"payload too large: {len(self.payload)} > {MAX_PAYLOAD_SIZE}"
            )

    def to_bytes(self) -> bytes:
        """Serialize the packet to its on-the-wire byte representation."""
        # struct format: '>BBBBBHB' = 8 bytes, big-endian
        #   B = 1-byte unsigned int (marker, version, type, src, dst, payload_len)
        #   H = 2-byte unsigned int (seq_num)
        # We use big-endian (network byte order) by convention for protocols.
        header = struct.pack(
            ">BBBBBHB",
            PACKET_MARKER,
            PROTOCOL_VERSION,
            int(self.packet_type),
            self.src_id,
            self.dst_id,
            self.seq_num,
            len(self.payload),
        )
        body = header + self.payload
        crc = crc16(body)
        return body + struct.pack(">H", crc)

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["Packet"]:
        """
        Parse bytes back into a Packet, or return None if invalid.

        Returns None instead of raising — receivers should silently drop
        bad packets, not crash. We log the reason for analysis.
        """
        if len(data) < MIN_PACKET_SIZE:
            return None

        # Verify CRC first — cheapest way to reject garbage
        body, crc_bytes = data[:-CRC_SIZE], data[-CRC_SIZE:]
        expected_crc = crc16(body)
        actual_crc = struct.unpack(">H", crc_bytes)[0]
        if expected_crc != actual_crc:
            return None

        # Parse header
        try:
            marker, version, ptype, src, dst, seq, plen = struct.unpack(
                ">BBBBBHB", body[:HEADER_SIZE]
            )
        except struct.error:
            return None

        if marker != PACKET_MARKER:
            return None
        if version != PROTOCOL_VERSION:
            # Future-proof: log but don't crash
            return None
        if plen != len(body) - HEADER_SIZE:
            return None  # length mismatch = corruption we missed

        try:
            ptype_enum = PacketType(ptype)
        except ValueError:
            return None  # unknown packet type

        return cls(
            packet_type=ptype_enum,
            src_id=src,
            dst_id=dst,
            seq_num=seq,
            payload=body[HEADER_SIZE:],
        )

    def size_bytes(self) -> int:
        """Total wire size — used for Time-on-Air calculation."""
        return MIN_PACKET_SIZE + len(self.payload)

    def __repr__(self) -> str:
        return (
            f"Packet({self.packet_type.name}, "
            f"src={self.src_id}, dst={self.dst_id}, "
            f"seq={self.seq_num}, payload={len(self.payload)}B)"
        )


# =============================================================================
# PAYLOAD ENCODERS — temperature heartbeat for now
# =============================================================================
def encode_temperature_payload(
    temperature_c: float,
    battery_mv: int,
    last_rssi: int = 0,
) -> bytes:
    """
    Encode a temperature reading payload.

    Layout:
        [sensor_type:1][temperature:int16][battery_mv:uint16][last_rssi:int8]
        Total: 6 bytes

    Temperature is stored as int16 with a 0.01°C scale factor.
    Range: -327.68°C to +327.67°C with 0.01°C resolution.
    More than enough for environmental sensors.

    Battery in millivolts as uint16 — covers 0 to 65,535 mV.
    Last RSSI as signed int8 — covers -128 to +127 dBm (LoRa range fits).
    """
    temp_int = int(round(temperature_c * 100))
    # Clamp to int16 range to prevent struct error on absurd inputs
    temp_int = max(-32768, min(32767, temp_int))
    battery_mv = max(0, min(65535, battery_mv))
    last_rssi = max(-128, min(127, last_rssi))

    # 'b' = signed int8, 'h' = signed int16, 'H' = unsigned int16
    return struct.pack(">BhHb", SensorType.TEMPERATURE, temp_int, battery_mv, last_rssi)


def decode_temperature_payload(payload: bytes) -> Optional[dict]:
    """Parse a temperature payload back into a dict, or None if invalid."""
    if len(payload) != 6:
        return None
    sensor_type, temp_int, battery_mv, last_rssi = struct.unpack(">BhHb", payload)
    if sensor_type != SensorType.TEMPERATURE:
        return None
    return {
        "sensor_type": "temperature",
        "temperature_c": temp_int / 100.0,
        "battery_mv": battery_mv,
        "last_rssi": last_rssi,
    }

def encode_ack_payload(seq_num: int) -> bytes:
    """
    Encode a minimal ACK payload.
    Layout: [seq_num: uint16] — 2 bytes, big-endian.
    uint16 matches the header seq_num field range (0–65535).
    """
    seq_num = max(0, min(65535, seq_num))
    return struct.pack(">H", seq_num)


def decode_ack_payload(payload: bytes) -> Optional[dict]:
    """
    Parse an ACK payload. Returns dict with key 'ack_seq_num', or None if malformed.
    Returns a dict (not a bare int) so callers can use ack_data["ack_seq_num"].
    """
    if len(payload) != 2:
        return None
    seq_num = struct.unpack(">H", payload)[0]
    return {"ack_seq_num": seq_num}


# -----------------------------------------------------------------------------
# Module 5: end-to-end ACK payload
# -----------------------------------------------------------------------------
# Layout: [ack_seq_num: uint16][originator_id: uint8] — 3 bytes.
#
# Why we need originator_id in the payload (and not just in the dst_id header
# field of the encapsulating Packet): on the way back from CONTROL the ACK_E2E
# is forwarded hop-by-hop, so each hop rewrites the Packet's dst_id field to
# the *next hop* address.  The originator_id in the payload tells every relay
# WHO the ack ultimately belongs to, so they can look it up in their
# downstream_to routing table.
ACK_E2E_PAYLOAD_SIZE = 3


def encode_e2e_ack_payload(ack_seq_num: int, originator_id: int) -> bytes:
    """
    Encode an end-to-end ACK payload.
    Layout: [ack_seq_num: uint16][originator_id: uint8] — 3 bytes, big-endian.
    """
    ack_seq_num   = max(0, min(0xFFFF, ack_seq_num))
    originator_id = max(0, min(0xFF, originator_id))
    return struct.pack(">HB", ack_seq_num, originator_id)


def decode_e2e_ack_payload(payload: bytes) -> Optional[dict]:
    """
    Parse an end-to-end ACK payload.  Returns dict with 'ack_seq_num' and
    'originator_id', or None if malformed.
    """
    if len(payload) != ACK_E2E_PAYLOAD_SIZE:
        return None
    ack_seq_num, originator_id = struct.unpack(">HB", payload)
    return {"ack_seq_num": ack_seq_num, "originator_id": originator_id}


def encode_beacon_payload(hop_count: int, network_id: int) -> bytes:
    """
    Encode a beacon payload.
    Layout: [hop_count: uint8][network_id: uint8] — 2 bytes.
    hop_count: distance in hops from CONTROL (0 = CONTROL itself).
    network_id: prevents joining a neighbouring farm's mesh.
    """
    hop_count  = max(0, min(255, hop_count))
    network_id = max(0, min(255, network_id))
    return struct.pack(">BB", hop_count, network_id)


def decode_beacon_payload(payload: bytes) -> Optional[dict]:
    """
    Parse a beacon payload. Returns dict with keys 'hop_count' and 'network_id',
    or None if malformed.
    """
    if len(payload) != 2:
        return None
    hop_count, network_id = struct.unpack(">BB", payload)
    return {"hop_count": hop_count, "network_id": network_id}


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Packets module — self-test")
    print("=" * 70)

    # Build a temperature heartbeat from node 2 to node 1
    payload = encode_temperature_payload(
        temperature_c=24.73, battery_mv=4150, last_rssi=-87
    )
    pkt = Packet(
        packet_type=PacketType.DATA,
        src_id=2, dst_id=1, seq_num=42, payload=payload,
    )
    print(f"\n[1] Built packet:  {pkt}")
    print(f"    Wire size:    {pkt.size_bytes()} bytes")

    # Serialize
    raw = pkt.to_bytes()
    print(f"\n[2] Serialized to {len(raw)} bytes:")
    print(f"    Hex: {raw.hex(' ')}")

    # Round-trip parse
    recovered = Packet.from_bytes(raw)
    print(f"\n[3] Parsed back:   {recovered}")
    decoded_payload = decode_temperature_payload(recovered.payload)
    print(f"    Decoded payload: {decoded_payload}")

    # Corruption test — flip a bit and confirm CRC catches it
    corrupted = bytearray(raw)
    corrupted[5] ^= 0x01    # flip a bit somewhere in the body
    result = Packet.from_bytes(bytes(corrupted))
    print(f"\n[4] Corruption test (1-bit flip): "
          f"{'CORRECTLY REJECTED' if result is None else 'FAILED — got through!'}")

    # Wrong marker test
    bad_marker = bytearray(raw)
    bad_marker[0] = 0xBB
    # Need to recompute CRC for this to actually test marker rejection
    body = bytes(bad_marker[:-2])
    new_crc = crc16(body).to_bytes(2, "big")
    bad_marker[-2:] = new_crc
    result = Packet.from_bytes(bytes(bad_marker))
    print(f"[5] Wrong marker test: "
          f"{'CORRECTLY REJECTED' if result is None else 'FAILED'}")

    print("\n" + "=" * 70)
    print("Packets module ready.")
    print("=" * 70)
