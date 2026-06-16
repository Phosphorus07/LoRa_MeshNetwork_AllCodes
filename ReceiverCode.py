/*
 * LoRa Mesh — Subscriber (TX) Node
 * Heltec Wireless Stick V3 (ESP32-S3 + SX1262)
 * Sends heartbeat packets every 5s.
 *
 * Temperature is now a REAL read from a DS18B20 (KY-001 module) on GPIO 7.
 * Pump status and voltage remain mocked until Malcolm's sensors arrive.
 *
 * Libraries required (install via Arduino Library Manager):
 *   - RadioLib            (Jan Gromeš)
 *   - OneWire             (Paul Stoffregen)
 *   - DallasTemperature   (Miles Burton)
 */

#include <RadioLib.h>
#include <SPI.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ============================================================
// HELTEC WIRELESS STICK V3 PIN MAP (ESP32-S3 + SX1262)
// ============================================================
#define LORA_NSS   8
#define LORA_DIO1  14
#define LORA_RST   12
#define LORA_BUSY  13
#define LORA_SCK   9
#define LORA_MOSI  10
#define LORA_MISO  11
#define VEXT_CTRL  36   // Powers OLED + external 3V3

// ------------------------------------------------------------
// DS18B20 ONE-WIRE BUS
// GPIO 7 is broken out on the J3 header (ADC1_CH6 / TOUCH7).
// NOT GPIO 17 — that is the onboard OLED's I2C SDA line and is
// internally claimed by the display, so it cannot be reused.
// The SX1262 owns GPIO 8-14, so GPIO 7 is clear of both.
// ------------------------------------------------------------
#define ONE_WIRE_BUS    7
#define DS18B20_BITS    12      // resolution: 12-bit = 0.0625°C, ~750ms conversion

// ============================================================
// LoRa RF PARAMETERS — must match between TX and RX
// ============================================================
#define LORA_FREQ       433.0   // MHz
#define LORA_BW         125.0   // kHz
#define LORA_SF         9       // Spreading Factor
#define LORA_CR         5       // Coding Rate 4/5
#define LORA_TX_POWER   14      // dBm (25mW, within ISM limits)
#define LORA_PREAMBLE   8       // symbols
#define LORA_SYNC_WORD  0x12    // Private network (non-LoRaWAN)

// ============================================================
// NODE IDENTITY & TIMING
// ============================================================
#define NODE_ID         0x01    // This subscriber's ID
#define DEST_ID         0xFF    // Control room ID
#define HEARTBEAT_MS    5000    // 5 second heartbeat interval

// Sentinel sent when the DS18B20 is missing / faulted, so the
// control room can distinguish "no reading" from a real 0°C.
// (-32768 = INT16_MIN, i.e. -3276.8°C in 0.1°C units — clearly invalid)
#define TEMP_SENTINEL   INT16_MIN

// ============================================================
// PACKET STRUCTURE — EXTEND THIS WHEN MALCOLM'S SENSORS ARRIVE
// Currently 10 bytes — well under 20-byte design target
// ============================================================
struct __attribute__((packed)) HeartbeatPacket {
  uint8_t  src_id;            // who sent it
  uint8_t  dst_id;            // destination
  uint8_t  packet_type;       // 0x01 = heartbeat
  uint16_t seq_num;           // sequence counter for PDR tracking

  // ----- Sensor data -----
  int16_t  temperature_x10;   // temp in 0.1°C (e.g. 235 = 23.5°C) — REAL (DS18B20)
  uint8_t  pump_status;       // 0 = OFF, 1 = ON                    — mocked
  uint16_t voltage_mv;        // voltage in millivolts              — mocked

  // ----- Future fields (uncomment when sensors are wired) -----
  // uint16_t flow_rate_lpm;    // flow in L/min
  // uint16_t pressure_kpa;     // pressure in kPa
  // uint16_t water_level_cm;   // water level in cm
  // uint16_t current_ma;       // pump current in mA
};

// ============================================================
// GLOBALS
// ============================================================
SX1262 radio = new Module(LORA_NSS, LORA_DIO1, LORA_RST, LORA_BUSY);
uint16_t seq_counter = 0;

OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature ds18b20(&oneWire);
bool temp_sensor_ok = false;   // set in setup() after probing the bus

// ============================================================
// SENSOR FUNCTIONS
// ============================================================

// Real DS18B20 read. Returns temperature in 0.1°C units, or
// TEMP_SENTINEL if the sensor is disconnected / not found.
//
// NOTE on timing: requestTemperatures() blocks for ~750ms at 12-bit
// (waitForConversion defaults to true). At a 5s heartbeat that's a
// harmless ~15% of the idle window. If you later move to a faster
// heartbeat or want the loop free during conversion, switch to the
// async pattern: setWaitForConversion(false), request on cycle N,
// read getTempCByIndex() on cycle N+1. Worth a sentence in the logbook.
int16_t readTemperature() {
  if (!temp_sensor_ok) return TEMP_SENTINEL;

  ds18b20.requestTemperatures();                 // blocking ~750ms @ 12-bit
  float tempC = ds18b20.getTempCByIndex(0);      // first device on the bus

  if (tempC == DEVICE_DISCONNECTED_C) {
    Serial.println("[WARN] DS18B20 read failed (disconnected?)");
    return TEMP_SENTINEL;
  }

  // Convert °C -> 0.1°C, rounding to nearest tenth before truncation.
  return (int16_t)(tempC * 10.0f + (tempC >= 0 ? 0.5f : -0.5f));
}

uint8_t readPumpStatus() {
  // Mock: toggle every 30 seconds
  return (millis() / 30000) % 2;
}

uint16_t readVoltage() {
  // Mock: ~12V with small variation
  return 12000 + random(-200, 200);
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1500);  // wait for USB serial to enumerate
  Serial.println("\n=== LoRa TX Node — Heltec Wireless Stick V3 ===");

  // Power up Vext (OLED + peripherals, LOW = ON on Heltec V3)
  pinMode(VEXT_CTRL, OUTPUT);
  digitalWrite(VEXT_CTRL, LOW);
  delay(100);

  // ---- DS18B20 init ----
  // The KY-001 module carries its own 4.7k pull-up (R1), so no
  // external resistor is needed on the data line.
  Serial.print("DS18B20 init... ");
  ds18b20.begin();
  ds18b20.setResolution(DS18B20_BITS);
  int device_count = ds18b20.getDeviceCount();

  if (device_count > 0) {
    temp_sensor_ok = true;
    Serial.print("OK (");
    Serial.print(device_count);
    Serial.println(" device(s) on bus)");
  } else {
    temp_sensor_ok = false;
    Serial.println("NONE FOUND — check wiring on GPIO 7 / pull-up");
    // Not fatal: node keeps transmitting and sends TEMP_SENTINEL so
    // the failure is visible at the control room rather than silent.
  }

  // Initialize SPI with Heltec V3 LoRa pins (not ESP32-S3 defaults)
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_NSS);

  // Initialize the radio
  Serial.print("Radio init... ");
  int state = radio.begin(LORA_FREQ, LORA_BW, LORA_SF, LORA_CR,
                          LORA_SYNC_WORD, LORA_TX_POWER, LORA_PREAMBLE);

  if (state == RADIOLIB_ERR_NONE) {
    Serial.println("OK");
  } else {
    Serial.print("FAILED, error code ");
    Serial.println(state);
    while (true) { delay(1000); }  // halt — check wiring/board version
  }

  Serial.print("Packet size: ");
  Serial.print(sizeof(HeartbeatPacket));
  Serial.println(" bytes");
  Serial.println("Starting heartbeat transmission...\n");
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
  // ---- Build packet ----
  HeartbeatPacket pkt;
  pkt.src_id          = NODE_ID;
  pkt.dst_id          = DEST_ID;
  pkt.packet_type     = 0x01;
  pkt.seq_num         = seq_counter++;
  pkt.temperature_x10 = readTemperature();
  pkt.pump_status     = readPumpStatus();
  pkt.voltage_mv      = readVoltage();

  // ---- Log to serial (so you can see what's being sent) ----
  Serial.print("[TX] seq=");
  Serial.print(pkt.seq_num);
  Serial.print("  temp=");
  if (pkt.temperature_x10 == TEMP_SENTINEL) {
    Serial.print("--.-");          // no valid reading
  } else {
    Serial.print(pkt.temperature_x10 / 10.0, 1);
  }
  Serial.print("°C  pump=");
  Serial.print(pkt.pump_status ? "ON " : "OFF");
  Serial.print("  V=");
  Serial.print(pkt.voltage_mv / 1000.0, 2);
  Serial.print("V  ... ");

  // ---- Transmit ----
  unsigned long tx_start = millis();
  int state = radio.transmit((uint8_t*)&pkt, sizeof(pkt));
  unsigned long tx_time = millis() - tx_start;

  if (state == RADIOLIB_ERR_NONE) {
    Serial.print("sent (");
    Serial.print(tx_time);
    Serial.println(" ms)");
  } else {
    Serial.print("FAIL code ");
    Serial.println(state);
  }

  delay(HEARTBEAT_MS);
}