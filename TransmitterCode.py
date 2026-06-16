/*
 * LoRa Mesh — Control (RX) Node
 * Heltec Wireless Stick V3 (ESP32-S3 + SX1262)
 *
 * Receives heartbeat packets from subscriber, displays on built-in OLED.
 * Three screens cycle every 3 seconds:
 *   Screen 1: Temperature
 *   Screen 2: Link health (RSSI, SNR)
 *   Screen 3: Reliability (PDR%)
 *
 * Metrics:
 *   - Rolling PDR over last 20 packets (sliding window)
 *   - Sequence gap detection (counts missed packets)
 */

#include <RadioLib.h>
#include <SPI.h>
#include <Wire.h>
#include <U8g2lib.h>

// ============================================================
// HELTEC WIRELESS STICK V3 PIN MAP
// ============================================================
// LoRa SX1262
#define LORA_NSS   8
#define LORA_DIO1  14
#define LORA_RST   12
#define LORA_BUSY  13
#define LORA_SCK   9
#define LORA_MOSI  10
#define LORA_MISO  11

// OLED SSD1306 (64x32)
#define OLED_SDA   17
#define OLED_SCL   18
#define OLED_RST   21

// Vext (powers OLED + peripherals, LOW = ON)
#define VEXT_CTRL  36

// ============================================================
// LoRa RF PARAMETERS — MUST MATCH TX EXACTLY
// ============================================================
#define LORA_FREQ       433.0
#define LORA_BW         125.0
#define LORA_SF         9
#define LORA_CR         5
#define LORA_TX_POWER   14
#define LORA_PREAMBLE   8
#define LORA_SYNC_WORD  0x12

// ============================================================
// NODE IDENTITY
// ============================================================
#define NODE_ID         0xFF    // This control node's ID
#define EXPECTED_SRC    0x01    // Subscriber we expect packets from

// ============================================================
// DISPLAY CYCLING
// ============================================================
#define SCREEN_CYCLE_MS  3000   // 3 seconds per screen
#define NUM_SCREENS      3

// ============================================================
// PDR ROLLING WINDOW
// ============================================================
#define PDR_WINDOW       20     // last N packets to calculate PDR

// Sentinel value sent by TX when its DS18B20 is missing/faulted.
// MUST match the TX definition (INT16_MIN).
#define TEMP_SENTINEL    INT16_MIN

// ============================================================
// PACKET STRUCTURE — MUST MATCH TX EXACTLY
// ============================================================
struct __attribute__((packed)) HeartbeatPacket {
  uint8_t  src_id;
  uint8_t  dst_id;
  uint8_t  packet_type;
  uint16_t seq_num;
  int16_t  temperature_x10;
  uint8_t  pump_status;
  uint16_t voltage_mv;
};

// ============================================================
// GLOBALS
// ============================================================
SX1262 radio = new Module(LORA_NSS, LORA_DIO1, LORA_RST, LORA_BUSY);

// Use hardware I2C with custom pins for the Heltec OLED
U8G2_SSD1306_64X32_1F_F_HW_I2C u8g2(U8G2_R0, OLED_RST, OLED_SCL, OLED_SDA);

// ---- Receive state ----
volatile bool packet_received_flag = false;

// ---- Latest packet data (for display) ----
HeartbeatPacket last_pkt;
float    last_rssi = 0;
float    last_snr  = 0;
uint16_t last_seq  = 0;
bool     first_packet = true;

// ---- Statistics ----
uint32_t packets_received = 0;
uint32_t packets_missed   = 0;     // detected via sequence gaps
uint32_t expected_count   = 0;     // total expected based on seq numbers

// Rolling window: 1 = received, 0 = missed
uint8_t  pdr_window[PDR_WINDOW] = {0};
uint8_t  pdr_index = 0;
uint8_t  pdr_filled = 0;

// ---- Display state ----
unsigned long last_screen_switch = 0;
uint8_t current_screen = 0;
unsigned long last_display_update = 0;

// ============================================================
// ISR — flag when packet arrives (don't do work in ISR)
// ============================================================
IRAM_ATTR void onReceive() {
  packet_received_flag = true;
}

// ============================================================
// SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n=== LoRa RX Node — Heltec Wireless Stick V3 ===");

  // Power up Vext (needed for OLED)
  pinMode(VEXT_CTRL, OUTPUT);
  digitalWrite(VEXT_CTRL, LOW);
  delay(100);

  // Reset OLED
  pinMode(OLED_RST, OUTPUT);
  digitalWrite(OLED_RST, LOW);
  delay(50);
  digitalWrite(OLED_RST, HIGH);
  delay(50);

  // Init OLED
  u8g2.begin();
  u8g2.setFont(u8g2_font_6x10_tf);  // small but readable
  u8g2.clearBuffer();
  u8g2.drawStr(0, 10, "Boot...");
  u8g2.sendBuffer();

  // Init SPI for LoRa
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_NSS);

  // Init LoRa radio
  Serial.print("Radio init... ");
  int state = radio.begin(LORA_FREQ, LORA_BW, LORA_SF, LORA_CR,
                          LORA_SYNC_WORD, LORA_TX_POWER, LORA_PREAMBLE);
  if (state == RADIOLIB_ERR_NONE) {
    Serial.println("OK");
  } else {
    Serial.print("FAILED, code ");
    Serial.println(state);
    u8g2.clearBuffer();
    u8g2.drawStr(0, 10, "RADIO");
    u8g2.drawStr(0, 22, "FAIL");
    u8g2.sendBuffer();
    while (true) { delay(1000); }
  }

  // Set interrupt for packet reception
  radio.setPacketReceivedAction(onReceive);

  // Start listening
  state = radio.startReceive();
  if (state == RADIOLIB_ERR_NONE) {
    Serial.println("Listening for packets...\n");
  } else {
    Serial.print("startReceive FAIL code ");
    Serial.println(state);
  }

  u8g2.clearBuffer();
  u8g2.drawStr(0, 10, "Listen");
  u8g2.drawStr(0, 22, "...");
  u8g2.sendBuffer();

  last_screen_switch = millis();
}

// ============================================================
// PDR CALCULATION (rolling window)
// ============================================================
void recordPDR(bool received) {
  pdr_window[pdr_index] = received ? 1 : 0;
  pdr_index = (pdr_index + 1) % PDR_WINDOW;
  if (pdr_filled < PDR_WINDOW) pdr_filled++;
}

float calculatePDR() {
  if (pdr_filled == 0) return 0.0;
  uint16_t sum = 0;
  for (uint8_t i = 0; i < pdr_filled; i++) sum += pdr_window[i];
  return (100.0 * sum) / pdr_filled;
}

// ============================================================
// HANDLE INCOMING PACKET
// ============================================================
void handlePacket() {
  HeartbeatPacket pkt;
  int state = radio.readData((uint8_t*)&pkt, sizeof(pkt));

  if (state == RADIOLIB_ERR_NONE) {
    last_pkt  = pkt;
    last_rssi = radio.getRSSI();
    last_snr  = radio.getSNR();

    // ---- Sequence gap detection ----
    if (first_packet) {
      first_packet = false;
      expected_count = 1;
    } else {
      uint16_t gap = pkt.seq_num - last_seq;
      if (gap > 1) {
        // missed (gap - 1) packets
        for (uint16_t i = 0; i < gap - 1; i++) {
          recordPDR(false);
          packets_missed++;
          expected_count++;
        }
      }
      expected_count++;
    }
    last_seq = pkt.seq_num;

    recordPDR(true);
    packets_received++;

    // ---- Serial log ----
    Serial.print("[RX] seq=");
    Serial.print(pkt.seq_num);
    Serial.print("  temp=");
    Serial.print(pkt.temperature_x10 / 10.0, 1);
    Serial.print("C  pump=");
    Serial.print(pkt.pump_status ? "ON " : "OFF");
    Serial.print("  V=");
    Serial.print(pkt.voltage_mv / 1000.0, 2);
    Serial.print("  RSSI=");
    Serial.print(last_rssi, 0);
    Serial.print("  SNR=");
    Serial.print(last_snr, 1);
    Serial.print("  PDR=");
    Serial.print(calculatePDR(), 1);
    Serial.println("%");
  } else {
    Serial.print("[RX] readData FAIL code ");
    Serial.println(state);
  }

  // Re-arm the receiver
  radio.startReceive();
}

// ============================================================
// DISPLAY — OLED is 64x32, ~3 rows at font 6x10
// Shows ONLY: temperature, RSSI/SNR, PDR
// ============================================================
void drawScreen() {
  u8g2.clearBuffer();
  char buf[16];

  switch (current_screen) {
    case 0:  // ---------- TEMPERATURE ----------
      if (first_packet) {
        u8g2.drawStr(0, 12, "Temp");
        u8g2.drawStr(0, 26, "no data");
      } else if (last_pkt.temperature_x10 == TEMP_SENTINEL) {
        u8g2.drawStr(0, 12, "Temp");
        u8g2.drawStr(0, 26, "--.- C");   // sensor fault on TX side
      } else {
        u8g2.drawStr(0, 12, "Temp");
        snprintf(buf, sizeof(buf), "%.1f C", last_pkt.temperature_x10 / 10.0);
        u8g2.drawStr(0, 26, buf);
      }
      break;

    case 1:  // ---------- LINK HEALTH (RSSI + SNR) ----------
      if (first_packet) {
        u8g2.drawStr(0, 12, "Link");
        u8g2.drawStr(0, 26, "no data");
      } else {
        snprintf(buf, sizeof(buf), "RSSI%d", (int)last_rssi);
        u8g2.drawStr(0, 12, buf);
        snprintf(buf, sizeof(buf), "SNR %.1f", last_snr);
        u8g2.drawStr(0, 26, buf);
      }
      break;

    case 2:  // ---------- RELIABILITY (PDR) ----------
      u8g2.drawStr(0, 12, "PDR");
      snprintf(buf, sizeof(buf), "%.0f %%", calculatePDR());
      u8g2.drawStr(0, 26, buf);
      break;
  }

  u8g2.sendBuffer();
}

// ============================================================
// MAIN LOOP
// ============================================================
void loop() {
  // ---- 1. Check for received packet (flag set by ISR) ----
  if (packet_received_flag) {
    packet_received_flag = false;
    handlePacket();
  }

  // ---- 2. Cycle display every 3 seconds ----
  unsigned long now = millis();
  if (now - last_screen_switch >= SCREEN_CYCLE_MS) {
    current_screen = (current_screen + 1) % NUM_SCREENS;
    last_screen_switch = now;
    drawScreen();
  }

  // ---- 3. Refresh current screen every 500ms to update live values ----
  if (now - last_display_update >= 500) {
    last_display_update = now;
    drawScreen();
  }
}