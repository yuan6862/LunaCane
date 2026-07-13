#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <SparkFun_BMI270_Arduino_Library.h>
#include "config.h"

#define SDA_PIN 1
#define SCL_PIN 2

const char* ssid = LUNACANE_WIFI_SSID;
const char* password = LUNACANE_WIFI_PASSWORD;
const char* serverUrl = LUNACANE_SENSOR_URL;

BMI270 imu;

struct Sample {
  unsigned long t;
  float ax, ay, az;
  float gx, gy, gz;
};

const int BATCH_SIZE = 20;   // 一次发20条
Sample buffer[BATCH_SIZE];
int sampleCount = 0;

unsigned long lastSampleTime = 0;
const unsigned long sampleIntervalMs = 20;  // 50Hz

void connectWiFi() {
  Serial.print("Connecting WiFi");
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

bool initIMU() {
  Wire.begin(SDA_PIN, SCL_PIN);

  int8_t err = imu.beginI2C(BMI2_I2C_PRIM_ADDR, Wire);
  if (err != BMI2_OK) {
    Serial.print("BMI270 init failed, err = ");
    Serial.println(err);
    return false;
  }

  Serial.println("BMI270 init OK");
  return true;
}

void sendBatch() {
  if (sampleCount == 0) return;

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected, reconnecting");
    WiFi.reconnect();
    return;
  }

  String json = "{\"samples\":[";
  for (int i = 0; i < sampleCount; i++) {
    json += "{";
    json += "\"t\":" + String(buffer[i].t) + ",";
    json += "\"ax\":" + String(buffer[i].ax, 5) + ",";
    json += "\"ay\":" + String(buffer[i].ay, 5) + ",";
    json += "\"az\":" + String(buffer[i].az, 5) + ",";
    json += "\"gx\":" + String(buffer[i].gx, 5) + ",";
    json += "\"gy\":" + String(buffer[i].gy, 5) + ",";
    json += "\"gz\":" + String(buffer[i].gz, 5);
    json += "}";
    if (i < sampleCount - 1) json += ",";
  }
  json += "]}";

  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");

  int code = http.POST(json);
  if (code > 0) {
    String resp = http.getString();
    Serial.print("POST code: ");
    Serial.println(code);
    Serial.println(resp);
    sampleCount = 0;  // 发送成功才清空
  } else {
    Serial.print("POST failed, code = ");
    Serial.println(code);
  }

  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("BMI270 WiFi uploader start...");

  connectWiFi();

  if (!initIMU()) {
    while (1) delay(1000);
  }

  Serial.println("Start sampling...");
}

void loop() {
  unsigned long now = millis();

  if (now - lastSampleTime >= sampleIntervalMs) {
    lastSampleTime = now;

    if (imu.getSensorData() == BMI2_OK) {
      buffer[sampleCount].t  = now;
      buffer[sampleCount].ax = imu.data.accelX;
      buffer[sampleCount].ay = imu.data.accelY;
      buffer[sampleCount].az = imu.data.accelZ;
      buffer[sampleCount].gx = imu.data.gyroX;
      buffer[sampleCount].gy = imu.data.gyroY;
      buffer[sampleCount].gz = imu.data.gyroZ;

      sampleCount++;

      Serial.printf("Buffered %d | %lu, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n",
                    sampleCount,
                    now,
                    imu.data.accelX, imu.data.accelY, imu.data.accelZ,
                    imu.data.gyroX, imu.data.gyroY, imu.data.gyroZ);

      if (sampleCount >= BATCH_SIZE) {
        sendBatch();
      }
    } else {
      Serial.println("BMI270 read failed");
    }
  }
}
