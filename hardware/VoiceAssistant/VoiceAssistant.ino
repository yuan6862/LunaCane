#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "driver/i2s.h"

// ================= 配置区 =================
const char* ssid = "YOUR_2G_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* serverUrl = "http://192.168.1.100:8000/upload_audio";
const char* downloadBaseUrl = "http://192.168.1.100:8000/get_audio/";

#define BUTTON_PIN     13   // 预留给开关的引脚 (GPIO 13)

// 麦克风引脚 (I2S_NUM_0)
#define MIC_I2S_SCK    7
#define MIC_I2S_WS     15
#define MIC_I2S_SD     16

// 喇叭引脚 (I2S_NUM_1)
#define SPK_I2S_BCK    5
#define SPK_I2S_WS     6
#define SPK_I2S_DO     4

#define SAMPLE_RATE    16000
#define MAX_RECORD_SECONDS 5 // 增加到5秒，防止说话长
#define MAX_PCM_BYTES (SAMPLE_RATE * 2 * MAX_RECORD_SECONDS)

uint8_t *recordBuffer = nullptr;
size_t recordBytes = 0;
bool isRecording = false;

void sendPostAndDownload() {
    Serial.println(">>> 正在上传音频数据...");
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "audio/wav");
    http.setTimeout(200000);

    // 上传数据：44字节头 + 录音PCM数据
    int httpCode = http.POST(recordBuffer, recordBytes + 44);

    if (httpCode == 200) {
        String payload = http.getString();
        Serial.print("服务器返回: ");
        Serial.println(payload);

        String fileName = getFileName(payload);
        if (fileName != "" && fileName != "none") {
            Serial.print("准备播放回答: ");
            Serial.println(fileName);
            downloadAndPlay(fileName);
        } else {
            Serial.println("未识别到文字或无有效回答");
        }
    } else {
        Serial.printf("上传失败,HTTP错误码: %d\n", httpCode);
    }
    http.end();
}

// ================= I2S 驱动管理 =================
void initMic() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .dma_buf_count = 8,
    .dma_buf_len = 256
  };
  i2s_pin_config_t pins = {.bck_io_num = MIC_I2S_SCK, .ws_io_num = MIC_I2S_WS, .data_out_num = I2S_PIN_NO_CHANGE, .data_in_num = MIC_I2S_SD};
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
}

void initSpeaker() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .dma_buf_count = 8,
    .dma_buf_len = 256
  };
  i2s_pin_config_t pins = {.bck_io_num = SPK_I2S_BCK, .ws_io_num = SPK_I2S_WS, .data_out_num = SPK_I2S_DO, .data_in_num = I2S_PIN_NO_CHANGE};
  i2s_driver_install(I2S_NUM_1, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_1, &pins);
}

// ================= 播放与录音逻辑 =================

void downloadAndPlay(String fileName) {
  i2s_stop(I2S_NUM_0); // 停止麦克风
  
  // 重新配置 I2S_NUM_1 为播放模式
  initSpeaker();
  i2s_zero_dma_buffer(I2S_NUM_1); 
  
  HTTPClient http;
  String url = downloadBaseUrl + fileName;
  http.begin(url);
  int httpCode = http.GET();
  http.setTimeout(20000);
  
  if (httpCode == 200) {
    Serial.println(">>> 正在下载并播放音频流...");
    int len = http.getSize();
    WiFiClient* stream = http.getStreamPtr();
    
    uint8_t buffer[1024];
    // 关键：如果你听到的声音像碎纸声，说明采样率不匹配。
    // 因为 edge-tts 默认可能是 24000Hz 或更高，
    // 如果声音太慢，尝试在 initSpeaker 里把 SAMPLE_RATE 改成 24000。
    
    while (http.connected() && (len > 0 || stream->available())) {
      size_t size = stream->available();
      if (size > 0) {
        int c = stream->readBytes(buffer, min((int)size, (int)sizeof(buffer)));
        size_t written;
        // 直接写入 I2S 
        i2s_write(I2S_NUM_1, buffer, c, &written, portMAX_DELAY);
        len -= c;
      }
      yield(); // 防止 ESP32 触发看门狗重启
    }
  } else {
    Serial.printf(">>> 下载音频失败，错误码: %d\n", httpCode);
  }
  
  http.end();
  delay(200); // 留点余量播完最后一段
  i2s_zero_dma_buffer(I2S_NUM_1);
  i2s_driver_uninstall(I2S_NUM_1); // 卸载喇叭驱动
  i2s_start(I2S_NUM_0); // 恢复麦克风
  Serial.println(">>> 播放结束，恢复录音就绪");
}

// 简单的 JSON 解析文件名
String getFileName(String json) {
  int start = json.indexOf("\"file\":\"") + 8;
  if (start < 8) return "";
  int end = json.indexOf("\"", start);
  if (end < 0) return "";
  return json.substring(start, end);
}

void writeWavHeader(uint8_t* header, int wavDataSize) {
    header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
    uint32_t fileSize = wavDataSize + 44 - 8;
    memcpy(&header[4], &fileSize, 4);
    header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
    header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
    uint32_t fmtSize = 16; memcpy(&header[16], &fmtSize, 4);
    uint16_t audioFormat = 1; memcpy(&header[20], &audioFormat, 2);
    uint16_t numChannels = 1; memcpy(&header[22], &numChannels, 2);
    uint32_t sampleRate = SAMPLE_RATE; memcpy(&header[24], &sampleRate, 4);
    uint32_t byteRate = SAMPLE_RATE * 2; memcpy(&header[28], &byteRate, 4);
    uint16_t blockAlign = 2; memcpy(&header[32], &blockAlign, 2);
    uint16_t bitsPerSample = 16; memcpy(&header[34], &bitsPerSample, 2);
    header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
    memcpy(&header[40], &wavDataSize, 4);
}

// ================= 主循环 =================

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP); // 重要：配置内部上拉
  
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  
  recordBuffer = (uint8_t*)malloc(MAX_PCM_BYTES + 44);
  initMic();
  Serial.println("系统就绪：按住开关录音，松开结束。");
}

// 增加这些全局变量用于消抖
unsigned long lastDebounceTime = 0;  
const unsigned long debounceDelay = 50; // 50ms 消抖
bool lastRawButtonState = HIGH;         // 记录上一次物理电平
bool confirmedState = HIGH;             // 最终确认的稳定电平

void loop() {
  bool currentRawState = digitalRead(BUTTON_PIN);

  // --- 1. 软件消抖核心逻辑 ---
  if (currentRawState != lastRawButtonState) {
    lastDebounceTime = millis(); // 只要电平动了，就重置计时器
  }

  // 只有当电平稳定了 50ms 以上，才认为动作有效
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (currentRawState != confirmedState) {
      confirmedState = currentRawState;

      // 情况 A：确认按下 (LOW)
      if (confirmedState == LOW && !isRecording) {
        Serial.println(">>> 按钮确认按下: 开始录音");
        isRecording = true;
        recordBytes = 0;
        // 建议在这里清除一下 DMA 缓存，保证录音开头没有杂音
        i2s_zero_dma_buffer(I2S_NUM_0);
      } 
      // 情况 B：确认松开 (HIGH)
      else if (confirmedState == HIGH && isRecording) {
        Serial.println(">>> 按钮确认松开: 停止并上传");
        isRecording = false; 
        
        // 执行上传流程
        writeWavHeader(recordBuffer, recordBytes);
        sendPostAndDownload(); // 这个函数里包含了 POST 和 GET 播放
      }
    }
  }

  // --- 2. 录音采样逻辑 (必须在消抖逻辑之外，保证采样频率) ---
  if (isRecording) {
    if (recordBytes < MAX_PCM_BYTES) {
      size_t readLen;
      int32_t raw[128]; 
      // 使用 portMAX_DELAY 确保读取完整
      i2s_read(I2S_NUM_0, raw, sizeof(raw), &readLen, portMAX_DELAY);
      
      for (int i=0; i < readLen/4; i++) {
        // 这里的右移 14 位是针对 INMP441 等 24/32位麦克风转 16位 PCM 的常用处理
        int16_t pcm = (int16_t)(raw[i] >> 14); 
        memcpy(recordBuffer + 44 + recordBytes, &pcm, 2);
        recordBytes += 2;
      }
    } else {
      // 超过最大录音时间，强制停止
      isRecording = false;
      Serial.println(">>> 录音达到上限，自动上传");
      writeWavHeader(recordBuffer, recordBytes);
      sendPostAndDownload();
    }
  }

  lastRawButtonState = currentRawState; // 更新物理状态记录
}
