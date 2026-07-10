# LunaCane Complete

LunaCane Complete 是从旧的 `da-chuang-crutch`、`LunaCane_Algorithm` 和 `LunaCane_Algorithm - 副本` 整理出的完整智能拐杖仓库。

它包含两条主线：

1. 跌倒检测数据采集与 TinyML 训练链路
2. Luna 语音助手的 ESP32 录音、服务器问答、TTS 回放链路

## 目录结构

```text
data_collection/
  data_server_labeled.py   # 带动作标签的 BMI270 数据采集服务
  data_server_debug.py     # 无标签 debug 采集服务，默认最多收 3000 条
hardware/
  CollectData/             # ESP32 + BMI270 采集 IMU 数据
  VoiceAssistant/          # ESP32 I2S 麦克风录音、上传、下载并播放回复
ml/
  preprocess.py            # 合并 CSV、标准化、训练 1D-CNN、导出 TFLite
voice/
  main_server.py           # FastAPI 语音服务
  luna_core.py             # ASR -> LLM -> TTS -> wav
docs/
  任务纲领.md
  实习日记提纲-2026-07-06.md
  故障排查记录.md
config/
```

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

语音服务需要 `ffmpeg`。推荐把 `ffmpeg` 加入系统 PATH；也可以在 `.env` 里设置 `LUNACANE_FFMPEG_PATH`。

依赖版本在 `requirements.txt` 中使用范围约束；TensorFlow 固定为 `2.21.0`，以便本地环境和 GitHub Actions 使用同一训练/转换基础。

快速验证：

```bash
python -m compileall data_collection ml voice tools tests
python -m unittest discover -s tests
```

## 无硬件开发与模拟

新硬件到货前，可以用电脑模拟 ESP32 请求。先启动对应服务，再运行模拟器。

模拟 BMI270 以 50Hz、每批 20 条回放 CSV：

```bash
python tools/simulate_imu.py data/sample/bmi_data_walk.csv
```

加上 `--no-realtime` 可以取消批次间等待，用于快速接口测试。服务地址可用 `--url` 修改。

模拟语音固件上传 WAV，并把服务器返回的纯 PCM 保存到本地：

```bash
python tools/simulate_voice.py input.wav --output reply.pcm
```

语音路由测试使用 mock 流程，不需要真实的阿里云或大模型密钥。

## 跌倒检测采集

1. 复制 `hardware/CollectData/config.example.h` 为同目录下的 `config.h`，填写 WiFi 和电脑局域网 IP。
2. 烧录 ESP32。
3. 在电脑上启动采集服务：

```bash
python data_collection/data_server_labeled.py
```

4. 采集不同动作前，修改 `data_collection/data_server_labeled.py` 里的 `CURRENT_LABEL`。常用标签包括 `stand`、`walk`、`sit_down`、`put_down`、`tap`、`fall`。
5. 数据会写入 `data/raw/bmi_data_<label>.csv`。

CSV 文件至少需要包含以下列：

```text
timestamp, ax, ay, az, gx, gy, gz, acc_mag, gyro_mag, label
```

`data/sample/bmi_data_walk.csv` 提供了一个最小格式样例，仅用于确认列名和数据形态，不用于真实训练。

## 模型训练

采集完成后运行：

```bash
cd ml
python3 preprocess.py
```

脚本会读取 `../data/raw/bmi_data_*.csv`，自动跳过没有 `label` 列的 debug 数据，先做数据质检和滑动窗口切分，再训练两个模型：

- RandomForest 对照模型：用于快速 sanity check 和答辩对比，不作为 ESP32 部署主模型。
- 轻量 1D-CNN：默认部署候选模型，可导出 TensorFlow Lite。

默认窗口是 100 个采样点，即 50Hz 下约 2 秒；步长 50，即 50% 重叠。常用参数示例：

```bash
python3 preprocess.py --window-size 100 --step-size 50 --epochs 50
```

训练完成后会导出：

- `models/lunacane_model.keras`
- `models/lunacane_model.tflite`
- `models/scaler.pkl`
- `models/scaler_params.h`
- `models/data_quality_report.json`
- `models/random_forest_report.json`
- `models/cnn_report.json`
- `models/training_summary.json`
- `models/model_metadata.json`
- `models/model_config.h`

`model_metadata.json` 和 `model_config.h` 固化了采样率、窗口大小、特征顺序、输入输出形状和判定阈值。部署时应同时使用 TFLite、`scaler_params.h` 和 `model_config.h`，避免固件参数与训练报告不一致。

## 语音助手

复制 `.env.example` 为 `.env`，填入大模型和阿里云 NLS 配置。

```bash
cp .env.example .env
```

启动服务：

```bash
python voice/main_server.py
```

然后复制 `hardware/VoiceAssistant/config.example.h` 为同目录下的 `config.h`，填写 WiFi 和电脑局域网 IP，再烧录 ESP32。按住按钮录音，松开后 ESP32 会上传音频；服务器完成 ASR、大模型回答和 TTS 后，ESP32 下载音频并播放。

两个固件由 PlatformIO CI 做无硬件编译检查。本地安装 PlatformIO 后也可运行：

```bash
cp hardware/CollectData/config.example.h hardware/CollectData/config.h
cp hardware/VoiceAssistant/config.example.h hardware/VoiceAssistant/config.h
platformio run
```

## 注意事项

- ESP32 只支持 2.4GHz WiFi 热点。
- `host` 建议保持 `0.0.0.0`，ESP32 端 URL 填电脑当前局域网 IP。
- 不要提交 `.env`、历史音频、模型输出、虚拟环境和 ffmpeg 二进制。
- 供电不稳会导致 WiFi 掉线、BMI270 读失败或 ESP32 brownout，详见 `docs/故障排查记录.md`。
