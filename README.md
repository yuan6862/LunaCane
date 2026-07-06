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

## 跌倒检测采集

1. 修改 `hardware/CollectData/CollectData.ino` 里的 WiFi 名、密码和电脑局域网 IP。
2. 烧录 ESP32。
3. 在电脑上启动采集服务：

```bash
python data_collection/data_server_labeled.py
```

4. 采集不同动作前，修改 `data_collection/data_server_labeled.py` 里的 `CURRENT_LABEL`。常用标签包括 `stand`、`walk`、`sit_down`、`put_down`、`tap`、`fall`。
5. 数据会写入 `data/raw/bmi_data_<label>.csv`。

## 模型训练

采集完成后运行：

```bash
cd ml
python preprocess.py
```

脚本会读取 `../data/raw/bmi_data_*.csv`，训练轻量 1D-CNN，并导出：

- `models/lunacane_model.keras`
- `models/lunacane_model.tflite`
- `models/scaler.pkl`
- `models/scaler_params.h`

## 语音助手

复制 `.env.example` 为 `.env`，填入大模型和阿里云 NLS 配置。

```bash
cp .env.example .env
```

启动服务：

```bash
python voice/main_server.py
```

然后修改 `hardware/VoiceAssistant/VoiceAssistant.ino` 里的 WiFi 名、密码和电脑局域网 IP，烧录 ESP32。按住按钮录音，松开后 ESP32 会上传音频；服务器完成 ASR、大模型回答和 TTS 后，ESP32 下载音频并播放。

## 注意事项

- ESP32 只支持 2.4GHz WiFi 热点。
- `host` 建议保持 `0.0.0.0`，ESP32 端 URL 填电脑当前局域网 IP。
- 不要提交 `.env`、历史音频、模型输出、虚拟环境和 ffmpeg 二进制。
- 供电不稳会导致 WiFi 掉线、BMI270 读失败或 ESP32 brownout，详见 `docs/故障排查记录.md`。
