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

## 跌倒检测采集

1. 修改 `hardware/CollectData/CollectData.ino` 里的 WiFi 名、密码和电脑局域网 IP。
2. 烧录 ESP32。
3. 在电脑上启动采集服务。推荐每次采集都用命令行指定标签、采集人和安装位置：

```bash
python data_collection/data_server_labeled.py --label walk --participant p01 --mount-position cane_body
```

4. 常用标签包括 `stand`、`walk`、`sit_down`、`put_down`、`tap`、`fall`。每次采集会生成独立文件，避免不同批次混在一起。
5. 数据会写入 `data/raw/bmi_data_<label>_<时间>.csv`，同时生成一份同名 JSON 元数据，记录标签、采集人、安装位置和备注。

## 模型训练

采集完成后运行：

```bash
cd ml
python3 preprocess.py
```

脚本会读取 `../data/raw/bmi_data_*.csv`，自动跳过没有 `label` 列的 debug 数据，先做数据质检和滑动窗口切分，再评估规则基线并训练两个模型：

- 规则基线：使用“冲击 + 旋转 + 尾部趋静”的可解释规则，方便和模型结果对照。
- RandomForest 对照模型：用于快速 sanity check 和答辩对比，不作为 ESP32 部署主模型。
- 轻量 1D-CNN：默认部署候选模型，可导出 TensorFlow Lite。

默认窗口是 100 个采样点，即 50Hz 下约 2 秒；步长 50，即 50% 重叠。默认把 `fall` 标签作为跌倒类别，其余标签作为正常动作。常用参数示例：

```bash
python3 preprocess.py --window-size 100 --step-size 50 --fall-label fall --epochs 50
```

训练完成后会导出：

- `models/lunacane_model.keras`
- `models/lunacane_model.tflite`
- `models/scaler.pkl`
- `models/scaler_params.h`
- `models/data_quality_report.json`
- `models/rule_baseline_report.json`
- `models/random_forest_report.json`
- `models/cnn_report.json`
- `models/training_summary.json`

其中 `data_quality_report.json` 会记录标签分布、文件分布、采样间隔和异常大间隔数量；`rule_baseline_report.json` 使用“冲击 + 旋转 + 尾部趋静”的可解释规则作为对照；`cnn_report.json` 会记录 1D-CNN 自动选择的跌倒判定阈值，默认优先提高跌倒召回率。

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
