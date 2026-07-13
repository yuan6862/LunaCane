from flask import Flask, request, jsonify
import argparse
import csv
import json
import os
import math
import re
from datetime import datetime

app = Flask(__name__)

# ===============================
# 当前采集动作标签
# 推荐通过命令行指定，例如:
# python data_collection/data_server_labeled.py --label fall --participant p01
# 可选示例：stand / walk / sit_down / put_down / tap / fall
# ===============================
CURRENT_LABEL = "walk"
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
PARTICIPANT = "unknown"
MOUNT_POSITION = "unknown"
NOTE = ""
STANDARD_LABELS = ("stand", "walk", "sit_down", "put_down", "tap", "fall")
LABEL_ALIASES = {
    "sitdown": "sit_down",
    "sit-down": "sit_down",
    "putdown": "put_down",
    "put-down": "put_down",
}

# CSV 统一存放到项目 data/raw/ 目录，供 ml/preprocess.py 直接读取
_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CSV_FILE = os.path.join(_RAW_DIR, f"bmi_data_{CURRENT_LABEL}_{SESSION_ID}.csv")
META_FILE = os.path.join(_RAW_DIR, f"bmi_data_{CURRENT_LABEL}_{SESSION_ID}.json")

HEADER = [
    "pc_time",
    "esp_time_ms",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "acc_mag",
    "gyro_mag",
    "label",
    "session_id",
    "participant",
    "mount_position",
    "note",
]


def sanitize_label(value):
    """Keep labels filename-safe and consistent for training."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value.strip().lower())
    cleaned = cleaned.strip("_") or "unknown"
    return LABEL_ALIASES.get(cleaned, cleaned)


def configure_session(label, participant, mount_position, note, output_dir):
    global CURRENT_LABEL, SESSION_ID, PARTICIPANT, MOUNT_POSITION, NOTE, CSV_FILE, META_FILE, _RAW_DIR

    CURRENT_LABEL = sanitize_label(label)
    PARTICIPANT = participant.strip() or "unknown"
    MOUNT_POSITION = mount_position.strip() or "unknown"
    NOTE = note.strip()
    _RAW_DIR = output_dir
    SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_FILE = os.path.join(_RAW_DIR, f"bmi_data_{CURRENT_LABEL}_{SESSION_ID}.csv")
    META_FILE = os.path.join(_RAW_DIR, f"bmi_data_{CURRENT_LABEL}_{SESSION_ID}.json")


def init_csv():
    """创建本次采集 CSV 和配套元数据文件。"""
    os.makedirs(_RAW_DIR, exist_ok=True)
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
    metadata = {
        "session_id": SESSION_ID,
        "label": CURRENT_LABEL,
        "participant": PARTICIPANT,
        "mount_position": MOUNT_POSITION,
        "note": NOTE,
        "standard_labels": STANDARD_LABELS,
        "csv_columns": HEADER,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def safe_float(value, default=0.0):
    """把传感器数据安全转换成 float，避免空值或异常数据导致程序崩溃。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@app.route("/", methods=["GET"])
def index():
    """用于浏览器检查服务器是否正常运行。"""
    return jsonify({
        "status": "running",
        "current_label": CURRENT_LABEL,
        "session_id": SESSION_ID,
        "participant": PARTICIPANT,
        "mount_position": MOUNT_POSITION,
        "standard_labels": STANDARD_LABELS,
        "csv_file": CSV_FILE,
        "message": "Flask server is ready to receive ESP32 BMI270 data."
    })


@app.route("/sensor", methods=["POST"])
def sensor():
    """接收 ESP32 发送来的 samples 数据，并保存到 CSV。"""
    data = request.get_json(force=True)
    samples = data.get("samples", [])

    if not isinstance(samples, list):
        return jsonify({
            "status": "error",
            "message": "samples must be a list"
        }), 400

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for s in samples:
            ax = safe_float(s.get("ax"))
            ay = safe_float(s.get("ay"))
            az = safe_float(s.get("az"))
            gx = safe_float(s.get("gx"))
            gy = safe_float(s.get("gy"))
            gz = safe_float(s.get("gz"))

            # 合加速度与合角速度，后面做机器学习特征时很有用
            acc_mag = math.sqrt(ax * ax + ay * ay + az * az)
            gyro_mag = math.sqrt(gx * gx + gy * gy + gz * gz)

            writer.writerow([
                datetime.now().isoformat(timespec="milliseconds"),
                s.get("t"),
                ax, ay, az,
                gx, gy, gz,
                acc_mag,
                gyro_mag,
                CURRENT_LABEL,
                SESSION_ID,
                PARTICIPANT,
                MOUNT_POSITION,
                NOTE,
            ])

    return jsonify({
        "status": "ok",
        "label": CURRENT_LABEL,
        "csv_file": CSV_FILE,
        "received": len(samples)
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Collect labeled BMI270 data from ESP32.")
    parser.add_argument(
        "--label",
        default=os.getenv("LUNACANE_LABEL", CURRENT_LABEL),
        help=f"Action label, recommended: {', '.join(STANDARD_LABELS)}.",
    )
    parser.add_argument("--participant", default=os.getenv("LUNACANE_PARTICIPANT", "unknown"), help="Collector/person id.")
    parser.add_argument("--mount-position", default=os.getenv("LUNACANE_MOUNT_POSITION", "cane_body"), help="IMU mounting position.")
    parser.add_argument("--note", default=os.getenv("LUNACANE_NOTE", ""), help="Short note for this session.")
    parser.add_argument("--output-dir", default=_RAW_DIR, help="Directory for CSV and metadata output.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_session(args.label, args.participant, args.mount_position, args.note, args.output_dir)
    init_csv()
    print(f"采集标签: {CURRENT_LABEL}")
    if CURRENT_LABEL not in STANDARD_LABELS:
        print(f"警告: {CURRENT_LABEL} 不是推荐标准标签，训练时会被归为非跌倒普通动作。")
    print(f"采集文件: {CSV_FILE}")
    print(f"元数据文件: {META_FILE}")

    # host="0.0.0.0" 表示允许局域网内 ESP32 访问电脑服务器
    # ESP32 代码里的 serverUrl 仍然要写电脑当前的局域网 IP，例如：
    # http://192.168.1.100:8000/sensor
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
