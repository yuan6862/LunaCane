from flask import Flask, request, jsonify
import csv
import os
import math
from datetime import datetime

app = Flask(__name__)

# ===============================
# 当前采集动作标签
# 每次采集一种动作前，手动修改这里
# 可选示例：stand / walk / sit_down / put_down / tap / fall
# ===============================
CURRENT_LABEL = "walk"

# CSV 统一存放到项目 data/raw/ 目录，供 ml/preprocess.py 直接读取
_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CSV_FILE = os.path.join(_RAW_DIR, f"bmi_data_{CURRENT_LABEL}.csv")

HEADER = [
    "pc_time",
    "esp_time_ms",
    "ax", "ay", "az",
    "gx", "gy", "gz",
    "acc_mag",
    "gyro_mag",
    "label"
]


def init_csv():
    """如果 CSV 文件不存在，就先创建并写入表头。"""
    os.makedirs(_RAW_DIR, exist_ok=True)
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


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
                CURRENT_LABEL
            ])

    return jsonify({
        "status": "ok",
        "label": CURRENT_LABEL,
        "csv_file": CSV_FILE,
        "received": len(samples)
    })


if __name__ == "__main__":
    init_csv()

    # host="0.0.0.0" 表示允许局域网内 ESP32 访问电脑服务器
    # ESP32 代码里的 serverUrl 仍然要写电脑当前的局域网 IP，例如：
    # http://192.168.1.100:8000/sensor
    app.run(host="0.0.0.0", port=8000, debug=True)
