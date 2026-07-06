from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime

app = Flask(__name__)

# 调试用（无标签），CSV 存到 data/raw/ 与训练代码保持一致
_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(_RAW_DIR, exist_ok=True)
CSV_FILE = os.path.join(_RAW_DIR, "bmi_data_debug.csv")
MAX_SAMPLES = int(os.getenv("LUNACANE_DEBUG_MAX_SAMPLES", "3000"))
sample_count = 0

# 如果文件不存在，先写表头
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pc_time",
            "esp_time_ms",
            "ax", "ay", "az",
            "gx", "gy", "gz"
        ])

@app.route("/sensor", methods=["POST"])
def sensor():
    global sample_count
    data = request.get_json(force=True)
    samples = data.get("samples", [])

    if sample_count >= MAX_SAMPLES:
        return jsonify({"status": "full", "received": 0, "max_samples": MAX_SAMPLES})

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writable_samples = samples[:max(0, MAX_SAMPLES - sample_count)]
        for s in writable_samples:
            writer.writerow([
                datetime.now().isoformat(),
                s.get("t"),
                s.get("ax"),
                s.get("ay"),
                s.get("az"),
                s.get("gx"),
                s.get("gy"),
                s.get("gz"),
            ])
        sample_count += len(writable_samples)

    return jsonify({"status": "ok", "received": len(writable_samples), "total": sample_count})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
