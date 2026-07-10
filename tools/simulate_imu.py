"""Replay a BMI270 CSV to the LunaCane collector as a virtual ESP32."""

import argparse
import csv
import time

import requests


SENSOR_FIELDS = ("ax", "ay", "az", "gx", "gy", "gz")


def load_samples(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in SENSOR_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV missing fields: {', '.join(missing)}")
        return [
            {"t": index * 20, **{field: float(row[field]) for field in SENSOR_FIELDS}}
            for index, row in enumerate(reader)
        ]


def replay(samples, url, batch_size=20, rate_hz=50.0, realtime=True, timeout=10):
    sent = 0
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        response = requests.post(url, json={"samples": batch}, timeout=timeout)
        response.raise_for_status()
        sent += len(batch)
        print(f"sent={sent}/{len(samples)} response={response.json()}")
        if realtime and start + batch_size < len(samples):
            time.sleep(len(batch) / rate_hz)
    return sent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="BMI270 CSV containing ax..gz columns")
    parser.add_argument("--url", default="http://127.0.0.1:8000/sensor")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--no-realtime", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replay(
        load_samples(args.csv_path),
        args.url,
        batch_size=args.batch_size,
        rate_hz=args.rate_hz,
        realtime=not args.no_realtime,
    )
