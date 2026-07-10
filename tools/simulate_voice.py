"""Upload a WAV file and download the PCM reply like the ESP32 client."""

import argparse
from pathlib import Path

import requests


def run(audio_path, base_url, output_path, timeout=200):
    audio = Path(audio_path).read_bytes()
    response = requests.post(
        f"{base_url.rstrip('/')}/upload_audio",
        data=audio,
        headers={"Content-Type": "audio/wav"},
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    filename = result.get("file")
    if result.get("status") != "success" or not filename or filename == "none":
        raise RuntimeError(f"voice pipeline returned no reply: {result}")

    reply = requests.get(
        f"{base_url.rstrip('/')}/get_audio/{filename}", timeout=timeout
    )
    reply.raise_for_status()
    Path(output_path).write_bytes(reply.content)
    print(f"saved {len(reply.content)} PCM bytes to {output_path}")
    return filename


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="reply.pcm")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.audio_path, args.base_url, args.output)
