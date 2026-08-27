"""LinkSeg 音频结构分析的独立进程入口。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

LINKSEG_ROOT = Path(__file__).resolve().parent / "vendor" / "linkseg"
LINKSEG_SRC = LINKSEG_ROOT / "src"
sys.path.insert(0, str(LINKSEG_SRC))

import imageio_ffmpeg
import librosa
import numpy as np
import torch
from data_utils import FileStruct, downsample_frames, read_beats
from post_processing import post_process
from predict import load_model
from preprocess_data import process_track

LABEL_ZH = {
    "silence": "静音",
    "intro": "前奏",
    "verse": "主歌",
    "chorus": "副歌",
    "bridge": "桥段",
    "inst": "间奏",
    "outro": "尾奏",
    "pre-chorus": "预副歌",
    "post-chorus": "后副歌",
}


def _model_args(model_path: Path, labels: int) -> Namespace:
    return Namespace(
        n_mels=64,
        n_fft=1024,
        hop_length=256,
        f_min=0,
        f_max=11025,
        sample_rate=22050,
        n_embedding=64,
        max_len=1500,
        conv_ndim=32,
        attention_ndim=32,
        attention_nheads=8,
        attention_nlayers=2,
        hidden_dim=32,
        dropout=0.1,
        nb_ssm_classes=3,
        nb_section_labels=labels,
        hidden_size=32,
        output_channels=16,
        dropout_gnn=0.1,
        dropout_cnn=0.2,
        dropout_egat=0.5,
        max_past=8,
        max_future=8,
        tau=0,
        model_name=str(model_path),
        gpu=-1,
    )


def _merge_segments(times: list[float], labels: list[str]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    for index, label in enumerate(labels):
        start = round(float(times[index]), 3)
        end = round(float(times[index + 1]), 3)
        if segments and segments[-1]["label"] == label:
            segments[-1]["end"] = end
        else:
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                    "label_zh": LABEL_ZH.get(label, label),
                }
            )
    return segments


def _normalize_audio(audio_path: Path) -> Path:
    if audio_path.suffix.lower() in {".wav", ".wave"}:
        return audio_path

    normalized_path = audio_path.with_name(f"{audio_path.stem}.wav")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "22050",
            str(normalized_path),
        ],
        check=True,
        capture_output=True,
    )
    return normalized_path


def analyze(audio_path: Path, labels: int) -> list[dict[str, object]]:
    audio_path = _normalize_audio(audio_path)
    job_root = audio_path.parents[1]
    for directory in ("audio_npy", "features", "predictions"):
        (job_root / directory).mkdir(parents=True, exist_ok=True)

    process_track(str(audio_path))
    file_struct = FileStruct(str(audio_path))
    model_path = LINKSEG_ROOT / "data" / f"model_{labels}_classes.pt"
    args = _model_args(model_path, labels)
    model = load_model(args).to(torch.device("cpu"))
    model.eval()

    beat_frames, duration = read_beats(file_struct.beat_file)
    beat_frames = librosa.util.fix_frames(beat_frames)
    beat_frames = downsample_frames(beat_frames, max_length=args.max_len)
    beat_times = librosa.frames_to_time(beat_frames, sr=22050, hop_length=256)
    beat_frames = librosa.time_to_frames(beat_times, sr=22050, hop_length=1)
    pad_width = ((args.hop_length * args.n_embedding) - 2) // 2
    waveform = np.load(file_struct.audio_npy_file)
    padded = np.pad(waveform, pad_width=(pad_width, pad_width), mode="edge")
    features = np.stack(
        [padded[index : index + pad_width * 2] for index in beat_frames],
        axis=0,
    )

    with torch.inference_mode():
        _embeddings, bounds, classes, _links = model(torch.tensor(features))
    times, predicted_labels = post_process(
        str(audio_path),
        beat_times,
        duration,
        bounds.detach().cpu().numpy(),
        classes.detach().cpu().numpy(),
        args.max_past,
        args.max_future,
        args.tau,
    )
    return _merge_segments(
        [float(value) for value in times],
        [str(value.item() if hasattr(value, "item") else value) for value in predicted_labels],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--labels", type=int, choices=(7, 9), default=7)
    args = parser.parse_args()

    result = analyze(args.audio_path, args.labels)
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()