#!/usr/bin/env python3
"""Prepare a reference clip and generate an authorized Qwen3-TTS voice clone."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_TEXT = (
    "This is a short test of my voice clone, generated locally with Qwen three TTS."
)


def parse_time(value: str) -> float:
    value = str(value).strip()
    if not value:
        raise argparse.ArgumentTypeError("time value cannot be empty")
    if ":" not in value:
        return float(value)
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise argparse.ArgumentTypeError(f"unsupported time format: {value}")


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:.3f}".rstrip("0").rstrip(".")


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def run(cmd: list[str], *, capture: bool = False, env: dict[str, str] | None = None) -> str:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture,
        env=env,
    )
    if capture:
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return result.stdout
    return ""


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required command not found on PATH: {name}")


def slug_from_source(source: str) -> str:
    if is_url(source):
        parsed = urlparse(source)
        stem = (parsed.path.rstrip("/").split("/")[-1] or parsed.netloc).split("?")[0]
    else:
        stem = Path(source).stem
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in stem)
    return safe.strip("-") or "voice-clone"


def make_output_dir(source: str, start: float, output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path.cwd() / f"{slug_from_source(source)}-{fmt_seconds(start)}s-{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_source(source: str, workdir: Path) -> Path:
    if is_url(source):
        require_tool("uvx")
        template = str(workdir / "source.%(ext)s")
        cmd = ["uvx", "yt-dlp"]
        if shutil.which("node"):
            cmd += ["--js-runtimes", "node"]
        cmd += [
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "-o",
            template,
            "--print",
            "after_move:filepath",
            source,
        ]
        output = run(cmd, capture=True)
        for line in reversed([line.strip() for line in output.splitlines()]):
            candidate = Path(line)
            if candidate.exists():
                return candidate
        candidates = sorted(workdir.glob("source.*"))
        if candidates:
            return candidates[0]
        raise RuntimeError("yt-dlp completed but no downloaded source file was found")

    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def trim_reference(source_path: Path, start: float, duration: float, workdir: Path) -> tuple[Path, Path]:
    require_tool("ffmpeg")
    reference_wav = workdir / "reference.wav"
    reference_mp3 = workdir / "reference.mp3"
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            fmt_seconds(start),
            "-i",
            str(source_path),
            "-t",
            fmt_seconds(duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            str(reference_wav),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(reference_wav),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(reference_mp3),
        ]
    )
    return reference_wav, reference_mp3


def transcribe(reference_wav: Path, workdir: Path) -> tuple[Path, str]:
    require_tool("uvx")
    transcript = workdir / "reference.txt"
    run(
        [
            "uvx",
            "parakeet-mlx",
            "--output-dir",
            str(workdir),
            "--output-format",
            "txt",
            "--output-template",
            "reference",
            str(reference_wav),
        ]
    )
    if not transcript.exists():
        raise RuntimeError(f"Parakeet did not create expected transcript: {transcript}")
    text = transcript.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("Parakeet transcript is empty")
    return transcript, text


def read_target_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).expanduser().read_text(encoding="utf-8").strip()
    if args.text:
        return args.text.strip()
    return DEFAULT_TEXT


def generate_clone(
    *,
    model_id: str,
    reference_wav: Path,
    reference_text: str,
    target_text: str,
    workdir: Path,
    max_new_tokens: int,
) -> tuple[Path, Path]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    if torch.cuda.is_available():
        kwargs = {"device_map": "cuda:0", "dtype": torch.bfloat16}
    elif torch.backends.mps.is_available():
        kwargs = {"device_map": "mps", "dtype": torch.float16}
    else:
        kwargs = {"device_map": "cpu", "dtype": torch.float32}

    model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
    wavs, sample_rate = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=str(reference_wav),
        ref_text=reference_text,
        max_new_tokens=max_new_tokens,
    )

    clone_wav = workdir / "clone.wav"
    clone_mp3 = workdir / "clone.mp3"
    sf.write(clone_wav, wavs[0], sample_rate)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(clone_wav),
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(clone_mp3),
        ]
    )
    return clone_wav, clone_mp3


def ffprobe_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        output = run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture=True,
        )
        return float(output.strip())
    except Exception:
        return None


def write_manifest(workdir: Path, data: dict) -> Path:
    manifest = workdir / "manifest.json"
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Qwen3-TTS voice clone sample from an authorized source voice."
    )
    parser.add_argument("source", help="Local audio/video path, direct media URL, or YouTube URL")
    parser.add_argument("--start", required=True, type=parse_time, help="Start time, e.g. 27 or 00:00:27")
    parser.add_argument("--duration", type=parse_time, default=30.0, help="Reference duration, default: 30")
    parser.add_argument("--text", help="Target text to synthesize")
    parser.add_argument("--text-file", help="Path to a UTF-8 text file with target text")
    parser.add_argument("--output-dir", help="Directory for generated files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Qwen3-TTS model id, default: {DEFAULT_MODEL}")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.duration < 20 or args.duration > 35:
        print(
            f"Warning: reference duration is {fmt_seconds(args.duration)}s; "
            "Qwen3-TTS cloning usually works best around 20-30s.",
            file=sys.stderr,
        )

    workdir = make_output_dir(args.source, args.start, args.output_dir)
    source_path = resolve_source(args.source, workdir)
    reference_wav, reference_mp3 = trim_reference(source_path, args.start, args.duration, workdir)
    transcript_path, reference_text = transcribe(reference_wav, workdir)
    target_text = read_target_text(args)
    clone_wav, clone_mp3 = generate_clone(
        model_id=args.model,
        reference_wav=reference_wav,
        reference_text=reference_text,
        target_text=target_text,
        workdir=workdir,
        max_new_tokens=args.max_new_tokens,
    )

    manifest = write_manifest(
        workdir,
        {
            "source": args.source,
            "source_path": str(source_path),
            "start_seconds": args.start,
            "duration_seconds": args.duration,
            "model": args.model,
            "target_text": target_text,
            "reference_text": reference_text,
            "reference_wav": str(reference_wav),
            "reference_mp3": str(reference_mp3),
            "transcript": str(transcript_path),
            "clone_wav": str(clone_wav),
            "clone_mp3": str(clone_mp3),
            "clone_duration_seconds": ffprobe_duration(clone_wav),
        },
    )

    print("\nDone.")
    print(f"Output directory: {workdir}")
    print(f"Reference WAV: {reference_wav}")
    print(f"Reference transcript: {transcript_path}")
    print(f"Clone WAV: {clone_wav}")
    print(f"Clone MP3: {clone_mp3}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
