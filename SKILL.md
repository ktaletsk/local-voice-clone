---
name: qwen3-voice-clone
description: >
  Create an authorized Qwen3-TTS voice clone sample from a local audio/video
  file, direct media URL, or YouTube URL. Use when the user provides a source
  recording, a start time, and a target phrase, and wants a roughly 30-second
  reference clip downloaded, trimmed, transcribed with Parakeet, normalized for
  cloning, and used with Qwen3-TTS Base to generate cloned speech. Only use for
  the user's own voice or a voice they explicitly have permission to clone.
metadata:
  short-description: Authorized Qwen3-TTS voice cloning workflow
---

# Qwen3 Voice Clone

Build a local, authorized voice-clone sample from a short reference clip.
The default workflow is optimized for a 30-second reference window and runs
locally with `uvx`, `ffmpeg`, `parakeet-mlx`, and `qwen-tts`.

## Consent Gate

Before cloning, confirm the source voice is the user's own voice or a voice the
user has explicit permission to clone. If consent is unclear, ask before
running the clone. Do not clone public figures, conference speakers, coworkers,
or other third parties without explicit authorization.

## Output Contract

A successful run creates an output directory containing:

- `source.*` - downloaded/staged source media when the input is a URL
- `reference.wav` - 24 kHz mono normalized WAV reference clip
- `reference.mp3` - playback copy of the reference clip
- `reference.txt` - Parakeet transcript of the reference clip
- `clone.wav` - Qwen3-TTS cloned speech output
- `clone.mp3` - playback copy of the cloned speech
- `manifest.json` - source, timing, model, transcript, and output paths

Report the absolute paths to `clone.wav`, `clone.mp3`, `reference.wav`, and
`reference.txt`.

## Workflow

1. Resolve the source:
   - Local path: use it directly.
   - YouTube or media URL: download with `uvx yt-dlp`.
2. Trim a 30-second reference clip:
   - Use the user-provided start time.
   - If the user provides an end time, compute duration and keep it close to 30s.
   - If no duration is specified, use `30`.
3. Normalize and convert to cloning format:
   - 24 kHz, mono, PCM WAV.
4. Transcribe with Parakeet:
   - Use `uvx parakeet-mlx`.
   - On macOS sandboxed sessions, Parakeet may need approval to run outside the
     sandbox so MLX can access Metal.
5. Generate speech with Qwen3-TTS:
   - Use `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
   - Provide `ref_audio=reference.wav` and `ref_text=reference.txt`.
   - On macOS sandboxed sessions, Qwen/PyTorch may need approval to run outside
     the sandbox so MPS can access Metal.
6. Verify:
   - Run `ffprobe` on `clone.wav` or `clone.mp3`.
   - Optionally transcribe `clone.wav` with Parakeet to confirm the target text.

## Script

Use the bundled script for the end-to-end workflow:

```bash
uvx --from qwen-tts --with soundfile python scripts/voice_clone.py \
  "https://www.youtube.com/watch?v=VIDEO_ID" \
  --start 27 \
  --duration 30 \
  --text "This is a short test of my voice clone." \
  --output-dir ./voice-clone-run
```

For a local source:

```bash
uvx --from qwen-tts --with soundfile python scripts/voice_clone.py \
  /path/to/source.mp4 \
  --start 00:00:27 \
  --text-file target.txt
```

If the target text is missing, use a short neutral test sentence and tell the
user exactly what was generated.

## Notes

- Prefer a clean, single-speaker 20-30 second reference segment.
- Avoid music, heavy applause, overlapping speakers, or long silence.
- Keep the exact transcript from Parakeet unless it is obviously wrong; Qwen3-TTS
  voice cloning quality depends on `ref_text` matching `ref_audio`.
- Do not create a manual `.venv`; use `uvx` so dependencies stay in uv's cache.
