# local-voice-clone

[![skills.sh](https://skills.sh/b/ktaletsk/local-voice-clone)](https://skills.sh/ktaletsk/local-voice-clone)

An Agent Skill for creating authorized local voice-clone samples with Qwen3-TTS.

The skill takes a local audio/video file, direct media URL, or YouTube URL, trims a roughly 30-second reference segment, transcribes it with Parakeet, prepares a normalized reference WAV, and generates cloned speech with `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.

Install with the skills CLI:

```bash
npx skills add ktaletsk/local-voice-clone
```

Example script invocation:

```bash
uvx --from qwen-tts --with soundfile python scripts/voice_clone.py \
  "https://www.youtube.com/watch?v=VIDEO_ID" \
  --start 27 \
  --duration 30 \
  --text "This is a short test of my voice clone." \
  --output-dir ./voice-clone-run
```

Only use this workflow for your own voice or a voice you have explicit permission to clone.
