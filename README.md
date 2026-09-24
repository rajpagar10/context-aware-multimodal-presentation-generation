# Context-Aware Multimodal Presentation Generation

This repository contains **Review 1 — Task 1** of the B.Tech CS(AI) project
*A Context-Aware Multimodal Presentation Generation Framework Using Few-Shot
Voice Cloning and Adaptive Speech Synthesis*.

Task 1 is deliberately a foundation, not a TTS or voice-cloning system. It
accepts presentation, document, media, and script inputs and turns them into a
single JSON representation for later research modules.

## Scope

Implemented:

- input detection and validation for PPTX, PDF, video, audio, TXT, and DOCX;
- structured PPTX/PDF/text extraction;
- audio/video metadata extraction, video audio extraction, and pluggable ASR;
- a unified Pydantic/JSON schema;
- deterministic narration generation and an optional OpenAI-compatible LLM
  backend;
- semantic (not time-aligned) instruction parsing;
- a CLI, logging, and automated tests.

Explicitly deferred to Reviews 2 and 3: speaker embeddings, voice cloning,
MFA/phoneme alignment, F0/energy/emotion extraction, prosody control,
controllable TTS, subtitles, and video rendering.

## Architecture

```text
PPTX / PDF / Video / Audio / TXT / DOCX
                 |
       validation + modality extractor
                 |
       NormalizedRepresentation (versioned JSON)
              /                         \
  NarrationGenerator              InstructionParser
              \                         /
                 narration + semantic control plan
```

Source content is immutable within a pipeline result. Generated narration and
the original user instruction are separate fields. Source timestamps only come
from an input media container or ASR; semantic directives intentionally have no
invented audio timing.

## Installation

Use Python 3.11+ and install the base dependencies:

```powershell
python -m pip install -r requirements.txt
```

For audio/video ingestion, install FFmpeg and ensure `ffmpeg` and `ffprobe`
are on `PATH`. For local transcription, install the optional ASR dependencies:

```powershell
python -m pip install -r requirements-asr.txt
```

`faster-whisper` downloads the selected Whisper model on its first use. A CPU
machine can use `--asr-model base` (or `tiny`) and `--asr-compute-type int8`.
GPU use is optional.

## Configuration

No secret is stored in the repository. To use an OpenAI-compatible chat API
for narration or instruction planning, configure these environment variables:

```powershell
$env:LLM_API_KEY = "..."
$env:LLM_BASE_URL = "https://your-provider.example/v1"
$env:LLM_MODEL = "your-model"
```

The default `template` narration mode is fully local and deterministic. The
default instruction parser is rule-based and produces a safe semantic plan.

## Run

```powershell
# Preserve a supplied script and add a semantic plan
python main.py --input samples/script.txt --instruction "Speak calmly during the introduction" --output-dir outputs/text

# PPTX/PDF with deterministic narration
python main.py --input samples/deck.pptx --generate-narration --instruction "Emphasize the accuracy value" --output-dir outputs/pptx
python main.py --input samples/report.pdf --generate-narration --output-dir outputs/pdf

# Audio/video: requires FFmpeg and faster-whisper
python main.py --input samples/talk.wav --asr-model base --output-dir outputs/audio
python main.py --input samples/talk.mp4 --asr-model base --output-dir outputs/video

# Use an API narration backend (requires the variables above)
python main.py --input samples/deck.pptx --generate-narration --narration-mode api --output-dir outputs/api
```

Each run writes `<output-dir>/normalized.json`. Video-derived WAV files are
placed in `<output-dir>/derived/` and referenced under `artifacts`.

## Browser demo

Start the local demo server with:

```powershell
python demo.py
```

Then open `http://127.0.0.1:8765` in a browser. Choose a supported source file,
select narration or Task 2 speech analysis, and run the pipeline. The page shows
a summary of the extracted content and speech evidence, plus the normalized
JSON with a download button. Demo inputs and results are saved under
`outputs/browser_demo/`. The server listens on localhost only and uses the same
pipeline CLI and optional dependencies described above.

## Output outline

```json
{
  "schema_version": "1.0",
  "input": {"kind": "pptx", "metadata": {}},
  "content_items": [{"kind": "slide", "index": 1, "text": "..."}],
  "narration": [{"origin": "generated", "content_item_ids": ["..."]}],
  "instructions": {"raw_text": "Emphasize accuracy"},
  "semantic_control_plan": {
    "status": "semantic_pending_alignment",
    "directives": [{"target": {"selector_value": "accuracy"}, "attributes": {"emphasis": 0.9}}]
  }
}
```

## Limitations

- PDF OCR is intentionally not enabled in this review. Image-only pages are
  retained as empty pages with an explicit warning.
- Slide images, charts, and video imagery are not interpreted.
- Media transcription requires the optional local model/runtime; the pipeline
  gives an actionable error if it is unavailable.
- ASR word/segment timestamps are source evidence, not forced alignment.
- The rule parser recognizes common control phrasing. An API parser is optional
  and always schema-validated; neither parser assigns future audio timestamps.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The test suite generates PPTX, PDF, TXT, WAV, and mocked-FFmpeg video fixtures
at runtime. It verifies actual extractor fields and normalized JSON, rather
than merely checking that calls do not raise errors.
