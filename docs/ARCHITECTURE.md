# Review 1 Architecture Decisions

## Contract first

`NormalizedRepresentation` is the one downstream contract. Extractors may add
metadata, but never define a modality-specific top-level schema. It separates
source `content_items`, `narration`, raw `instructions`, and a
`semantic_control_plan`.

## Time is evidence

`source_timing` is populated only from ASR in this review. A semantic directive
has `timing_status: semantic_pending_alignment` and no start/end timestamps.
Review 2 can introduce alignment objects without mutating source timing.

## Interchangeable services

`Transcriber`, `NarrationGenerator`, `InstructionParser`, and
`TextGenerationBackend` are narrow interfaces. The local deterministic/template
implementations make tests reproducible. `FasterWhisperTranscriber` and the
OpenAI-compatible HTTP backend are optional runtime adapters.

## Media boundaries

FFmpeg is isolated in `MediaTools`; video extraction writes a derived mono,
16-kHz WAV artifact. No video frames, speaker embeddings, prosody features, or
rendered media are produced in Review 1.
