from __future__ import annotations

from typing import Protocol

from .llm import TextGenerationBackend
from .models import ContentItem, NarrationItem
from .utils import normalize_whitespace, stable_id


class NarrationGenerator(Protocol):
    name: str

    def generate(self, item: ContentItem) -> str: ...


class TemplateNarrationGenerator:
    """Deterministic baseline. It is intentionally not represented as an LLM."""

    name = "template_v1"

    def generate(self, item: ContentItem) -> str:
        title = item.structure.title
        points = item.structure.bullets or item.structure.body
        if title and points:
            return normalize_whitespace(f"This section, {title}, covers the following key points: {'; '.join(points)}.")
        if title:
            return normalize_whitespace(f"This section covers {title}.")
        return normalize_whitespace(item.text)


class ApiNarrationGenerator:
    name = "api"

    def __init__(self, backend: TextGenerationBackend):
        self.backend = backend

    def generate(self, item: ContentItem) -> str:
        payload = {
            "content_id": item.id,
            "kind": item.kind.value,
            "title": item.structure.title,
            "body": item.structure.body,
            "speaker_notes": item.structure.speaker_notes,
            "original_text": item.text,
        }
        return self.backend.generate(
            system_prompt=("Create a concise, accurate presentation narration for one source item. "
                           "Do not invent facts. Return only narration text, without labels or markdown."),
            user_prompt=str(payload),
        )


def generated_narration(items: list[ContentItem], generator: NarrationGenerator) -> list[NarrationItem]:
    narration: list[NarrationItem] = []
    for item in items:
        if not item.text.strip():
            continue
        text = generator.generate(item)
        if not text:
            continue
        narration.append(NarrationItem(
            id=stable_id("narration", item.id, generator.name), content_item_ids=[item.id], text=text,
            origin="generated", generation_status="generated", metadata={"generator": generator.name},
        ))
    return narration
