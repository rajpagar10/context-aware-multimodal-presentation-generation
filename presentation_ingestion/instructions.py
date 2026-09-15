from __future__ import annotations

import json
import re
from typing import Protocol

from .llm import TextGenerationBackend
from .models import ContentItem, ControlAttributes, ControlTarget, SemanticControlPlan, SemanticDirective
from .utils import normalize_whitespace, stable_id


class InstructionParser(Protocol):
    name: str

    def parse(self, instruction: str, items: list[ContentItem]) -> SemanticControlPlan: ...


def _resolve_target(value: str, items: list[ContentItem]) -> ControlTarget:
    clean = normalize_whitespace(value.strip(" .,!;:"))
    query = clean.casefold()
    for item in items:
        searchable = " ".join(filter(None, [item.structure.title or "", item.text])).casefold()
        if query and query in searchable:
            selector_type = "section" if item.structure.title and query == item.structure.title.casefold() else "exact_text"
            return ControlTarget(content_item_id=item.id, selector_type=selector_type, selector_value=clean)
    tokens = [token for token in re.findall(r"[a-z0-9]+", query) if len(token) > 2 and token not in {"the", "value", "section"}]
    for item in items:
        searchable = " ".join(filter(None, [item.structure.title or "", item.text])).casefold()
        if tokens and all(token in searchable for token in tokens):
            return ControlTarget(content_item_id=item.id, selector_type="keyword", selector_value=clean)
    return ControlTarget(selector_type="unresolved", selector_value=clean or "unspecified")


class RuleBasedInstructionParser:
    """Conservative semantic parser; it never produces timing fields."""

    name = "rule_based_v1"

    def parse(self, instruction: str, items: list[ContentItem]) -> SemanticControlPlan:
        clauses = [normalize_whitespace(part) for part in re.split(r"(?:,|;|\band\b)", instruction, flags=re.IGNORECASE) if normalize_whitespace(part)]
        directives: list[SemanticDirective] = []
        for clause in clauses:
            attrs = self._attributes(clause)
            target = self._target(clause)
            if attrs is None or target is None:
                continue
            resolved = _resolve_target(target, items)
            directives.append(SemanticDirective(
                id=stable_id("directive", clause, len(directives) + 1), target=resolved, attributes=attrs,
                confidence=0.9 if resolved.content_item_id else 0.55,
            ))
        warnings = []
        if not directives:
            warnings.append("No supported semantic control phrase was recognized. Try phrases such as 'calm during introduction' or 'emphasize accuracy'.")
        if any(directive.target.content_item_id is None for directive in directives):
            warnings.append("One or more control targets could not be matched to extracted content and remain semantic anchors.")
        return SemanticControlPlan(status="semantic_pending_alignment" if directives else "no_directives", directives=directives, parser=self.name, warnings=warnings)

    @staticmethod
    def _attributes(clause: str) -> ControlAttributes | None:
        lower = clause.casefold()
        attrs = ControlAttributes()
        found = False
        emotion_map = {"calm": "calm", "calmly": "calm", "excited": "excited", "exciting": "excited", "serious": "serious", "warm": "warm"}
        for word, emotion in emotion_map.items():
            if re.search(rf"\b{word}\b", lower):
                attrs.emotion = emotion
                found = True
                break
        if re.search(r"\b(energetic|energy|energetically)\b", lower):
            attrs.energy = 0.8
            found = True
        if re.search(r"\b(emphasize|emphasis|stress)\b", lower):
            attrs.emphasis = 0.9
            found = True
        if re.search(r"\b(slow|slower)\b", lower):
            attrs.rate = 0.3
            found = True
        if re.search(r"\b(fast|faster|quickly)\b", lower):
            attrs.rate = 0.8
            found = True
        if re.search(r"\b(high[- ]?pitch|raise pitch)\b", lower):
            attrs.pitch = 0.8
            found = True
        if "pause before" in lower:
            attrs.pause_before = True
            found = True
        if "pause after" in lower:
            attrs.pause_after = True
            found = True
        return attrs if found else None

    @staticmethod
    def _target(clause: str) -> str | None:
        patterns = [
            r"\b(?:during|for|before|after)\s+(?:the\s+)?(.+?)(?:\s*$)",
            r"\b(?:emphasize|emphasis on|stress)\s+(?:the\s+)?(.+?)(?:\s+value)?(?:\s*$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, clause, flags=re.IGNORECASE)
            if match:
                target = normalize_whitespace(match.group(1)).strip(" .,!;:")
                return re.sub(r"\bvalue$", "", target, flags=re.IGNORECASE).strip() or target
        return None


class ApiInstructionParser:
    name = "api"

    def __init__(self, backend: TextGenerationBackend):
        self.backend = backend

    def parse(self, instruction: str, items: list[ContentItem]) -> SemanticControlPlan:
        inventory = [{"id": item.id, "title": item.structure.title, "text": item.text} for item in items]
        answer = self.backend.generate(
            system_prompt=("Convert the instruction to JSON with a 'directives' array. Every directive requires "
                           "target selector_value and attributes only from emotion, energy, pitch, rate, emphasis, pause_before, pause_after. "
                           "Never create timestamps. Use content_item_id only from the supplied inventory."),
            user_prompt=json.dumps({"instruction": instruction, "content_items": inventory}),
        )
        try:
            payload = json.loads(answer)
            directives = []
            for index, raw in enumerate(payload.get("directives", []), start=1):
                raw_target = raw.get("target", {})
                value = str(raw_target.get("selector_value") or raw_target.get("target") or "unspecified")
                target = _resolve_target(value, items)
                proposed_id = raw_target.get("content_item_id")
                if proposed_id in {item.id for item in items}:
                    target.content_item_id = proposed_id
                directives.append(SemanticDirective(
                    id=stable_id("directive", instruction, index), target=target,
                    attributes=ControlAttributes.model_validate(raw.get("attributes", {})), confidence=0.75,
                ))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"LLM instruction response did not match the semantic-plan contract: {exc}") from exc
        return SemanticControlPlan(status="semantic_pending_alignment" if directives else "no_directives", directives=directives, parser=self.name)
