#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/audio_cut/lyrics/models.py
# AI-SUMMARY: Dataclass models for ASR words, sentences, singing regions and complete lyrics timelines.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Type, TypeVar

from audio_cut.exceptions import TimelineValidationError

_EPS = 1e-9
_DURATION_ROUNDING_TOLERANCE_S = 0.001
_T = TypeVar("_T", bound="_TimedItem")


def _optional_float(value: Any, *, field_name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TimelineValidationError(f"{field_name} must be a number or null") from exc


def _required_float(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TimelineValidationError(f"{field_name} must be a number") from exc


def _validate_confidence(confidence: Optional[float]) -> None:
    if confidence is None:
        return
    if confidence < 0.0 or confidence > 1.0:
        raise TimelineValidationError("confidence must be in [0, 1]")


def _validate_interval(start_s: float, end_s: float, duration_s: Optional[float]) -> None:
    if start_s < 0.0:
        raise TimelineValidationError("start_s must be >= 0")
    if end_s <= start_s + _EPS:
        raise TimelineValidationError("end_s must be greater than start_s")
    if duration_s is not None and end_s > duration_s + _EPS:
        raise TimelineValidationError("end_s exceeds timeline duration")


@dataclass
class _TimedItem:
    start_s: float
    end_s: float
    confidence: Optional[float] = None

    def validate(self, duration_s: Optional[float] = None) -> None:
        _validate_interval(self.start_s, self.end_s, duration_s)
        _validate_confidence(self.confidence)


@dataclass
class Word(_TimedItem):
    """Single ASR word with global timeline seconds."""

    text: str = ""

    def __init__(
        self,
        text: str,
        start_s: float,
        end_s: float,
        confidence: Optional[float] = None,
    ) -> None:
        self.text = str(text)
        self.start_s = float(start_s)
        self.end_s = float(end_s)
        self.confidence = _optional_float(confidence, field_name="confidence")
        self.validate()
        if not self.text:
            raise TimelineValidationError("word text must not be empty")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Word":
        return cls(
            text=str(data.get("text", "")),
            start_s=_required_float(data.get("start_s"), field_name="word.start_s"),
            end_s=_required_float(data.get("end_s"), field_name="word.end_s"),
            confidence=_optional_float(data.get("confidence"), field_name="word.confidence"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "confidence": self.confidence,
        }


@dataclass
class Sentence(_TimedItem):
    """Sentence-level ASR phrase span in global timeline seconds."""

    text: str = ""

    def __init__(
        self,
        text: str,
        start_s: float,
        end_s: float,
        confidence: Optional[float] = None,
    ) -> None:
        self.text = str(text)
        self.start_s = float(start_s)
        self.end_s = float(end_s)
        self.confidence = _optional_float(confidence, field_name="confidence")
        self.validate()
        if not self.text:
            raise TimelineValidationError("sentence text must not be empty")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sentence":
        return cls(
            text=str(data.get("text", "")),
            start_s=_required_float(data.get("start_s"), field_name="sentence.start_s"),
            end_s=_required_float(data.get("end_s"), field_name="sentence.end_s"),
            confidence=_optional_float(data.get("confidence"), field_name="sentence.confidence"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "confidence": self.confidence,
        }


def _clamp_minor_duration_overshoot(item: _T, duration_s: Optional[float]) -> bool:
    """Clamp millisecond-rounded ASR spans that barely exceed the audio tail."""

    if duration_s is None or item.end_s <= duration_s + _EPS:
        return False
    if item.end_s <= duration_s + _DURATION_ROUNDING_TOLERANCE_S and item.start_s < duration_s:
        item.end_s = float(duration_s)
        return True
    return False


@dataclass
class VadRegion(_TimedItem):
    """mVAD or singing region emitted by an ASR provider."""

    kind: str = "singing"

    def __init__(
        self,
        start_s: float,
        end_s: float,
        confidence: Optional[float] = None,
        kind: str = "singing",
    ) -> None:
        self.start_s = float(start_s)
        self.end_s = float(end_s)
        self.confidence = _optional_float(confidence, field_name="confidence")
        self.kind = str(kind or "singing")
        self.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VadRegion":
        return cls(
            start_s=_required_float(data.get("start_s"), field_name="vad_region.start_s"),
            end_s=_required_float(data.get("end_s"), field_name="vad_region.end_s"),
            confidence=_optional_float(data.get("confidence"), field_name="vad_region.confidence"),
            kind=str(data.get("kind", "singing")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "confidence": self.confidence,
            "kind": self.kind,
        }


@dataclass
class LyricsTimeline:
    """Full-track ASR timeline used as a soft prior for boundary planning."""

    words: List[Word] = field(default_factory=list)
    sentences: List[Sentence] = field(default_factory=list)
    vad_regions: List[VadRegion] = field(default_factory=list)
    duration_s: Optional[float] = None
    source: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.duration_s = _optional_float(self.duration_s, field_name="duration_s")
        if self.duration_s is not None and self.duration_s <= 0.0:
            raise TimelineValidationError("duration_s must be positive")
        self.words = sorted(list(self.words), key=lambda item: (item.start_s, item.end_s))
        self.sentences = sorted(list(self.sentences), key=lambda item: (item.start_s, item.end_s))
        self.vad_regions = sorted(list(self.vad_regions), key=lambda item: (item.start_s, item.end_s))
        self.validate(strict=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, strict: bool = False) -> "LyricsTimeline":
        duration_s = _optional_float(data.get("duration_s"), field_name="duration_s")
        warnings: List[str] = list(data.get("warnings", []))
        words = _load_items(Word, data.get("words", []), duration_s, strict, warnings)
        sentences = _load_items(Sentence, data.get("sentences", []), duration_s, strict, warnings)
        vad_regions = _load_items(VadRegion, data.get("vad_regions", []), duration_s, strict, warnings)
        return cls(
            words=words,
            sentences=sentences,
            vad_regions=vad_regions,
            duration_s=duration_s,
            source=str(data.get("source", "unknown")),
            warnings=warnings,
            meta=dict(data.get("meta", {})),
        )

    def validate(self, *, strict: bool = True) -> None:
        errors: List[str] = []
        for group_name, items in (
            ("words", self.words),
            ("sentences", self.sentences),
            ("vad_regions", self.vad_regions),
        ):
            for index, item in enumerate(items):
                try:
                    item.validate(self.duration_s)
                except TimelineValidationError as exc:
                    errors.append(f"{group_name}[{index}]: {exc}")
        if errors and strict:
            raise TimelineValidationError("; ".join(errors))
        self.warnings.extend(errors)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "source": self.source,
            "warnings": list(self.warnings),
            "meta": dict(self.meta),
            "words": [word.to_dict() for word in self.words],
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "vad_regions": [region.to_dict() for region in self.vad_regions],
        }


def _load_items(
    item_type: Type[_T],
    raw_items: Iterable[Dict[str, Any]],
    duration_s: Optional[float],
    strict: bool,
    warnings: List[str],
) -> List[_T]:
    items: List[_T] = []
    for index, raw in enumerate(raw_items):
        try:
            item = item_type.from_dict(raw)  # type: ignore[attr-defined]
            if _clamp_minor_duration_overshoot(item, duration_s):
                warnings.append(
                    f"{item_type.__name__}[{index}]: "
                    "end_s clamped to timeline duration after minor rounding overshoot"
                )
            item.validate(duration_s)
            items.append(item)
        except TimelineValidationError as exc:
            message = f"{item_type.__name__}[{index}]: {exc}"
            if strict:
                raise TimelineValidationError(message) from exc
            warnings.append(message)
    return items
