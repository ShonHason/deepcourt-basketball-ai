from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


BALL_CLASS = 0
RIM_CLASS = 1


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass
class ClassMetrics:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    confidences: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    @property
    def average_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "avg_confidence": round(self.average_confidence, 4),
        }


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read_yolo_labels(label_path: Path, image_w: int, image_h: int) -> list[Box]:
    if not label_path.exists():
        return []

    boxes: list[Box] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = (cx - bw / 2) * image_w
        y1 = (cy - bh / 2) * image_h
        x2 = (cx + bw / 2) * image_w
        y2 = (cy + bh / 2) * image_h
        boxes.append(Box(class_id, x1, y1, x2, y2))
    return boxes


def evaluate_class(
    predictions: Iterable[Box],
    ground_truth: Iterable[Box],
    class_id: int,
    iou_threshold: float,
) -> ClassMetrics:
    preds = [b for b in predictions if b.class_id == class_id]
    labels = [b for b in ground_truth if b.class_id == class_id]
    matched_labels: set[int] = set()
    metrics = ClassMetrics(confidences=[p.confidence for p in preds])

    for pred in sorted(preds, key=lambda b: b.confidence, reverse=True):
        best_idx: Optional[int] = None
        best_iou = 0.0
        for idx, label in enumerate(labels):
            if idx in matched_labels:
                continue
            score = iou(pred, label)
            if score > best_iou:
                best_iou = score
                best_idx = idx

        if best_idx is not None and best_iou >= iou_threshold:
            matched_labels.add(best_idx)
            metrics.true_positive += 1
        else:
            metrics.false_positive += 1

    metrics.false_negative = len(labels) - len(matched_labels)
    return metrics


def merge_metrics(items: Iterable[ClassMetrics]) -> ClassMetrics:
    merged = ClassMetrics()
    for item in items:
        merged.true_positive += item.true_positive
        merged.false_positive += item.false_positive
        merged.false_negative += item.false_negative
        merged.confidences.extend(item.confidences)
    return merged
