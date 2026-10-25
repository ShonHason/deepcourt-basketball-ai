import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vision.detection_eval import Box, evaluate_class, iou, merge_metrics


def test_iou_overlap():
    a = Box(0, 0, 0, 10, 10)
    b = Box(0, 5, 5, 15, 15)
    assert round(iou(a, b), 4) == 0.1429


def test_evaluate_class_counts_tp_fp_fn():
    predictions = [
        Box(0, 0, 0, 10, 10, 0.9),
        Box(0, 50, 50, 60, 60, 0.8),
        Box(1, 0, 0, 10, 10, 0.7),
    ]
    labels = [
        Box(0, 1, 1, 11, 11),
        Box(0, 100, 100, 110, 110),
    ]

    metrics = evaluate_class(predictions, labels, class_id=0, iou_threshold=0.5)

    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert round(metrics.precision, 4) == 0.5
    assert round(metrics.recall, 4) == 0.5


def test_merge_metrics():
    a = evaluate_class([Box(0, 0, 0, 10, 10, 0.9)], [Box(0, 0, 0, 10, 10)], 0, 0.5)
    b = evaluate_class([], [Box(0, 0, 0, 10, 10)], 0, 0.5)

    merged = merge_metrics([a, b])

    assert merged.true_positive == 1
    assert merged.false_negative == 1
    assert merged.false_positive == 0
