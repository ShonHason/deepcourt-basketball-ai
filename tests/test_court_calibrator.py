import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vision.court_calibrator import CourtCalibrator
from src.vision.detector import BallDetection, FrameDetections, HoopDetection


def _fd(frame_idx, hoops=None, ball=None):
    return FrameDetections(
        frame_idx=frame_idx,
        timestamp_ms=frame_idx * 200,
        frame_width=1000,
        frame_height=600,
        hoops=hoops or [],
        ball=ball,
    )


def test_calibrator_uses_real_hoop_detections():
    cal = CourtCalibrator()

    for idx in range(5):
        cal.update(_fd(
            idx,
            hoops=[
                HoopDetection((95, 180, 125, 210), 0.82, "left"),
                HoopDetection((875, 180, 905, 210), 0.80, "right"),
            ],
        ))

    left, right = cal.result()
    assert cal.ready is True
    assert 0.10 < left < 0.13
    assert 0.88 < right < 0.91


def test_calibrator_ignores_ball_positions_and_synthetic_hoops():
    cal = CourtCalibrator()

    for idx in range(20):
        cal.update(_fd(
            idx,
            hoops=[
                HoopDetection((80, 240, 110, 270), 0.20, "left"),
                HoopDetection((890, 240, 920, 270), 0.20, "right"),
            ],
            ball=BallDetection((500, 300, 520, 320), 0.95),
        ))

    assert cal.result() == (0.10, 0.90)
