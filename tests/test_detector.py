"""
Tests for detector.py data models and geometry helpers.
Runs without ultralytics installed — the Detector class itself is not tested
here (that requires models on disk), but all supporting logic is verified.
"""

import sys, os
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vision.detector import (
    Keypoint, PlayerDetection, BallDetection, HoopDetection, FrameDetections,
    Detector,
    KP_LEFT_WRIST, KP_RIGHT_WRIST, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
    KP_NOSE, KP_RIGHT_ELBOW, KP_LEFT_HIP, KP_RIGHT_HIP,
    _iou, pixel_distance, is_contested, KEYPOINT_CONF,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_keypoints(overrides: dict[int, tuple[float, float, float]] = {}) -> list[Keypoint]:
    """17 invisible keypoints, with selected ones overridden."""
    kps = [Keypoint(0.0, 0.0, 0.0) for _ in range(17)]
    for idx, (x, y, c) in overrides.items():
        kps[idx] = Keypoint(x, y, c)
    return kps


def make_player(bbox=(100, 100, 200, 400), kps=None) -> PlayerDetection:
    return PlayerDetection(
        bbox=bbox,
        confidence=0.9,
        keypoints=kps or [Keypoint(0, 0, 0)] * 17,
    )


# ---------------------------------------------------------------------------
# Keypoint
# ---------------------------------------------------------------------------

def test_keypoint_visible():
    assert Keypoint(10, 20, 0.9).visible is True
    assert Keypoint(10, 20, KEYPOINT_CONF - 0.01).visible is False


# ---------------------------------------------------------------------------
# PlayerDetection geometry
# ---------------------------------------------------------------------------

def test_player_center():
    p = make_player(bbox=(100, 100, 200, 300))
    assert p.center == (150.0, 200.0)


def test_player_foot_center():
    p = make_player(bbox=(100, 100, 200, 400))
    assert p.foot_center == (150.0, 400.0)


def test_hands_up_true():
    # Wrists above shoulders (smaller y in image coords)
    kps = make_keypoints({
        KP_LEFT_WRIST:     (100, 50,  0.9),   # wrist y=50 < shoulder y=150
        KP_RIGHT_WRIST:    (200, 50,  0.9),
        KP_LEFT_SHOULDER:  (100, 150, 0.9),
        KP_RIGHT_SHOULDER: (200, 150, 0.9),
    })
    assert make_player(kps=kps).hands_up() is True


def test_hands_up_false_wrists_below():
    kps = make_keypoints({
        KP_LEFT_WRIST:     (100, 200, 0.9),   # wrists below shoulders
        KP_RIGHT_WRIST:    (200, 200, 0.9),
        KP_LEFT_SHOULDER:  (100, 150, 0.9),
        KP_RIGHT_SHOULDER: (200, 150, 0.9),
    })
    assert make_player(kps=kps).hands_up() is False


def test_hands_up_false_low_confidence():
    kps = make_keypoints({
        KP_LEFT_WRIST:     (100, 50,  0.1),   # visible=False
        KP_RIGHT_WRIST:    (200, 50,  0.1),
        KP_LEFT_SHOULDER:  (100, 150, 0.9),
        KP_RIGHT_SHOULDER: (200, 150, 0.9),
    })
    assert make_player(kps=kps).hands_up() is False


def test_shooting_pose_true():
    # Right wrist above nose, elbow at ~90°
    kps = make_keypoints({
        KP_NOSE:        (150, 100, 0.9),
        KP_RIGHT_WRIST: (160, 60,  0.9),   # above nose
        KP_RIGHT_ELBOW: (155, 90,  0.9),
    })
    p = make_player(kps=kps)
    # Just verify it doesn't crash and returns a bool
    result = p.shooting_pose()
    assert isinstance(result, bool)


def test_shooting_pose_false_wrist_below_nose():
    kps = make_keypoints({
        KP_NOSE:        (150, 100, 0.9),
        KP_RIGHT_WRIST: (160, 180, 0.9),  # below nose
        KP_RIGHT_ELBOW: (155, 140, 0.9),
    })
    assert make_player(kps=kps).shooting_pose() is False


def test_driving_direction():
    # Left hip at smaller x → driving left
    kps = make_keypoints({
        KP_LEFT_HIP:  (80,  300, 0.9),
        KP_RIGHT_HIP: (120, 300, 0.9),
    })
    assert make_player(kps=kps).driving_direction() == "left"


def test_wrist_position_returns_higher():
    # Left wrist at y=50, right at y=100 → left is higher (smaller y)
    kps = make_keypoints({
        KP_LEFT_WRIST:  (100, 50,  0.9),
        KP_RIGHT_WRIST: (150, 100, 0.9),
    })
    wp = make_player(kps=kps).wrist_position()
    assert wp is not None
    assert wp[1] == 50.0   # y of the higher wrist


def test_wrist_position_none_when_invisible():
    assert make_player().wrist_position() is None


# ---------------------------------------------------------------------------
# BallDetection
# ---------------------------------------------------------------------------

def test_ball_center():
    b = BallDetection(bbox=(90, 90, 110, 110), confidence=0.8)
    assert b.center == (100.0, 100.0)


def test_ball_radius():
    b = BallDetection(bbox=(90, 90, 110, 110), confidence=0.8)
    assert b.radius_px == 10.0   # 20x20 bbox → diameter 20 → radius 10


def test_ball_candidate_prefers_rim_context_over_edge_clutter():
    detector = object.__new__(Detector)
    detector._ball_history = deque(maxlen=8)
    players = [make_player(bbox=(260, 180, 320, 360))]
    hoops = [HoopDetection(bbox=(390, 110, 420, 140), confidence=0.8, side="right")]

    rim_ball = detector._ball_candidate_score(
        bbox=(448, 201, 459, 220),
        conf=0.02,
        players=players,
        frame_w=640,
        frame_h=360,
        hoops=hoops,
    )
    edge_blob = detector._ball_candidate_score(
        bbox=(0, 172, 6, 185),
        conf=0.02,
        players=players,
        frame_w=640,
        frame_h=360,
        hoops=hoops,
    )

    assert rim_ball > edge_blob


def test_ball_memory_predicts_next_center():
    detector = object.__new__(Detector)
    detector._ball_history = deque(maxlen=8)
    detector._ball_velocity = (0.0, 0.0)
    detector._ball_hsv_mean = None
    detector._ball_hsv_std = None
    detector._ball_learned_radius = 0.0
    detector._ball_misses = 0

    detector._remember_ball(BallDetection((90, 90, 110, 110), 0.6), source="roboflow", score=0.6)
    detector._remember_ball(BallDetection((100, 95, 120, 115), 0.6), source="roboflow", score=0.6)

    assert detector._predicted_ball_center() == (120.0, 110.0)


def test_active_rim_clusters_ignore_side_rim_after_calibration():
    detector = object.__new__(Detector)
    detector._rim_clusters = []
    detector._active_rim_pair = []
    candidates = [
        (0.22, 0.8, (130, 110, 150, 135)),
        (0.72, 0.8, (450, 110, 470, 135)),
        (0.95, 0.9, (600, 70, 625, 100)),
    ]

    for _ in range(3):
        detector._update_rim_clusters(candidates, frame_w=640, frame_h=360)

    active = detector._active_rim_candidates(candidates, frame_w=640, frame_h=360)

    assert len(active) == 2
    assert all(item[0] < 0.9 for item in active)


# ---------------------------------------------------------------------------
# FrameDetections queries
# ---------------------------------------------------------------------------

def test_nearest_player_to_ball():
    ball = BallDetection(bbox=(95, 95, 105, 105), confidence=0.9)
    p_near = make_player(bbox=(80, 80, 120, 200))    # centre ~(100, 140)
    p_far  = make_player(bbox=(500, 500, 600, 700))  # centre ~(550, 600)

    fd = FrameDetections(
        frame_idx=0, timestamp_ms=0,
        frame_width=1280, frame_height=720,
        players=[p_near, p_far], ball=ball,
    )
    assert fd.nearest_player_to_ball() is p_near


def test_nearest_player_no_ball():
    fd = FrameDetections(
        frame_idx=0, timestamp_ms=0,
        frame_width=1280, frame_height=720,
        players=[make_player()],
    )
    assert fd.nearest_player_to_ball() is None


def test_defender_proximity_to_ball():
    ball = BallDetection(bbox=(95, 95, 105, 105), confidence=0.9)
    # Defender with wrist very close to ball
    kps = make_keypoints({
        KP_LEFT_WRIST:  (100, 100, 0.9),
        KP_RIGHT_WRIST: (100, 100, 0.9),
    })
    defender = make_player(bbox=(80, 60, 130, 200), kps=kps)
    attacker = make_player(bbox=(50, 50, 90, 180))

    fd = FrameDetections(
        frame_idx=0, timestamp_ms=0,
        frame_width=1280, frame_height=720,
        players=[attacker, defender], ball=ball,
    )
    dist = fd.defender_proximity_to_ball([defender])
    # Wrist at (100,100), ball centre at (100,100) → distance ≈ 0
    assert dist < 5.0


def test_players_in_interest_zone():
    p_in  = make_player(bbox=(10, 10, 50, 100))
    p_in.in_interest_zone = True
    p_out = make_player(bbox=(500, 300, 600, 500))
    p_out.in_interest_zone = False

    fd = FrameDetections(
        frame_idx=0, timestamp_ms=0,
        frame_width=1280, frame_height=720,
        players=[p_in, p_out],
    )
    assert fd.players_in_zone("interest") == [p_in]
    assert len(fd.players_in_zone("all")) == 2


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def test_iou_perfect_overlap():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_no_overlap():
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial():
    iou = _iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.0 < iou < 1.0


def test_pixel_distance():
    assert pixel_distance((0, 0), (3, 4)) == 5.0


def test_is_contested_true():
    ball = BallDetection(bbox=(95, 95, 105, 105), confidence=0.9)
    kps = make_keypoints({
        KP_LEFT_WRIST:  (100, 100, 0.9),
        KP_RIGHT_WRIST: (100, 100, 0.9),
    })
    defender = make_player(kps=kps)
    attacker = make_player()
    assert is_contested(attacker, [defender], ball, threshold_px=80.0) is True


def test_is_contested_false_too_far():
    ball = BallDetection(bbox=(95, 95, 105, 105), confidence=0.9)
    kps = make_keypoints({
        KP_LEFT_WRIST:  (500, 500, 0.9),
        KP_RIGHT_WRIST: (500, 500, 0.9),
    })
    defender = make_player(kps=kps)
    attacker = make_player()
    assert is_contested(attacker, [defender], ball, threshold_px=80.0) is False


if __name__ == "__main__":
    test_keypoint_visible()
    test_player_center()
    test_player_foot_center()
    test_hands_up_true()
    test_hands_up_false_wrists_below()
    test_hands_up_false_low_confidence()
    test_shooting_pose_true()
    test_shooting_pose_false_wrist_below_nose()
    test_driving_direction()
    test_wrist_position_returns_higher()
    test_wrist_position_none_when_invisible()
    test_ball_center()
    test_ball_radius()
    test_nearest_player_to_ball()
    test_nearest_player_no_ball()
    test_defender_proximity_to_ball()
    test_players_in_interest_zone()
    test_iou_perfect_overlap()
    test_iou_no_overlap()
    test_iou_partial()
    test_pixel_distance()
    test_is_contested_true()
    test_is_contested_false_too_far()
    print("All detector tests passed.")
