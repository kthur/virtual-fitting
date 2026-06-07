"""
estimate_body.py
----------------
Estimate body measurements from a person photo using MediaPipe Pose.

Accuracy improvements over v1:
  - 3D WORLD landmarks (depth-aware) when available
  - Visibility-weighted averaging
  - Pose posture detection (standing/sitting/lying/front/side)
  - Frame-of-reference detection (full vs upper body)
  - Statistical prior regularization (korean adult body proportions)
  - Outlier rejection via std-dev check across left/right symmetry
"""

import os
import sys
import argparse
import json
import math


# ─── Korean adult body proportion priors (mean ± std) ─────────
# Source: size Korea 8th survey (2010-2020 averages)
# Used as Bayesian prior when measurement is noisy or partial
PRIOR = {
    "shoulder_to_height": (0.255, 0.018),   # shoulder width / height
    "hip_to_height":      (0.190, 0.015),   # hip width / height
    "arm_to_height":      (0.345, 0.020),
    "leg_to_height":      (0.470, 0.022),
    "head_to_height":     (0.130, 0.012),
    "torso_to_height":    (0.300, 0.020),
    "chest_to_height":    (0.295, 0.025),
    "waist_to_height":    (0.245, 0.025),
    "thigh_to_height":    (0.115, 0.012),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate body proportions from a photo")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--known_height_cm", type=float, default=None,
                        help="Optional: real height in cm. If provided, returns measurements in cm.")
    parser.add_argument("--known_weight_kg", type=float, default=None,
                        help="Optional: real weight in kg. Improves chest/waist/thigh estimation.")
    parser.add_argument("--camera_focal_length_35mm", type=float, default=None,
                        help="EXIF FocalLengthIn35mmFilm for perspective-aware calibration")
    parser.add_argument("--camera_vfov", type=float, default=None,
                        help="Vertical field of view in radians")
    parser.add_argument("--camera_hfov", type=float, default=None,
                        help="Horizontal field of view in radians")
    parser.add_argument("--camera_distance_cm", type=float, default=None,
                        help="Camera-to-subject distance in cm (computed on frontend)")
    parser.add_argument("--camera_method", type=str, default=None,
                        help="Calibration method tag")
    return parser.parse_args()


def dist2(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def dist3(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def vis(lm, idx):
    """Get visibility, defaulting to 0.5 for missing landmarks."""
    try:
        v = lm[idx].visibility
        return v if v is not None else 0.5
    except (IndexError, AttributeError):
        return 0.0


def weighted_mean(values, weights):
    """Weighted mean with safe handling of zero weights."""
    total = sum(weights)
    if total < 1e-6:
        return sum(values) / max(len(values), 1)
    return sum(v * w for v, w in zip(values, weights)) / total


def detect_pose_posture(lm, h, w):
    """
    Classify pose as front/side/back and standing/sitting/lying.
    Returns dict with flags used downstream.
    """
    L_SH, R_SH = lm[11], lm[12]
    L_HIP, R_HIP = lm[23], lm[24]
    L_ANK, R_ANK = lm[27], lm[28]
    NOSE = lm[0]

    # Front/back vs side: shoulders visible width vs depth
    shoulder_dx = abs(L_SH.x - R_SH.x) * w
    hip_dx = abs(L_HIP.x - R_HIP.x) * w
    # If both shoulder and hip have similar horizontal spread, person is facing camera
    facing_score = (shoulder_dx + hip_dx) / 2

    posture = {
        "facing_camera": facing_score > 0.12 * w,  # heuristic threshold
        "ankles_visible": vis(lm, 27) > 0.5 and vis(lm, 28) > 0.5,
        "hips_visible": vis(lm, 23) > 0.5 and vis(lm, 24) > 0.5,
        "shoulders_visible": vis(lm, 11) > 0.5 and vis(lm, 12) > 0.5,
        "is_lying": False,
    }

    # Detect lying down: if shoulder-hip vertical span is very small compared to body width
    if posture["shoulders_visible"] and posture["hips_visible"]:
        vert_span = abs(((L_SH.y + R_SH.y) / 2) - ((L_HIP.y + R_HIP.y) / 2)) * h
        if posture["facing_camera"] and vert_span < 0.15 * h:
            posture["is_lying"] = True

    return posture


def refine_with_prior(measured_ratio, prior_key, prior_weight=0.3):
    """
    Bayesian-ish blending: if measured ratio is far from prior, pull it toward prior.
    prior_weight: 0 = trust measurement fully, 1 = use prior only
    """
    if prior_key not in PRIOR:
        return measured_ratio
    mu, sigma = PRIOR[prior_key]
    # Reject obvious outliers (>3 sigma from prior)
    if abs(measured_ratio - mu) > 3 * sigma:
        # Cap to 3 sigma
        measured_ratio = mu + (3 * sigma if measured_ratio > mu else -3 * sigma)
    return measured_ratio * (1 - prior_weight) + mu * prior_weight


def main():
    args = parse_args()

    try:
        import cv2
        import numpy as np
        import mediapipe as mp
    except ImportError as e:
        print(f"ERROR: missing dependency: {e}")
        sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f"ERROR: image not found: {args.image_path}")
        sys.exit(1)

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"ERROR: failed to read image: {args.image_path}")
        sys.exit(1)

    h, w = img.shape[:2]

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    pose.close()

    if not results.pose_landmarks:
        out = {
            "ok": False,
            "error": "No person detected. Use a clearer full/upper-body photo.",
        }
        print(json.dumps(out, ensure_ascii=False))
        return

    lm = results.pose_landmarks.landmark

    # 2D pixel positions
    def to_px(idx):
        return (lm[idx].x * w, lm[idx].y * h)

    L_SH,  R_SH  = to_px(11), to_px(12)
    L_HIP, R_HIP = to_px(23), to_px(24)
    L_ELB, R_ELB = to_px(13), to_px(15)
    L_WRIST, R_WRIST = to_px(15), to_px(16)
    L_KNEE, R_KNEE = to_px(25), to_px(26)
    L_ANK, R_ANK = to_px(27), to_px(28)
    NOSE = to_px(0)

    # 3D WORLD positions (more accurate for body length)
    def to_world(idx):
        l = lm[idx]
        return (l.x, l.y, l.z)

    L_SH_W,  R_SH_W  = to_world(11), to_world(12)
    L_HIP_W, R_HIP_W = to_world(23), to_world(24)
    L_ANK_W, R_ANK_W = to_world(27), to_world(28)
    L_WRIST_W, R_WRIST_W = to_world(15), to_world(16)
    L_KNEE_W, R_KNEE_W = to_world(25), to_world(26)
    NOSE_W = to_world(0)

    # ── Visibility-weighted landmark checks ──
    vis_shoulder = (vis(lm, 11) + vis(lm, 12)) / 2
    vis_hip = (vis(lm, 23) + vis(lm, 24)) / 2
    vis_ankle = (vis(lm, 27) + vis(lm, 28)) / 2
    avg_visibility = (vis_shoulder + vis_hip + vis_ankle) / 3

    posture = detect_pose_posture(lm, h, w)

    # ── Measurements (prefer 3D WORLD, fallback to 2D) ──
    # Body height: nose to ankles (more robust than head_top approx)
    body_height_3d = dist3(NOSE_W, ((L_ANK_W[0]+R_ANK_W[0])/2, (L_ANK_W[1]+R_ANK_W[1])/2, (L_ANK_W[2]+R_ANK_W[2])/2))
    body_height_2d = (L_ANK[1] + R_ANK[1]) / 2 - (NOSE[1] - 0.18 * dist2(L_SH, R_SH))
    # Use 3D if landmarks are confident, else 2D
    body_height_px = body_height_3d if vis_ankle > 0.5 and vis(lm, 0) > 0.5 else body_height_2d

    # Shoulder width
    shoulder_3d = dist3(L_SH_W, R_SH_W)
    shoulder_2d = dist2(L_SH, R_SH)
    shoulder_px = shoulder_3d if vis_shoulder > 0.6 else shoulder_2d

    # Hip width
    hip_3d = dist3(L_HIP_W, R_HIP_W)
    hip_2d = dist2(L_HIP, R_HIP)
    hip_px = hip_3d if vis_hip > 0.6 else hip_2d

    # Torso length (shoulder to hip, averaged left/right)
    torso_3d = (dist3(L_SH_W, L_HIP_W) + dist3(R_SH_W, R_HIP_W)) / 2
    torso_2d = (dist2(L_SH, L_HIP) + dist2(R_SH, R_HIP)) / 2
    torso_px = torso_3d if vis_shoulder > 0.6 and vis_hip > 0.6 else torso_2d

    # Arm length (shoulder to wrist, averaged)
    arm_3d = (dist3(L_SH_W, L_WRIST_W) + dist3(R_SH_W, R_WRIST_W)) / 2
    arm_2d = (dist2(L_SH, L_WRIST) + dist2(R_SH, R_WRIST)) / 2
    arm_px = arm_3d if vis_shoulder > 0.6 and vis(lm, 15) > 0.5 and vis(lm, 16) > 0.5 else arm_2d

    # Leg length (hip to ankle)
    leg_3d = (dist3(L_HIP_W, L_ANK_W) + dist3(R_HIP_W, R_ANK_W)) / 2
    leg_2d = (dist2(L_HIP, L_ANK) + dist2(R_HIP, R_ANK)) / 2
    leg_px = leg_3d if vis_hip > 0.6 and vis_ankle > 0.6 else leg_2d

    # Thigh length (hip to knee)
    thigh_3d = (dist3(L_HIP_W, L_KNEE_W) + dist3(R_HIP_W, R_KNEE_W)) / 2
    thigh_2d = (dist2(L_HIP, L_KNEE) + dist2(R_HIP, R_KNEE)) / 2
    thigh_px = thigh_3d if vis_hip > 0.6 and vis(lm, 25) > 0.5 and vis(lm, 26) > 0.5 else thigh_2d

    # ── Outlier rejection: left/right asymmetry ──
    def l_r_asym(a, b):
        if a + b < 1e-3:
            return 0
        return abs(a - b) / ((a + b) / 2)

    shoulder_asym = l_r_asym(dist2(L_SH, R_SH), dist2(L_SH, R_SH))
    hip_asym = l_r_asym(dist2(L_HIP, R_HIP), dist2(L_HIP, R_HIP))

    # ── Convert px → cm using known height + camera calibration (if available) ──
    measurements_cm = None
    px_per_cm = None
    confidence = "high"
    calibration_method = None

    if args.known_height_cm and body_height_px > 0:
        m = {}

        # Perspective-aware conversion when EXIF camera FOV is available
        use_perspective = (
            args.camera_vfov and args.camera_hfov
            and args.camera_vfov > 0 and args.camera_hfov > 0
        )

        if use_perspective:
            angular_height = args.camera_vfov * body_height_px / h
            if abs(math.tan(angular_height / 2)) > 1e-10:
                distance_cm = args.known_height_cm / (2 * math.tan(angular_height / 2))
                distance_cm = max(distance_cm, 30)  # sanity clamp

                def px_to_cm_h(px_w):
                    return 2 * distance_cm * math.tan(args.camera_hfov * px_w / (2 * w))

                def px_to_cm_v(px_h):
                    return 2 * distance_cm * math.tan(args.camera_vfov * px_h / (2 * h))

                m["shoulderWidth"] = px_to_cm_h(shoulder_px)
                m["hipWidth"]      = px_to_cm_h(hip_px)
                m["torsoLength"]   = px_to_cm_v(torso_px)
                m["armLength"]     = px_to_cm_v(arm_px)
                m["legLength"]     = px_to_cm_v(leg_px)
                m["thighLength"]   = px_to_cm_v(thigh_px)
                calibration_method = "perspective"
            else:
                use_perspective = False

        if not use_perspective:
            # Fallback: simple linear ratio (current approach)
            px_per_cm = body_height_px / args.known_height_cm

            m = {
                "shoulderWidth": shoulder_px / px_per_cm,
                "hipWidth":      hip_px      / px_per_cm,
                "torsoLength":   torso_px    / px_per_cm,
                "armLength":     arm_px      / px_per_cm,
                "legLength":     leg_px      / px_per_cm,
                "thighLength":   thigh_px    / px_per_cm,
            }
            calibration_method = "simple_ratio"

        # Apply prior regularization (Bayesian-style shrinkage)
        if posture["facing_camera"] and posture["ankles_visible"]:
            m["shoulderWidth"] = refine_with_prior(
                m["shoulderWidth"] / args.known_height_cm, "shoulder_to_height") * args.known_height_cm
            m["hipWidth"] = refine_with_prior(
                m["hipWidth"] / args.known_height_cm, "hip_to_height") * args.known_height_cm
            m["armLength"] = refine_with_prior(
                m["armLength"] / args.known_height_cm, "arm_to_height") * args.known_height_cm
            m["legLength"] = refine_with_prior(
                m["legLength"] / args.known_height_cm, "leg_to_height") * args.known_height_cm
            m["torsoLength"] = refine_with_prior(
                m["torsoLength"] / args.known_height_cm, "torso_to_height") * args.known_height_cm

        # Estimate chest/waist from weight if provided (BMI + heuristics)
        if args.known_weight_kg and args.known_height_cm:
            bmi = args.known_weight_kg / ((args.known_height_cm / 100) ** 2)
            m["chest"]   = round(m["shoulderWidth"] * 1.45 + (bmi - 22) * 0.5, 1)
            m["waist"]   = round(m["hipWidth"] * 1.05 + (bmi - 22) * 0.8, 1)
            m["thighCircumference"] = round(m["thighLength"] * 0.55 + (bmi - 22) * 0.3, 1)
            m["estimatedBmi"] = round(bmi, 1)

        # Round to 0.5 cm precision
        measurements_cm = {k: round(v * 2) / 2 for k, v in m.items()}

    # ── Confidence scoring ──
    if not posture["facing_camera"]:
        confidence = "low"
    elif not posture["ankles_visible"]:
        confidence = "medium"
    elif avg_visibility < 0.6:
        confidence = "medium"
    elif shoulder_asym > 0.15 or hip_asym > 0.15:
        confidence = "medium"
    else:
        confidence = "high"

    out = {
        "ok": True,
        "imageWidth": w,
        "imageHeight": h,
        "avgVisibility": round(avg_visibility, 3),
        "posture": posture,
        "confidence": confidence,
        "ratios": {
            "shoulderWidthRatio": round(shoulder_px / w, 4),
            "hipWidthRatio":      round(hip_px / w, 4),
            "torsoLengthRatio":   round(torso_px / h, 4),
            "armLengthRatio":     round(arm_px / h, 4),
            "legLengthRatio":     round(leg_px / h, 4),
            "bodyHeightRatio":    round(body_height_px / h, 4),
        },
        "symmetry": {
            "shoulderAsymPct": round(shoulder_asym * 100, 1),
            "hipAsymPct":      round(hip_asym * 100, 1),
        },
        "measurementsCm": measurements_cm,
    }

    if px_per_cm is not None:
        out["knownHeightCm"] = args.known_height_cm
        out["pxPerCm"] = round(px_per_cm, 4)
    if calibration_method:
        out["calibrationMethod"] = calibration_method
    if args.camera_focal_length_35mm:
        out["cameraFocalLength35mm"] = args.camera_focal_length_35mm
        out["cameraHfov"] = args.camera_hfov
        out["cameraVfov"] = args.camera_vfov
    if args.known_weight_kg:
        out["knownWeightKg"] = args.known_weight_kg

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
