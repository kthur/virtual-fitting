"""
estimate_body.py
----------------
Estimate approximate body measurements from a person photo using MediaPipe Pose.

Outputs (all in arbitrary "pixel ratios" since we don't know the real camera distance):
  - shoulderWidthRatio  (ratio of shoulder width to image width)
  - torsoLengthRatio    (ratio of neck->hip to image height)
  - armLengthRatio      (ratio of shoulder->wrist to image height)
  - bodyHeightRatio     (ratio of head_top->ankle to image height)
  - visibleLandmarks    (count of detected key landmarks)

The client (frontend) is expected to convert these ratios into real cm values
by asking the user for their real height in cm and scaling.
"""

import os
import sys
import argparse
import json
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate body proportions from a photo")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--known_height_cm", type=float, default=None,
                        help="Optional: real height in cm. If provided, returns measurements in cm.")
    return parser.parse_args()


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


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
        min_detection_confidence=0.5
    )

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    pose.close()

    if not results.pose_landmarks:
        out = {
            "ok": False,
            "error": "No person detected. Use a clearer full/upper-body photo."
        }
        print(json.dumps(out, ensure_ascii=False))
        return

    lm = results.pose_landmarks.landmark

    def to_px(idx):
        return (lm[idx].x * w, lm[idx].y * h)

    # Key landmarks
    L_SH, R_SH = to_px(mp_pose.PoseLandmark.LEFT_SHOULDER),  to_px(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    L_HIP, R_HIP = to_px(mp_pose.PoseLandmark.LEFT_HIP),    to_px(mp_pose.PoseLandmark.RIGHT_HIP)
    L_ELB, R_ELB = to_px(mp_pose.PoseLandmark.LEFT_ELBOW),  to_px(mp_pose.PoseLandmark.RIGHT_ELBOW)
    L_WRIST, R_WRIST = to_px(mp_pose.PoseLandmark.LEFT_WRIST), to_px(mp_pose.PoseLandmark.RIGHT_WRIST)
    NOSE = to_px(mp_pose.PoseLandmark.NOSE)
    L_ANK, R_ANK = to_px(mp_pose.PoseLandmark.LEFT_ANKLE), to_px(mp_pose.PoseLandmark.RIGHT_ANKLE)

    # Visibility check
    vis_scores = [lm[i].visibility for i in [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]]
    avg_visibility = sum(vis_scores) / len(vis_scores)

    # Pixel distances
    shoulder_px = distance(L_SH, R_SH)
    hip_px = distance(L_HIP, R_HIP)
    torso_px = (distance(L_SH, L_HIP) + distance(R_SH, R_HIP)) / 2.0
    arm_px = (distance(L_SH, L_WRIST) + distance(R_SH, R_WRIST)) / 2.0
    head_top_px = NOSE[1] - 0.18 * (distance(L_SH, R_SH))  # approx top of head
    body_height_px = ((R_ANK[1] + L_ANK[1]) / 2.0) - max(head_top_px, 0)

    # Ratios (relative to image size)
    ratios = {
        "shoulderWidthRatio": round(shoulder_px / w, 4),
        "hipWidthRatio":      round(hip_px / w, 4),
        "torsoLengthRatio":   round(torso_px / h, 4),
        "armLengthRatio":     round(arm_px / h, 4),
        "bodyHeightRatio":    round(body_height_px / h, 4),
    }

    out = {
        "ok": True,
        "imageWidth": w,
        "imageHeight": h,
        "avgVisibility": round(avg_visibility, 3),
        "ratios": ratios,
        "measurementsCm": None,
    }

    # If user provided a real height, convert to cm
    if args.known_height_cm and body_height_px > 0:
        px_per_cm = body_height_px / args.known_height_cm
        out["knownHeightCm"] = args.known_height_cm
        out["pxPerCm"] = round(px_per_cm, 4)
        out["measurementsCm"] = {
            "shoulderWidth": round(shoulder_px / px_per_cm, 1),
            "hipWidth":      round(hip_px / px_per_cm, 1),
            "torsoLength":   round(torso_px / px_per_cm, 1),
            "armLength":     round(arm_px / px_per_cm, 1),
        }

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
