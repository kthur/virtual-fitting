import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Patch huggingface_hub.cached_download to avoid ImportError with newer versions
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "cached_download"):
        def dummy_cached_download(*args, **kwargs):
            return huggingface_hub.hf_hub_download(*args, **kwargs)
        huggingface_hub.cached_download = dummy_cached_download
except ImportError:
    pass

from diffusers import StableDiffusionInpaintPipeline


try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except ImportError:
    mp = None
    mp_pose = None

# SegFormer LIP label map (from model config)
# 0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes
# 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe
# 10: Right-shoe, 11: Face, 12: Left-leg, 13: Right-leg
# 14: Left-arm, 15: Right-arm, 16: Bag, 17: Scarf

# Region that MUST never be inpainted (skin, face, hair, limbs).
# Keeping face/skin in this set ensures the deepfake illusion: the
# person is preserved at the pixel level; only clothing changes.
DEFAULT_PRESERVE_LABELS = {2, 9, 10, 11, 12, 13, 14, 15}  # Hair, shoes, face, legs, arms/hands
BACKGROUND_LABEL = 0                                     # Background - never mask


def parse_args():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Local AI Virtual Try-On (Deepfake-quality)")
    parser.add_argument("--person_path", type=str, required=True)
    parser.add_argument("--garment_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a garment")
    parser.add_argument("--garment_type", type=str, default=None)
    parser.add_argument("--fit_type", type=str, default="regular")
    parser.add_argument("--colors", type=str, default="")
    parser.add_argument("--materials", type=str, default="")
    parser.add_argument("--styles", type=str, default="")
    parser.add_argument("--ip_scale", type=float, default=0.55,
                        help="IP-Adapter scale (lower = safer face preservation). 0.4~0.7 recommended")
    parser.add_argument("--inference_steps", type=int, default=35)
    parser.add_argument("--target_w", type=int, default=768)
    parser.add_argument("--target_h", type=int, default=1024)
    parser.add_argument("--upscale", action="store_true",
                        help="Apply Real-ESRGAN upscale after synthesis for sharper details")
    parser.add_argument("--use_controlnet", action="store_true",
                        help="Use ControlNet OpenPose for accurate clothing deformation on unusual poses")
    parser.add_argument("--controlnet_scale", type=float, default=0.6,
                        help="ControlNet conditioning scale (0.3~0.9, higher = pose-following)")
    return parser.parse_args()


def remove_background(img_pil):
    try:
        from rembg import remove as rembg_remove
        print("[AI] Removing garment background with rembg...")
        img_rgba = rembg_remove(img_pil.convert("RGBA"))
        white_bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        white_bg.paste(img_rgba, mask=img_rgba.split()[3])
        result = white_bg.convert("RGB")
        print("[AI] Background removal complete.")
        return result
    except ImportError:
        print("[AI] [WARN] rembg not installed, skipping background removal.")
        return img_pil
    except Exception as e:
        print(f"[AI] [WARN] Background removal failed: {e}, using original.")
        return img_pil


def build_prompt(base_prompt, garment_type, fit_type, colors, materials, styles):
    fit_map = {
        "slim":    "slim fit, fitted, form-fitting, tailored silhouette",
        "overfit": "oversized, loose, baggy, oversized fit",
        "regular": "regular fit, standard fit",
    }
    fit_text = fit_map.get(fit_type, "regular fit")

    feature_parts = []
    if colors:
        feature_parts.append(colors)
    if materials:
        feature_parts.append(f"{materials} fabric")
    if styles:
        feature_parts.append(styles)
        if "short sleeve" in styles.lower() or "반소매" in styles.lower():
            feature_parts.append("showing bare forearms, bare skin arms, short sleeves")
    feature_str = ", ".join(feature_parts) if feature_parts else ""

    full = f"a person wearing {base_prompt}"
    if feature_str:
        full += f", {feature_str}"
    full += f", {fit_text}"
    full += ", high quality, photorealistic, 8k, detailed fabric texture, natural lighting, fashion photography"

    neg_parts = [
        "deformed, bad anatomy, ugly, blurry, low quality, distorted, extra limbs",
        "mutated hands, fused fingers, missing fingers, watermark, duplicate",
        "bad proportions, extra arms, unnatural body, cartoon, drawing, painted",
        "different face, different person, changed face, different identity",
    ]
    if garment_type == "upper" or garment_type is None:
        neg_parts.append("long sleeves, suit, jacket, tie, collar, white shirt, undershirt, necktie, business suit")

    negative = ", ".join(neg_parts)
    return full, negative


def build_face_protect_mask(person_pil, seg_model, seg_processor, device, w, h, lm=None, garment_type=None):
    """
    Build a mask of the FACE region that should NEVER be inpainted.
    This mask is binary (0 or 255). 255 = preserve, 0 = safe to change.
    """
    arr = np.array(person_pil.resize((w, h))).astype(np.uint8)
    inputs = seg_processor(images=arr, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = seg_model(**inputs)
    upsampled = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
    pred = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Reclassify white collar misclassified as Face (11) to Upper-clothes (4)
    if lm is not None:
        mouth_y = (lm[9].y + lm[10].y) / 2.0
        mouth_y_px = int(mouth_y * h)
        y_indices = np.arange(h).reshape(h, 1)
        white_mask = (arr[:, :, 0] > 190) & (arr[:, :, 1] > 190) & (arr[:, :, 2] > 190) & (np.abs(arr[:, :, 0].astype(np.int32) - arr[:, :, 2].astype(np.int32)) < 22)
        pred[(pred == 11) & (y_indices >= mouth_y_px) & white_mask] = 4

    # Face (11) + hair (2) + neck skin (sometimes mis-classed) + a small safety margin
    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_mask[pred == 11] = 255
    face_mask[pred == 2]  = 255

    if lm is not None:
        # Reduce the face mask dilation kernel (use a smaller ellipse kernel like (11, 11) with 1 iteration)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        face_mask = cv2.dilate(face_mask, k, iterations=1)
        
        # Calculate chin Y
        mouth_y = (lm[9].y + lm[10].y) / 2.0
        nose_y = lm[0].y
        chin_y = mouth_y + 1.2 * (mouth_y - nose_y)
        chin_y_px = int(chin_y * h)
        
        # Zero out the face mask below chin_y_px globally
        chin_y_clip = max(0, min(h, chin_y_px))
        face_mask[chin_y_clip:, :] = 0
        face_mask[pred == 2] = 255  # Protect hair below chin
        
        g_type = garment_type or "upper"
        # Hand protection (for 'upper', 'outer', 'full')
        if g_type in ["upper", "outer", "full"]:
            left_hand_pts = np.array([
                [int(lm[15].x * w), int(lm[15].y * h)],
                [int(lm[17].x * w), int(lm[17].y * h)],
                [int(lm[19].x * w), int(lm[19].y * h)],
                [int(lm[21].x * w), int(lm[21].y * h)]
            ], dtype=np.int32)
            right_hand_pts = np.array([
                [int(lm[16].x * w), int(lm[16].y * h)],
                [int(lm[18].x * w), int(lm[18].y * h)],
                [int(lm[20].x * w), int(lm[20].y * h)],
                [int(lm[22].x * w), int(lm[22].y * h)]
            ], dtype=np.int32)
            
            hand_mask = np.zeros((h, w), dtype=np.uint8)
            hull_left = cv2.convexHull(left_hand_pts)
            cv2.drawContours(hand_mask, [hull_left], -1, 255, -1)
            hull_right = cv2.convexHull(right_hand_pts)
            cv2.drawContours(hand_mask, [hull_right], -1, 255, -1)
            
            k_hand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            hand_mask = cv2.dilate(hand_mask, k_hand, iterations=1)
            
            # Add to face protection mask
            face_mask = cv2.bitwise_or(face_mask, hand_mask)
            
        # Feet protection (for 'lower', 'full')
        if g_type in ["lower", "full"]:
            left_foot_pts = np.array([
                [int(lm[27].x * w), int(lm[27].y * h)],
                [int(lm[29].x * w), int(lm[29].y * h)],
                [int(lm[31].x * w), int(lm[31].y * h)]
            ], dtype=np.int32)
            right_foot_pts = np.array([
                [int(lm[28].x * w), int(lm[28].y * h)],
                [int(lm[30].x * w), int(lm[30].y * h)],
                [int(lm[32].x * w), int(lm[32].y * h)]
            ], dtype=np.int32)
            
            foot_mask = np.zeros((h, w), dtype=np.uint8)
            hull_left_foot = cv2.convexHull(left_foot_pts)
            cv2.drawContours(foot_mask, [hull_left_foot], -1, 255, -1)
            hull_right_foot = cv2.convexHull(right_foot_pts)
            cv2.drawContours(foot_mask, [hull_right_foot], -1, 255, -1)
            
            k_foot = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            foot_mask = cv2.dilate(foot_mask, k_foot, iterations=1)
            
            # Add to face protection mask
            face_mask = cv2.bitwise_or(face_mask, foot_mask)
    else:
        # Dilate to add safety margin around face/hair
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        face_mask = cv2.dilate(face_mask, k, iterations=2)
        
    # Ensure no clothing pixels are protected by the face mask
    for label in [4, 5, 6, 7, 8, 17]:
        face_mask[pred == label] = 0
        
    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
    return face_mask, pred


def generate_clothing_mask(seg_model, seg_processor, person_resized,
                           target_w, target_h, garment_type, device,
                           preserve_labels, lm=None):
    """
    Clothing mask = ONLY the clothing pixels.  Face, skin, hair, limbs, background are all
    explicitly zeroed out. The dilation step is corrected by re-applying PRESERVE.
    """
    inputs = processor_safe(seg_processor, person_resized, device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
        logits = outputs.logits

    upsampled = F.interpolate(logits, size=(target_h, target_w),
                              mode="bilinear", align_corners=False)
    pred_seg = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Reclassify white collar misclassified as Face (11) to Upper-clothes (4)
    if lm is not None:
        mouth_y = (lm[9].y + lm[10].y) / 2.0
        mouth_y_px = int(mouth_y * target_h)
        person_np = np.array(person_resized)
        y_indices = np.arange(target_h).reshape(target_h, 1)
        white_mask = (person_np[:, :, 0] > 190) & (person_np[:, :, 1] > 190) & (person_np[:, :, 2] > 190) & (np.abs(person_np[:, :, 0].astype(np.int32) - person_np[:, :, 2].astype(np.int32)) < 22)
        pred_seg[(pred_seg == 11) & (y_indices >= mouth_y_px) & white_mask] = 4

    g_type = garment_type or "upper"
    if g_type == "lower":
        clothing_labels = {5, 6, 8}                 # Skirt, Pants, Belt
    elif g_type == "full":
        clothing_labels = {4, 5, 6, 7, 8, 17}       # Upper-clothes, Skirt, Pants, Dress, Belt, Scarf
    elif g_type == "outer":
        clothing_labels = {4, 17}                   # Upper-clothes, Scarf
    else:                                           # upper (default)
        clothing_labels = {4, 17}                   # Upper-clothes, Scarf

    print(f"[AI] Garment type: {g_type.upper()} | Mask labels: {clothing_labels}")

    mask_np = np.zeros(pred_seg.shape, dtype=np.uint8)
    for label in clothing_labels:
        mask_np[pred_seg == label] = 255

    # Explicitly ZERO OUT face, skin, limbs
    for label in preserve_labels:
        mask_np[pred_seg == label] = 0

    # Dynamic Neck Collar Masking: Neck trapezoid polygon
    neck_mask = None
    if lm is not None:
        neck_mask = np.zeros(pred_seg.shape, dtype=np.uint8)
        mouth_y = (lm[9].y + lm[10].y) / 2.0
        nose_y = lm[0].y
        chin_y = mouth_y + 1.2 * (mouth_y - nose_y)
        chin_y_px = int(chin_y * target_h)
        
        pt1 = (int(lm[9].x * target_w), chin_y_px)
        pt2 = (int(lm[10].x * target_w), chin_y_px)
        pt3 = (int(lm[12].x * target_w), int(lm[12].y * target_h))
        pt4 = (int(lm[11].x * target_w), int(lm[11].y * target_h))
        
        pts = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
        cv2.fillPoly(neck_mask, [pts], 255)
        
        # Merge it into the base clothing mask
        mask_np = cv2.bitwise_or(mask_np, neck_mask)

    # Modest dilation to capture clothing edges
    kernel = np.ones((7, 7), np.uint8)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)

    # Re-apply preserve: undo any dilation into skin areas
    for label in preserve_labels:
        mask_dilated[pred_seg == label] = 0
    # Background must never be in the mask either
    mask_dilated[pred_seg == BACKGROUND_LABEL] = 0

    # Make sure neck mask is fully covered and not zeroed out by background/preserve
    if lm is not None and neck_mask is not None:
        mask_dilated = cv2.bitwise_or(mask_dilated, neck_mask)

    mask_blurred = cv2.GaussianBlur(mask_dilated, (11, 11), 0)

    print(f"[AI] Mask coverage: {(mask_blurred > 0).sum()} / {mask_blurred.size} px")
    return mask_dilated, mask_blurred, pred_seg


def processor_safe(seg_processor, img, device):
    """Run SegFormer image processor on PIL image, moving tensors to device."""
    inputs = seg_processor(images=img, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    return inputs


def compute_color_match(orig_np, result_np, mask_np, sample_band=15):
    """
    Estimate per-channel mean color of the original image just outside the mask boundary
    and the AI result just inside. Returns a per-channel multiplicative correction
    so the AI result blends better with the original lighting.
    """
    mask_bin = (mask_np > 128).astype(np.uint8)
    # Erode mask to get "interior" of result, dilate to get "exterior" of original
    k = np.ones((sample_band * 2 + 1, sample_band * 2 + 1), np.uint8)
    interior = cv2.erode(mask_bin, k, iterations=1)
    exterior = cv2.dilate(mask_bin, k, iterations=1) - mask_bin

    if interior.sum() < 50 or exterior.sum() < 50:
        return 1.0, 0.0  # not enough pixels - skip

    orig_ext = orig_np[exterior > 0].astype(np.float32)
    res_int  = result_np[interior > 0].astype(np.float32)

    orig_mean = orig_ext.mean(axis=0)
    res_mean  = res_int.mean(axis=0)
    ratio = orig_mean / (res_mean + 1e-6)
    ratio = np.clip(ratio, 0.6, 1.4)  # never over-correct
    return ratio, orig_mean


def apply_localized_skin_match(original_pil, result_pil, lm, seg_model, seg_processor, device, garment_type, pred_seg):
    """
    Replace global color correction with local skin-tone blending applied only to skin pixels.
    """
    orig_np = np.array(original_pil).astype(np.float32)
    res_np = np.array(result_pil).astype(np.float32)
    h, w = orig_np.shape[:2]

    # Run SegFormer parser on the synthesized output image to identify skin pixels
    inputs = processor_safe(seg_processor, result_pil, device)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    upsampled = F.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
    pred_out = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # Resize pred_seg if shape doesn't match
    if pred_seg.shape[0] != h or pred_seg.shape[1] != w:
        pred_seg_resized = cv2.resize(pred_seg, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        pred_seg_resized = pred_seg

    # Convert res_np to LAB color space
    res_lab = cv2.cvtColor(res_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    # Define zones (Neck, Left Arm, Right Arm, and optionally Legs)
    zones = []

    # 1. Neck Zone
    mouth_y = (lm[9].y + lm[10].y) / 2.0
    nose_y = lm[0].y
    chin_y = mouth_y + 1.2 * (mouth_y - nose_y)
    chin_y_px = int(chin_y * h)

    ear_l_x = int(lm[7].x * w)
    ear_r_x = int(lm[8].x * w)
    x1 = min(ear_l_x, ear_r_x)
    x2 = max(ear_l_x, ear_r_x)

    Y_grid, X_grid = np.ogrid[:h, :w]
    # Neck corridor bounding box/mask
    neck_zone_mask = (pred_out == 11) & (Y_grid >= chin_y_px) & (X_grid >= x1) & (X_grid <= x2)

    neck_pt_start = np.array([int((lm[9].x + lm[10].x) / 2.0 * w), chin_y_px], dtype=np.float32)
    neck_pt_end = np.array([
        int((lm[11].x + lm[12].x) / 2.0 * w),
        int((lm[11].y + lm[12].y) / 2.0 * h)
    ], dtype=np.float32)

    zones.append({
        "name": "Neck",
        "pt_start": neck_pt_start,
        "pt_end": neck_pt_end,
        "label_seg": 11,
        "label_out": 11,
        "zone_mask": neck_zone_mask
    })

    # 2. Left Arm Zone
    la_pt_start = np.array([lm[15].x * w, lm[15].y * h], dtype=np.float32)
    la_pt_end = np.array([lm[13].x * w, lm[13].y * h], dtype=np.float32)
    zones.append({
        "name": "Left Arm",
        "pt_start": la_pt_start,
        "pt_end": la_pt_end,
        "label_seg": 14,
        "label_out": 14,
        "zone_mask": (pred_out == 14)
    })

    # 3. Right Arm Zone
    ra_pt_start = np.array([lm[16].x * w, lm[16].y * h], dtype=np.float32)
    ra_pt_end = np.array([lm[14].x * w, lm[14].y * h], dtype=np.float32)
    zones.append({
        "name": "Right Arm",
        "pt_start": ra_pt_start,
        "pt_end": ra_pt_end,
        "label_seg": 15,
        "label_out": 15,
        "zone_mask": (pred_out == 15)
    })

    # 4. Legs Zone (optionally)
    g_type = garment_type or "upper"
    if g_type in ["lower", "full"]:
        # Left Leg
        ll_pt_start = np.array([lm[27].x * w, lm[27].y * h], dtype=np.float32)
        ll_pt_end = np.array([lm[25].x * w, lm[25].y * h], dtype=np.float32)
        zones.append({
            "name": "Left Leg",
            "pt_start": ll_pt_start,
            "pt_end": ll_pt_end,
            "label_seg": 12,
            "label_out": 12,
            "zone_mask": (pred_out == 12)
        })
        
        # Right Leg
        rl_pt_start = np.array([lm[28].x * w, lm[28].y * h], dtype=np.float32)
        rl_pt_end = np.array([lm[26].x * w, lm[26].y * h], dtype=np.float32)
        zones.append({
            "name": "Right Leg",
            "pt_start": rl_pt_start,
            "pt_end": rl_pt_end,
            "label_seg": 13,
            "label_out": 13,
            "zone_mask": (pred_out == 13)
        })

    for zone in zones:
        pt_start = zone["pt_start"]
        pt_end = zone["pt_end"]
        label_seg = zone["label_seg"]
        label_out = zone["label_out"]
        zone_mask = zone["zone_mask"]

        x_s, y_s = int(pt_start[0]), int(pt_start[1])

        # Sample original skin pixels around pt_start (which is preserved, so original)
        # Using a 30x30 bounding box
        y1, y2 = max(0, y_s - 15), min(h, y_s + 15)
        x1_box, x2_box = max(0, x_s - 15), min(w, x_s + 15)

        orig_window = orig_np[y1:y2, x1_box:x2_box]
        seg_window = pred_seg_resized[y1:y2, x1_box:x2_box]
        orig_mask = (seg_window == label_seg)

        if orig_mask.sum() > 5:
            orig_pixels = orig_window[orig_mask]
        else:
            orig_pixels = orig_window.reshape(-1, 3)

        # Inpainted skin pixels: sample slightly shifted (15%) towards the elbow/shoulder
        v = pt_end - pt_start
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-3:
            continue
        pt_inp = pt_start + 0.15 * v
        x_i, y_i = int(pt_inp[0]), int(pt_inp[1])

        y1_i, y2_i = max(0, y_i - 15), min(h, y_i + 15)
        x1_i, x2_i = max(0, x_i - 15), min(w, x_i + 15)

        res_window = res_np[y1_i:y2_i, x1_i:x2_i]
        out_window = pred_out[y1_i:y2_i, x1_i:x2_i]
        inp_mask = (out_window == label_out)

        if inp_mask.sum() > 5:
            inp_pixels = res_window[inp_mask]
        else:
            inp_pixels = res_window.reshape(-1, 3)

        # Calculate average channel deltas in LAB space
        mean_orig_rgb = orig_pixels.mean(axis=0)
        mean_inp_rgb = inp_pixels.mean(axis=0)

        rgb_orig = np.uint8([[mean_orig_rgb]])
        rgb_inp = np.uint8([[mean_inp_rgb]])
        lab_orig = cv2.cvtColor(rgb_orig, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        lab_inp = cv2.cvtColor(rgb_inp, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

        dL = lab_orig[0] - lab_inp[0]
        dA = lab_orig[1] - lab_inp[1]
        dB = lab_orig[2] - lab_inp[2]

        # Apply a damping factor of 0.8 to dL
        dL = dL * 0.8

        # Project pixels onto the limb vectors to calculate normalized distance t
        ys, xs = np.where(zone_mask)
        if len(ys) > 0:
            v_x, v_y = v[0], v[1]
            dot_v_v = v_x * v_x + v_y * v_y
            if dot_v_v > 0:
                dx = xs - pt_start[0]
                dy = ys - pt_start[1]
                t = (dx * v_x + dy * v_y) / dot_v_v
                t = np.clip(t, 0.0, 1.0)
                s = 1.0 - t

                # Apply the delta scaled by s = 1 - t
                res_lab[ys, xs, 0] += s * dL
                res_lab[ys, xs, 1] += s * dA
                res_lab[ys, xs, 2] += s * dB

    # Convert corrected LAB image back to RGB
    res_lab = np.clip(res_lab, 0, 255).astype(np.uint8)
    corrected_rgb = cv2.cvtColor(res_lab, cv2.COLOR_LAB2RGB)
    return corrected_rgb.astype(np.float32)


def blend_with_face_and_bg_preservation(original_pil, result_pil, soft_mask_np,
                                        face_mask_np, edge_band=20,
                                        use_pose_enhancements=False, lm=None,
                                        seg_model=None, seg_processor=None, device=None,
                                        garment_type=None, pred_seg=None):
    """
    Three-layer composite:
      1. face/hair region       -> 100% original
      2. background region      -> 100% original
      3. clothing region        -> soft blend (soft_mask)
      4. boundary band (clothing-skin) -> color-matched gradient blend
    """
    orig = np.array(original_pil).astype(np.float32)
    res  = np.array(result_pil).astype(np.float32)

    # 1. Compute color-match correction for boundary pixels / Localized LAB blending
    if use_pose_enhancements and lm is not None and seg_model is not None:
        res_color_corrected = apply_localized_skin_match(
            original_pil, result_pil, lm, seg_model, seg_processor, device, garment_type, pred_seg
        )
    else:
        ratio, _ = compute_color_match(orig, res, soft_mask_np, sample_band=edge_band)
        res_color_corrected = np.clip(res * ratio, 0, 255)

    # 2. Build per-pixel selection map
    alpha = soft_mask_np.astype(np.float32) / 255.0
    alpha_3d = np.stack([alpha, alpha, alpha], axis=-1)

    blended = orig * (1.0 - alpha_3d) + res_color_corrected * alpha_3d

    # 3. Force face/hair to 100% original
    face_alpha = (face_mask_np.astype(np.float32) / 255.0)
    face_alpha_3d = np.stack([face_alpha, face_alpha, face_alpha], axis=-1)
    blended = blended * (1.0 - face_alpha_3d) + orig * face_alpha_3d

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# ─────────────────────────────────────────────────────────────
# OpenPose / stick-figure helpers for ControlNet conditioning
# ─────────────────────────────────────────────────────────────
# MediaPipe Pose landmark indices that map to OpenPose COCO-18 keypoints.
# OpenPose ordering used by lllyasviel/control_v11p_sd15_openpose:
#  0 nose, 1 neck, 2 r_shoulder, 3 r_elbow, 4 r_wrist,
#  5 l_shoulder, 6 l_elbow, 7 l_wrist, 8 r_hip, 9 r_knee, 10 r_ankle,
# 11 l_hip, 12 l_knee, 13 l_ankle, 14 r_eye, 15 l_eye, 16 r_ear, 17 l_ear
MP_TO_OPENPOSE = {
    0: 0,    # nose -> nose
    2: 14,   # left_eye -> r_eye (mirrored)
    5: 15,   # right_eye -> l_eye
    7: 16,   # left_ear -> r_ear
    8: 17,   # right_ear -> l_ear
    11: 5,   # left_shoulder -> l_shoulder
    12: 7,   # left_elbow -> l_elbow
    13: 7+1, # left_wrist -> l_wrist (corrected below)
    14: 2,   # right_shoulder -> r_shoulder
    15: 4,   # right_elbow -> r_elbow
    16: 4+1, # right_wrist -> r_wrist (corrected below)
    23: 11,  # left_hip -> l_hip
    25: 13,  # left_knee -> l_knee
    27: 13+1,# left_ankle -> l_ankle (corrected)
    24: 8,   # right_hip -> r_hip
    26: 10,  # right_knee -> r_knee
    28: 10+1,# right_ankle -> r_ankle (corrected)
}
# Apply explicit fixes for the +1 placeholders
MP_TO_OPENPOSE[13] = 8  # wait, 13 is left_wrist, OpenPose index 8 = r_wrist? No
# Re-derive from MediaPipe Pose spec to avoid confusion:
# 11=l_shoulder, 12=l_elbow, 13=l_wrist, 14=r_shoulder, 15=r_elbow, 16=r_wrist
# 23=l_hip, 25=l_knee, 27=l_ankle, 24=r_hip, 26=r_knee, 28=r_ankle
# OpenPose: 2=r_shoulder, 3=r_elbow, 4=r_wrist, 5=l_shoulder, 6=l_elbow, 7=l_wrist
#            8=r_hip, 9=r_knee, 10=r_ankle, 11=l_hip, 12=l_knee, 13=l_ankle
MP_TO_OPENPOSE = {
    0: 0,   # nose
    2: 14, 5: 15,           # eyes
    7: 16, 8: 17,           # ears
    11: 5, 12: 6, 13: 7,    # left arm
    14: 2, 15: 3, 16: 4,    # right arm
    23: 11, 25: 12, 27: 13, # left leg
    24: 8, 26: 9, 28: 10,   # right leg
}

# OpenPose skeleton: pairs (keypoint_a, keypoint_b)
OPENPOSE_PAIRS = [
    (0, 1),   # nose-neck
    (0, 14), (0, 15),     # nose-eyes
    (14, 16), (15, 17),   # eyes-ears
    (1, 2), (1, 5),       # neck-shoulders
    (2, 3), (3, 4),       # right arm
    (5, 6), (6, 7),       # left arm
    (1, 8), (1, 11),      # neck-hips
    (8, 9), (9, 10),      # right leg
    (11, 12), (12, 13),   # left leg
]


def mediapipe_to_openpose_stick_figure(person_pil, lm, w, h, visibility_threshold=0.4):
    """
    Render an OpenPose-compatible stick figure from MediaPipe landmarks.
    Returned image is RGB, on black background, ready for ControlNet conditioning.
    """
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Build openpose keypoint dict from mediapipe landmarks
    op = {i: None for i in range(18)}
    for mp_idx, op_idx in MP_TO_OPENPOSE.items():
        try:
            l = lm[mp_idx]
            if l.visibility >= visibility_threshold:
                op[op_idx] = (int(l.x * w), int(l.y * h))
        except (IndexError, AttributeError):
            pass

    # Neck: midpoint of shoulders; fallback midpoint of eyes
    if op[2] and op[5]:
        op[1] = ((op[2][0] + op[5][0]) // 2, (op[2][1] + op[5][1]) // 2)
    elif op[0]:
        op[1] = (op[0][0], op[0][1] + int(0.05 * h))

    # Draw bones
    for a, b in OPENPOSE_PAIRS:
        if op[a] and op[b]:
            cv2.line(canvas, op[a], op[b], (255, 255, 255), 4, cv2.LINE_AA)

    # Draw joints
    for pt in op.values():
        if pt:
            cv2.circle(canvas, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)

    n_visible = sum(1 for v in op.values() if v is not None)
    return Image.fromarray(canvas), n_visible


def extract_openpose_image(person_pil, lm, w, h):
    """
    Extract an OpenPose-style conditioning image.
    Prefers controlnet-aux (more accurate), falls back to MediaPipe stick figure.
    Returns (image, n_keypoints_or_-1_if_unavailable).
    """
    # Path A: controlnet-aux OpenposeDetector
    try:
        from controlnet_aux import OpenposeDetector
        detector = OpenposeDetector.from_pretrained(
            "lllyasviel/Annotators", filename="body_pose_model.pth")
        print("[AI] Using controlnet-aux OpenposeDetector")
        return detector(person_pil), 18
    except Exception as e:
        print(f"[AI] controlnet-aux unavailable ({type(e).__name__}: {str(e)[:80]}), "
              f"using MediaPipe stick figure")

    # Path B: MediaPipe landmarks -> stick figure
    if lm is None:
        return None, 0
    img, n = mediapipe_to_openpose_stick_figure(person_pil, lm, w, h)
    print(f"[AI] MediaPipe stick figure with {n}/18 keypoints")
    return img, n


def optional_upscale(img_pil):
    """Apply Real-ESRGAN x2 if available. Falls back silently if not installed."""
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch as _t
        print("[AI] Real-ESRGAN upscaling 2x...")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        upsampler = RealESRGANer(
            scale=2, model_path="weights/RealESRGAN_x2plus.pth",
            model=model, tile=0, tile_pad=10, pre_pad=0,
            half=_t.cuda.is_available(),
        )
        arr = np.array(img_pil)
        out, _ = upsampler.enhance(arr, outscale=2)
        return Image.fromarray(out)
    except Exception as e:
        print(f"[AI] [WARN] Real-ESRGAN unavailable ({type(e).__name__}), skipping upscale.")
        return img_pil


def main():
    args = parse_args()

    print(f"[AI] ===== Local Virtual Try-On (v3 - Deepfake Quality) =====")
    print(f"[AI] Person:   {args.person_path}")
    print(f"[AI] Garment:  {args.garment_path}")
    print(f"[AI] Output:   {args.output_path}")
    print(f"[AI] Type: {args.garment_type} | Fit: {args.fit_type}")
    print(f"[AI] Features -> Colors: '{args.colors}' | Materials: '{args.materials}'")
    print(f"[AI] IP-Adapter scale: {args.ip_scale} | Steps: {args.inference_steps} | Res: {args.target_w}x{args.target_h}")

    # 1. Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AI] Device: {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0).lower()
        print(f"[AI] GPU: {torch.cuda.get_device_name(0)}")
        if any(k in gpu_name for k in ["1660", "1650", "1080", "1070", "1060", "1050", "geforce gtx"]):
            print("[AI] GTX series detected -> forcing float32 to prevent NaN black images")
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # 2. Load images
    try:
        person_img = Image.open(args.person_path).convert("RGB")
        garment_img = Image.open(args.garment_path).convert("RGB")
    except Exception as e:
        print(f"[AI] [ERROR] Failed to load images: {e}")
        return

    orig_w, orig_h = person_img.size
    print(f"[AI] Original person image: {orig_w}x{orig_h}")

    target_w, target_h = args.target_w, args.target_h

    # 3. Garment preprocessing
    garment_clean = remove_background(garment_img)
    garment_resized = garment_clean.resize((512, 512), Image.Resampling.LANCZOS)
    person_resized = person_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 4. Human parsing + MediaPipe Pose
    print("[AI] Loading SegFormer human parser...")
    try:
        seg_processor = SegformerImageProcessor.from_pretrained(
            "matei-dorian/segformer-b5-finetuned-human-parsing")
        seg_model = SegformerForSemanticSegmentation.from_pretrained(
            "matei-dorian/segformer-b5-finetuned-human-parsing").to(device)
    except Exception as e:
        print(f"[AI] [ERROR] Failed to load parser: {e}")
        return

    # MediaPipe Pose Initialization
    use_pose_enhancements = False
    lm = None
    if mp_pose is not None:
        try:
            print("[AI] Running MediaPipe Pose on person image...")
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            person_np = np.array(person_resized)
            pose_results = pose.process(person_np)
            pose.close()
            if pose_results.pose_landmarks:
                lm = pose_results.pose_landmarks.landmark
                use_pose_enhancements = True
                print("[AI] MediaPipe Pose keypoints successfully detected.")
            else:
                print("[AI] [WARN] MediaPipe Pose landmarks not detected. Falling back to original behavior.")
        except Exception as e:
            print(f"[AI] [WARN] MediaPipe Pose execution failed: {e}. Falling back to original behavior.")
    else:
        print("[AI] [WARN] MediaPipe not available. Falling back to original behavior.")

    # Determine PRESERVE_LABELS dynamically
    if use_pose_enhancements:
        g_type = args.garment_type or "upper"
        if g_type == "lower":
            preserve_labels = {2, 4, 11, 14, 15, 17}
        elif g_type == "full":
            preserve_labels = {2, 11}
        elif g_type == "outer":
            preserve_labels = {2, 5, 6, 8, 9, 10, 11, 12, 13}
        else:  # upper
            preserve_labels = {2, 9, 10, 11, 12, 13}
        print(f"[AI] Dynamic PRESERVE_LABELS for category '{g_type}': {preserve_labels}")
    else:
        preserve_labels = DEFAULT_PRESERVE_LABELS
        print(f"[AI] Using default PRESERVE_LABELS: {preserve_labels}")

    print("[AI] Generating clothing mask and face protect mask...")
    try:
        mask_dilated, mask_blurred, pred_seg = generate_clothing_mask(
            seg_model, seg_processor, person_resized,
            target_w, target_h, args.garment_type, device,
            preserve_labels=preserve_labels, lm=lm)
        mask_image = Image.fromarray(mask_blurred)

        face_mask, _ = build_face_protect_mask(
            person_resized, seg_model, seg_processor, device, target_w, target_h,
            lm=lm, garment_type=args.garment_type)

        # Final mask used by the pipeline: clothing region MINUS face protection
        combined_mask = mask_dilated.copy()
        combined_mask[face_mask > 128] = 0
        # Smooth again
        combined_mask = cv2.GaussianBlur(combined_mask, (7, 7), 0)
        combined_mask_pil = Image.fromarray(combined_mask)

        # A slightly larger version for final blending (to cover any boundary halo)
        kernel_blend = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_blend = cv2.dilate(combined_mask, kernel_blend, iterations=1)
        mask_blend[face_mask > 128] = 0
        mask_blend[pred_seg == BACKGROUND_LABEL] = 0
        print(f"[AI] Inpaint mask: {(combined_mask > 0).sum()} px | "
              f"Face protected: {(face_mask > 128).sum()} px")
    except Exception as e:
        print(f"[AI] [ERROR] Mask generation failed: {e}")
        return

    # 5. Build prompt
    prompt, negative_prompt = build_prompt(
        args.prompt,
        garment_type=args.garment_type,
        fit_type=args.fit_type,
        colors=args.colors,
        materials=args.materials,
        styles=args.styles
    )
    print(f"[AI] Prompt: {prompt}")

    # 6. Load SD Inpainting + IP-Adapter (+ optional ControlNet OpenPose)
    print("[AI] Loading Stable Diffusion Inpainting + IP-Adapter" +
          (" + ControlNet OpenPose" if args.use_controlnet else "") + "...")
    try:
        from transformers import CLIPVisionModelWithProjection
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            torch_dtype=torch_dtype
        ).to(device)

        controlnet = None
        if args.use_controlnet:
            try:
                from diffusers import ControlNetModel
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/control_v11p_sd15_openpose",
                    torch_dtype=torch_dtype,
                )
                print("[AI] ControlNet OpenPose loaded")
            except Exception as e:
                print(f"[AI] [WARN] ControlNet load failed ({type(e).__name__}: {str(e)[:80]}), "
                      f"continuing without ControlNet")
                controlnet = None
                args.use_controlnet = False

        if controlnet is not None:
            from diffusers import StableDiffusionControlNetInpaintPipeline
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                safety_checker=None,
                image_encoder=image_encoder,
            )
        else:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch_dtype,
                safety_checker=None,
                image_encoder=image_encoder,
            )

        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                             weight_name="ip-adapter_sd15.bin")
        pipe.set_ip_adapter_scale(args.ip_scale)

        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")
    except Exception as e:
        print(f"[AI] [ERROR] Failed to load pipeline: {e}")
        return

    # 6.5 Extract ControlNet conditioning image (if enabled)
    control_image = None
    if args.use_controlnet:
        try:
            control_image, n_kp = extract_openpose_image(
                person_resized, lm, target_w, target_h)
            if n_kp < 6:
                print(f"[AI] [WARN] Only {n_kp} keypoints visible, ControlNet will have limited effect")
        except Exception as e:
            print(f"[AI] [WARN] OpenPose extraction failed: {e}")
            control_image = None
            args.use_controlnet = False

    # 7. Inference
    print("[AI] Running clothing synthesis (inpainting)...")
    try:
        generator = torch.Generator(device=device).manual_seed(42)

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person_resized,
            mask_image=combined_mask_pil,
            ip_adapter_image=garment_resized,
            num_inference_steps=args.inference_steps,
            guidance_scale=7.5,
            strength=0.99,
            generator=generator,
        )
        if control_image is not None:
            pipe_kwargs["control_image"] = control_image
            pipe_kwargs["controlnet_conditioning_scale"] = args.controlnet_scale
            print(f"[AI] ControlNet conditioning scale: {args.controlnet_scale}")

        result_resized = pipe(**pipe_kwargs).images[0]

        print("[AI] Synthesis complete. Saving debug output...")
        result_resized.save("debug_sd_output.png")
    except Exception as e:
        print(f"[AI] [ERROR] Inference failed: {e}")
        return

    # 8. Blend at output resolution
    max_limit = 1600
    if orig_w > max_limit or orig_h > max_limit:
        scale = max_limit / max(orig_w, orig_h)
        out_w = int(orig_w * scale)
        out_h = int(orig_h * scale)
    else:
        out_w, out_h = orig_w, orig_h

    print(f"[AI] Output size: {out_w}x{out_h}")

    person_out = person_img.resize((out_w, out_h), Image.Resampling.LANCZOS)
    result_out = result_resized.resize((out_w, out_h), Image.Resampling.LANCZOS)

    # Resize masks to output size
    mask_out = cv2.resize(mask_blend, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mask_out = cv2.GaussianBlur(mask_out, (5, 5), 0)
    face_out = cv2.resize(face_mask, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # 9. Composite with deepfake-style preservation
    final_img = blend_with_face_and_bg_preservation(
        person_out, result_out, mask_out, face_out, edge_band=20,
        use_pose_enhancements=use_pose_enhancements, lm=lm,
        seg_model=seg_model, seg_processor=seg_processor, device=device,
        garment_type=args.garment_type, pred_seg=pred_seg)

    # 10. Optional upscale for sharper clothing texture
    if args.upscale:
        final_img = optional_upscale(final_img)

    # 11. Save
    out_path = args.output_path
    if out_path.lower().endswith(".jpg") or out_path.lower().endswith(".jpeg"):
        final_img.save(out_path, format="JPEG", quality=92)
    else:
        final_img.save(out_path)

    print(f"[AI] SUCCESS: Saved final image to: {out_path}")


if __name__ == "__main__":
    main()
