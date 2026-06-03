import os
import sys
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline

# SegFormer LIP label map (from model config)
# 0: Background, 1: Hat, 2: Hair, 3: Sunglasses, 4: Upper-clothes
# 5: Skirt, 6: Pants, 7: Dress, 8: Belt, 9: Left-shoe
# 10: Right-shoe, 11: Face, 12: Left-leg, 13: Right-leg
# 14: Left-arm, 15: Right-arm, 16: Bag, 17: Scarf

PRESERVE_LABELS = {2, 9, 10, 11, 12, 13, 14, 15}  # Hair, shoes, face, legs, arms/hands => NEVER mask

def parse_args():
    # Force UTF-8 output so emoji/unicode doesn't crash on Windows cp949 terminals
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Local AI Virtual Try-On")
    parser.add_argument("--person_path", type=str, required=True)
    parser.add_argument("--garment_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a garment")
    parser.add_argument("--garment_type", type=str, default=None)
    parser.add_argument("--fit_type", type=str, default="regular")
    # Feature hints from shopping mall
    parser.add_argument("--colors", type=str, default="")
    parser.add_argument("--materials", type=str, default="")
    parser.add_argument("--styles", type=str, default="")
    return parser.parse_args()


def remove_background(img_pil):
    """Remove background from garment image using rembg."""
    try:
        from rembg import remove as rembg_remove
        print("[AI] Removing garment background with rembg...")
        img_rgba = rembg_remove(img_pil.convert("RGBA"))
        # Composite on white background for SD input
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
    """Build a rich, specific prompt from extracted features."""
    # Fit modifier
    fit_map = {
        "slim":    "slim fit, fitted, form-fitting, tailored silhouette",
        "overfit": "oversized, loose, baggy, oversized fit",
        "regular": "regular fit, standard fit",
    }
    fit_text = fit_map.get(fit_type, "regular fit")

    # Feature hints
    feature_parts = []
    if colors:
        feature_parts.append(colors)
    if materials:
        feature_parts.append(f"{materials} fabric")
    if styles:
        feature_parts.append(styles)
    feature_str = ", ".join(feature_parts) if feature_parts else ""

    full = f"a person wearing {base_prompt}"
    if feature_str:
        full += f", {feature_str}"
    full += f", {fit_text}"
    full += ", high quality, photorealistic, 8k, detailed fabric texture, natural lighting, fashion photography"

    negative = (
        "deformed, bad anatomy, ugly, blurry, low quality, distorted, extra limbs, "
        "mutated hands, fused fingers, missing fingers, watermark, duplicate, "
        "bad proportions, extra arms, unnatural body, cartoon, drawing, painted"
    )
    return full, negative


def generate_clothing_mask(segmentation_model, processor, person_img_resized,
                           target_w, target_h, garment_type, device):
    """
    Generate a precise clothing mask.
    - Includes only clothing labels for the given garment_type
    - Explicitly EXCLUDES face, skin, arms, legs
    """
    inputs = processor(images=person_img_resized, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = segmentation_model(**inputs)
        logits = outputs.logits

    upsampled = F.interpolate(logits, size=(target_h, target_w),
                              mode="bilinear", align_corners=False)
    pred_seg = upsampled.argmax(dim=1)[0].cpu().numpy()

    # Determine which labels to INCLUDE in the mask
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

    # Build mask: clothing area = 255, everything else = 0
    mask_np = np.zeros(pred_seg.shape, dtype=np.uint8)
    for label in clothing_labels:
        mask_np[pred_seg == label] = 255

    # Explicitly ZERO OUT face, skin, limbs – they must NEVER be inpainted
    for label in PRESERVE_LABELS:
        mask_np[pred_seg == label] = 0

    # Dilate mask slightly to catch garment edges
    kernel = np.ones((10, 10), np.uint8)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)

    # Re-apply preserve: undo any dilation into skin areas
    for label in PRESERVE_LABELS:
        mask_dilated[pred_seg == label] = 0

    # Soft blur for smooth blending at edges
    mask_blurred = cv2.GaussianBlur(mask_dilated, (11, 11), 0)

    print(f"[AI] Mask coverage: {(mask_blurred > 0).sum()} / {mask_blurred.size} px")
    return mask_blurred, pred_seg


def blend_result_with_original(original_pil, result_pil, soft_mask_np, pred_seg):
    """
    Only apply the AI result in the clothing region.
    Face, skin, background remain exactly from the original image.
    This is the 'deepfake clothes' approach: person is untouched.
    """
    orig_np = np.array(original_pil).astype(np.float32)
    result_np = np.array(result_pil).astype(np.float32)

    # Normalize mask to [0, 1]
    alpha = soft_mask_np.astype(np.float32) / 255.0
    alpha_3d = np.stack([alpha, alpha, alpha], axis=-1)

    # Blend: result in clothing area, original everywhere else
    blended = orig_np * (1.0 - alpha_3d) + result_np * alpha_3d
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def main():
    args = parse_args()

    print(f"[AI] ===== Local Virtual Try-On (v2 - ClothingOnly) =====")
    print(f"[AI] Person:   {args.person_path}")
    print(f"[AI] Garment:  {args.garment_path}")
    print(f"[AI] Output:   {args.output_path}")
    print(f"[AI] Type: {args.garment_type} | Fit: {args.fit_type}")
    print(f"[AI] Features -> Colors: '{args.colors}' | Materials: '{args.materials}'")

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

    # Clamp to max 1024px for inference (balance quality vs speed)
    target_w, target_h = 512, 768

    # 3. Garment preprocessing: remove background for cleaner IP-Adapter signal
    garment_clean = remove_background(garment_img)
    garment_resized = garment_clean.resize((512, 512), Image.Resampling.LANCZOS)
    person_resized = person_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 4. Human parsing for clothing mask
    print("[AI] Loading SegFormer human parser...")
    try:
        seg_processor = SegformerImageProcessor.from_pretrained(
            "matei-dorian/segformer-b5-finetuned-human-parsing")
        seg_model = SegformerForSemanticSegmentation.from_pretrained(
            "matei-dorian/segformer-b5-finetuned-human-parsing").to(device)
    except Exception as e:
        print(f"[AI] [ERROR] Failed to load parser: {e}")
        return

    print("[AI] Generating clothing mask...")
    try:
        mask_np, pred_seg = generate_clothing_mask(
            seg_model, seg_processor, person_resized,
            target_w, target_h, args.garment_type, device)
        mask_image = Image.fromarray(mask_np)
    except Exception as e:
        print(f"[AI] [ERROR] Mask generation failed: {e}")
        return

    # 5. Build prompt from extracted features
    prompt, negative_prompt = build_prompt(
        args.prompt,
        garment_type=args.garment_type,
        fit_type=args.fit_type,
        colors=args.colors,
        materials=args.materials,
        styles=args.styles
    )
    print(f"[AI] Prompt: {prompt}")

    # 6. Load SD Inpainting + IP-Adapter
    print("[AI] Loading Stable Diffusion Inpainting + IP-Adapter...")
    try:
        from transformers import CLIPVisionModelWithProjection
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            torch_dtype=torch_dtype
        ).to(device)

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch_dtype,
            safety_checker=None,
            image_encoder=image_encoder
        )
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                             weight_name="ip-adapter_sd15.bin")
        # Higher scale = more faithful to garment image
        pipe.set_ip_adapter_scale(0.85)

        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")
    except Exception as e:
        print(f"[AI] [ERROR] Failed to load pipeline: {e}")
        return

    # 7. Inference
    print("[AI] Running clothing synthesis (inpainting)...")
    try:
        generator = torch.Generator(device=device).manual_seed(42)

        result_resized = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=person_resized,
            mask_image=mask_image,
            ip_adapter_image=garment_resized,
            num_inference_steps=25,
            guidance_scale=7.5,
            strength=0.99,
            generator=generator
        ).images[0]

        print("[AI] Synthesis complete. Blending with original...")
    except Exception as e:
        print(f"[AI] [ERROR] Inference failed: {e}")
        return

    # 8. CRITICAL: Blend only clothing region – restore face/skin from original
    # Scale result back to output resolution (clamped for performance)
    max_limit = 1200
    if orig_w > max_limit or orig_h > max_limit:
        scale = max_limit / max(orig_w, orig_h)
        out_w = int(orig_w * scale)
        out_h = int(orig_h * scale)
    else:
        out_w, out_h = orig_w, orig_h

    print(f"[AI] Output size: {out_w}x{out_h}")

    # Resize all components to output size
    person_out = person_img.resize((out_w, out_h), Image.Resampling.LANCZOS)
    result_out  = result_resized.resize((out_w, out_h), Image.Resampling.LANCZOS)
    # Resize mask to match (use the raw dilated mask, not blurred, to avoid leaking into face)
    mask_out = cv2.resize(mask_np, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    # Smooth edges of the resized mask
    mask_out = cv2.GaussianBlur(mask_out, (15, 15), 0)

    # Composite: original face/skin + AI clothing
    final_img = blend_result_with_original(person_out, result_out, mask_out, pred_seg)

    # 9. Save
    out_path = args.output_path
    if out_path.lower().endswith(".jpg") or out_path.lower().endswith(".jpeg"):
        final_img.save(out_path, format="JPEG", quality=90)
    else:
        final_img.save(out_path)

    print(f"[AI] SUCCESS: Saved final image to: {out_path}")


if __name__ == "__main__":
    main()
