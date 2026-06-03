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

# Region that MUST never be inpainted (skin, face, hair, limbs).
# Keeping face/skin in this set ensures the deepfake illusion: the
# person is preserved at the pixel level; only clothing changes.
PRESERVE_LABELS = {2, 9, 10, 11, 12, 13, 14, 15}  # Hair, shoes, face, legs, arms/hands
BACKGROUND_LABEL = 0                                # Background - never mask


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


def build_face_protect_mask(person_pil, seg_model, seg_processor, device, w, h):
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

    # Face (11) + hair (2) + neck skin (sometimes mis-classed) + a small safety margin
    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_mask[pred == 11] = 255
    face_mask[pred == 2]  = 255
    # Dilate to add safety margin around face/hair
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    face_mask = cv2.dilate(face_mask, k, iterations=2)
    face_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
    return face_mask, pred


def generate_clothing_mask(seg_model, seg_processor, person_resized,
                           target_w, target_h, garment_type, device):
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
    for label in PRESERVE_LABELS:
        mask_np[pred_seg == label] = 0

    # Modest dilation to capture clothing edges
    kernel = np.ones((7, 7), np.uint8)
    mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)

    # Re-apply preserve: undo any dilation into skin areas
    for label in PRESERVE_LABELS:
        mask_dilated[pred_seg == label] = 0
    # Background must never be in the mask either
    mask_dilated[pred_seg == BACKGROUND_LABEL] = 0

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


def blend_with_face_and_bg_preservation(original_pil, result_pil, soft_mask_np,
                                        face_mask_np, edge_band=20):
    """
    Three-layer composite:
      1. face/hair region       -> 100% original
      2. background region      -> 100% original
      3. clothing region        -> soft blend (soft_mask)
      4. boundary band (clothing-skin) -> color-matched gradient blend
    """
    orig = np.array(original_pil).astype(np.float32)
    res  = np.array(result_pil).astype(np.float32)

    # 1. Compute color-match correction for boundary pixels
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

    # 4. Human parsing
    print("[AI] Loading SegFormer human parser...")
    try:
        seg_processor = SegformerImageProcessor.from_pretrained(
            "matei-dorian/segformer-b5-finetuned-human-parsing")
        seg_model = SegformerForSemanticSegmentation.from_pretrained(
            "matei-dorian/segformer-b5-finetuned-human-parsing").to(device)
    except Exception as e:
        print(f"[AI] [ERROR] Failed to load parser: {e}")
        return

    print("[AI] Generating clothing mask and face protect mask...")
    try:
        mask_dilated, mask_blurred, pred_seg = generate_clothing_mask(
            seg_model, seg_processor, person_resized,
            target_w, target_h, args.garment_type, device)
        mask_image = Image.fromarray(mask_blurred)

        face_mask, _ = build_face_protect_mask(
            person_resized, seg_model, seg_processor, device, target_w, target_h)

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
        pipe.set_ip_adapter_scale(args.ip_scale)

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
            mask_image=combined_mask_pil,
            ip_adapter_image=garment_resized,
            num_inference_steps=args.inference_steps,
            guidance_scale=7.5,
            strength=0.99,
            generator=generator
        ).images[0]

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
        person_out, result_out, mask_out, face_out, edge_band=20)

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
