import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers import StableDiffusionInpaintPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Local AI Virtual Try-On")
    parser.add_argument("--person_path", type=str, required=True, help="Path to user person image")
    parser.add_argument("--garment_path", type=str, required=True, help="Path to garment image")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output image")
    parser.add_argument("--prompt", type=str, default="a garment", help="Description of the garment")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"[AI] Initializing Local AI Try-On...")
    print(f"[AI] Person path: {args.person_path}")
    print(f"[AI] Garment path: {args.garment_path}")
    print(f"[AI] Output path: {args.output_path}")
    print(f"[AI] Prompt: {args.prompt}")
    
    # 1. Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AI] Using device: {device.upper()}")
    
    if device == "cuda":
        print(f"[AI] GPU Model: {torch.cuda.get_device_name(0)}")
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
    print(f"[AI] Original image size: {orig_w}x{orig_h}")
    
    # Resize for inference (512x768 is a standard portrait size for SD 1.5)
    target_w, target_h = 512, 768
    person_img_resized = person_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    garment_img_resized = garment_img.resize((512, 512), Image.Resampling.LANCZOS)

    # 3. Load Human Parsing Model (SegFormer)
    print("[AI] Loading Human Parsing Model (SegFormer)...")
    try:
        processor = SegformerImageProcessor.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
        segmentation_model = SegformerForSemanticSegmentation.from_pretrained("matei-dorian/segformer-b5-finetuned-human-parsing")
        segmentation_model = segmentation_model.to(device)
    except Exception as e:
        print(f"[AI] [ERROR] Failed to load human parser: {e}")
        return

    # 4. Generate Mask
    print("[AI] Segmenting person image...")
    try:
        inputs = processor(images=person_img_resized, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = segmentation_model(**inputs)
            logits = outputs.logits
            
        # Upsample logits to match resized image size
        upsampled_logits = F.interpolate(
            logits,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Determine garment type based on prompt
        is_lower = False
        is_full = False
        
        prompt_lower = args.prompt.lower()
        lower_keywords = ["바지", "팬츠", "데님", "스커트", "치마", "슬랙스", "pants", "skirt", "trouser", "jeans", "denim", "slacks", "trousers"]
        full_keywords = ["원피스", "드레스", "jumpsuit", "dress", "onepiece", "one-piece", "점프수트"]
        
        if any(kw in prompt_lower for kw in lower_keywords):
            is_lower = True
            print("[AI] Detected garment type: LOWER body (pants/skirt)")
        elif any(kw in prompt_lower for kw in full_keywords):
            is_full = True
            print("[AI] Detected garment type: FULL body (dress/jumpsuit)")
        else:
            print("[AI] Detected garment type: UPPER body (shirt/outer/sweater)")
            
        # Mapping for LIP labels:
        # 5: UpperClothes, 6: Dress, 7: Coat, 9: Pants, 10: Jumpsuits, 11: Scarf, 12: Skirt
        if is_lower:
            mask_labels = [9, 12]
        elif is_full:
            mask_labels = [5, 6, 9, 10, 12]
        else:
            mask_labels = [5, 7, 11]
            
        # Create binary mask
        mask_np = np.zeros(pred_seg.shape, dtype=np.uint8)
        for label in mask_labels:
            mask_np[pred_seg == label] = 255
        
        # Dilate mask slightly to handle baggy clothes or pose adjustments
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_np, kernel, iterations=1)
        
        # Blur the mask edges for smooth blending
        mask_blurred = cv2.GaussianBlur(mask_dilated, (15, 15), 0)
        
        mask_image = Image.fromarray(mask_blurred.astype(np.uint8))
        print("[AI] Mask generated successfully.")
    except Exception as e:
        print(f"[AI] [ERROR] Segmentation/Masking failed: {e}")
        return

    # 5. Load Stable Diffusion Inpainting + IP-Adapter
    print("[AI] Loading Stable Diffusion Inpainting model & IP-Adapter...")
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch_dtype,
            safety_checker=None
        )
        
        print("[AI] Loading IP-Adapter weights...")
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        pipe.set_ip_adapter_scale(0.75)
        
        if device == "cuda":
            print("[AI] Applying GPU memory offloading optimizations...")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")
            
    except Exception as e:
        print(f"[AI] [ERROR] Failed to load SD pipeline: {e}")
        return

    # 6. Run Inference
    print("[AI] Running try-on synthesis...")
    try:
        generator = torch.Generator(device=device).manual_seed(42) if device == "cuda" else torch.Generator().manual_seed(42)
        
        # Enrich the prompt with quality terms
        full_prompt = f"a person wearing {args.prompt}, high quality, realistic, photorealistic, 4k"
        negative_prompt = "deformed, bad quality, blurry, low resolution, ugly, distorted, unrealistic, extra limbs, bad anatomy, cartoon, drawing"
        
        result_img_resized = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=person_img_resized,
            mask_image=mask_image,
            ip_adapter_image=garment_img_resized,
            num_inference_steps=25,
            guidance_scale=7.0,
            generator=generator
        ).images[0]
        
        # Resize back to original dimensions
        print(f"[AI] Resizing result back to original size: {orig_w}x{orig_h}...")
        result_img = result_img_resized.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        
        # Save output
        result_img.save(args.output_path)
        print(f"[AI] Synthesis finished! Saved output to: {args.output_path}")
        
    except Exception as e:
        print(f"[AI] [ERROR] Try-On inference failed: {e}")
        return

if __name__ == "__main__":
    main()
