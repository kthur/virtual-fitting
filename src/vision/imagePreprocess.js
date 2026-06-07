import * as ImageManipulator from 'expo-image-manipulator';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';

const MIN_SHORT_EDGE = 512;     // 짧은 변 최소 픽셀 (AI 합성 품질 보장)
const MAX_LONG_EDGE  = 1600;    // 긴 변 최대 픽셀 (전송 부담 방지)
const JPEG_QUALITY   = 0.82;

/**
 * Normalize a user photo for AI virtual try-on.
 *
 * Steps:
 *  1. EXIF orientation is auto-applied (rotate 0/90/180/270) — Expo image-picker gives
 *     raw URIs and the AI server needs upright pixels.
 *  2. Long edge clamped to MAX_LONG_EDGE to avoid 10MB payloads.
 *  3. JPEG re-encoded at JPEG_QUALITY for consistent backend handling.
 *  4. Returned as a data URI for the existing /api/tryon pipeline.
 *
 * @param {string} sourceUri   The file:// or content:// URI from ImagePicker.
 * @param {object} [options]
 * @param {number} [options.maxLongEdge=1600]
 * @returns {Promise<{ uri: string, width: number, height: number, base64: string, dataUri: string, quality: object }>}
 */
export async function preprocessUserPhoto(sourceUri, options = {}) {
  const maxLong = options.maxLongEdge || MAX_LONG_EDGE;

  const result = await manipulateAsync(
    sourceUri,
    [],
    {
      compress: JPEG_QUALITY,
      format: SaveFormat.JPEG,
      base64: true,
    }
  );

  let { uri, width, height, base64 } = result;
  const actions = [];

  // EXIF orientation is already applied by manipulateAsync. We still need to clamp size.
  const longEdge = Math.max(width, height);
  if (longEdge > maxLong) {
    const ratio = maxLong / longEdge;
    const targetW = Math.round(width  * ratio);
    const targetH = Math.round(height * ratio);
    actions.push({ resize: { width: targetW, height: targetH } });
  }

  let final = result;
  if (actions.length) {
    final = await manipulateAsync(uri, actions, {
      compress: JPEG_QUALITY,
      format: SaveFormat.JPEG,
      base64: true,
    });
    uri      = final.uri;
    width    = final.width;
    height   = final.height;
    base64   = final.base64;
  }

  if (!base64) {
    throw new Error('Image preprocessing failed: no base64 produced.');
  }

  const dataUri = `data:image/jpeg;base64,${base64}`;
  const quality = assessPhotoQuality(width, height);

  return { uri, width, height, base64, dataUri, quality };
}

/**
 * Lightweight client-side quality assessment (no TFLite).
 * Flags photos that are very likely to fail at the AI server.
 */
export function assessPhotoQuality(width, height) {
  const issues = [];
  const shortEdge = Math.min(width, height);

  if (shortEdge < MIN_SHORT_EDGE) {
    issues.push({
      code: 'low_resolution',
      message: `이미지가 너무 작습니다 (${width}×${height}). ${MIN_SHORT_EDGE}px 이상 권장.`,
      severity: 'warn',
    });
  }

  const aspect = width / height;
  if (aspect < 0.4 || aspect > 2.5) {
    issues.push({
      code: 'extreme_aspect',
      message: '이미지 비율이 극단적입니다. 정면 전신/상반신 사진이 가장 좋아요.',
      severity: 'info',
    });
  }

  return {
    width,
    height,
    shortEdge,
    aspect: Number(aspect.toFixed(3)),
    ok: !issues.some(i => i.severity === 'error'),
    issues,
  };
}

/**
 * Convert a remote image URL to a base64 data URI.
 * Used when we need to feed a scraped garment image into local processing
 * (e.g. for client-side background removal later).
 */
export async function urlToBase64DataUri(url, maxBytes = 8 * 1024 * 1024) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.status}`);
  const blob = await res.blob();
  if (blob.size > maxBytes) {
    throw new Error(`Image too large (${(blob.size / 1024 / 1024).toFixed(1)} MB)`);
  }
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('FileReader error'));
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}
