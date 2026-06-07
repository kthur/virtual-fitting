import exifr from 'exifr';

const SENSOR_W = 36;
const SENSOR_H = 24;

const FRONT_CAM_DIST_CM = 70;
const REAR_CAM_DIST_CM = 150;

export async function parseCameraCalibration(photoUri) {
  try {
    const raw = await exifr.parse(photoUri, {
      tiff: true,
      xmp: false,
      iptc: false,
      ifd0: ['Make', 'Model', 'ImageWidth', 'ImageHeight', 'Orientation'],
      exif: ['FocalLength', 'FocalLengthIn35mmFilm', 'ExifImageWidth', 'ExifImageHeight', 'LensModel'],
    });
    if (!raw) return { available: false };

    const f35 = raw.FocalLengthIn35mmFilm || raw.FocalLength || null;
    let hfov = null, vfov = null;
    if (f35 != null && f35 > 0) {
      hfov = 2 * Math.atan2(SENSOR_W, 2 * f35);
      vfov = 2 * Math.atan2(SENSOR_H, 2 * f35);
    }

    return {
      available: true,
      focalLength35mm: f35,
      hfov,
      vfov,
      orientation: raw.Orientation || 1,
      deviceMake: raw.Make || null,
      deviceModel: raw.Model || null,
      lensModel: raw.LensModel || null,
      isWideAngle: f35 != null && f35 < 28,
      exifWidth: raw.ExifImageWidth || raw.ImageWidth || null,
      exifHeight: raw.ExifImageHeight || raw.ImageHeight || null,
    };
  } catch (e) {
    console.warn('EXIF parse failed:', e.message);
    return { available: false };
  }
}

export function computeCalibration(cal, imgW, imgH, personPxH, knownHeightCm, isFrontCam) {
  if (!cal.available || !cal.vfov || personPxH < 1) {
    return { method: 'simple_ratio', confidence: 'low', needsHeight: !knownHeightCm };
  }

  const assumeDist = isFrontCam ? FRONT_CAM_DIST_CM : REAR_CAM_DIST_CM;

  if (knownHeightCm > 0) {
    const angularH = cal.vfov * personPxH / imgH;
    if (Math.abs(Math.tan(angularH / 2)) < 1e-10) {
      return { method: 'simple_ratio', confidence: 'low', needsHeight: false };
    }
    const dist = knownHeightCm / (2 * Math.tan(angularH / 2));
    return {
      method: 'perspective',
      confidence: dist > 50 && dist < 500 ? 'high' : 'medium',
      distanceCm: Math.round(dist),
      focalLength35mm: cal.focalLength35mm,
      hfov: cal.hfov,
      vfov: cal.vfov,
      isWideAngle: cal.isWideAngle,
      deviceModel: cal.deviceModel,
    };
  }

  const estimatedHeight = 2 * assumeDist * Math.tan(cal.vfov * personPxH / (2 * imgH));
  return {
    method: 'estimated_from_distance',
    confidence: 'medium',
    assumedDistanceCm: assumeDist,
    estimatedHeightCm: Math.round(estimatedHeight),
    focalLength35mm: cal.focalLength35mm,
    hfov: cal.hfov,
    vfov: cal.vfov,
    isWideAngle: cal.isWideAngle,
    deviceModel: cal.deviceModel,
  };
}
