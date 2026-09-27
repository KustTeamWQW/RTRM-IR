from PIL import Image
import numpy as np
import cv2


def to_pseudo_rgb(image):
    if image.mode != 'L':
        image = image.convert('L')
    gray_array = np.array(image)
    rgb_array = np.dstack((gray_array, gray_array, gray_array))
    return Image.fromarray(rgb_array.astype('uint8'), 'RGB')


def blend_with_raw(raw_img, proc_img, mix):
    mix = float(np.clip(mix, 0.0, 1.0))
    raw_np = np.array(raw_img.convert('L'), dtype=np.float32)
    proc_np = np.array(proc_img.convert('L'), dtype=np.float32)

    # Keep compatibility with existing option values (e.g. 0.01), but make
    # preprocessing truly effective: low mix now means less raw retention.
    raw_weight = mix
    proc_weight = 1.0 - mix
    blended = raw_weight * raw_np + proc_weight * proc_np

    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended, mode='L')


def _retinex_base_texture(
    gray,
    illumination_sigma=80.0,
    low_percentile=1.0,
    high_percentile=99.0,
):
    gray_u8 = np.asarray(gray, dtype=np.uint8)
    gray_f = gray_u8.astype(np.float32) + 1.0

    # Estimate low-frequency illumination / thermal gradient.
    illumination = cv2.GaussianBlur(gray_f, (0, 0), float(illumination_sigma))
    illumination = np.maximum(illumination, 1e-3)

    # Retinex in log-domain to suppress large-scale intensity bias.
    log_reflectance = np.log(gray_f) - np.log(illumination)

    p_low, p_high = np.percentile(log_reflectance, (float(low_percentile), float(high_percentile)))
    if p_high <= p_low:
        return gray_u8

    clipped = np.clip(log_reflectance, p_low, p_high)
    base_texture = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(base_texture, 0, 255).astype(np.uint8)


def _enhance_texture_directly_gray(
    image,
    illumination_sigma=80.0,
    sigma_s=10,
    sigma_r=0.15,
    usm_sigma=2.0,
    usm_alpha=1.5,
    usm_beta=-0.5,
    clahe_clip=2.0,
    clahe_tile=(8, 8),
):
    image_l = image.convert('L')
    gray = np.array(image_l, dtype=np.uint8)

    # 1) Remove low-frequency illumination first, then enhance true texture.
    base_texture = _retinex_base_texture(gray, illumination_sigma=float(illumination_sigma))

    # detailEnhance requires a 3-channel image.
    bgr = cv2.cvtColor(base_texture, cv2.COLOR_GRAY2BGR)
    detail_enhanced = cv2.detailEnhance(bgr, sigma_s=float(sigma_s), sigma_r=float(sigma_r))

    # Unsharp Mask sharpening.
    gaussian_blur = cv2.GaussianBlur(detail_enhanced, (0, 0), float(usm_sigma))
    sharpened = cv2.addWeighted(detail_enhanced, float(usm_alpha), gaussian_blur, float(usm_beta), 0)

    gray_sharpened = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_tile))
    enhanced = clahe.apply(gray_sharpened)
    return Image.fromarray(enhanced)


def enhance_infrared_with_gabor(image, mix=0.12, ksize=5, sigma=4.0, lambd=8.0, gamma=0.5):
    # Keep signature stable for existing call sites.
    # The old gabor params are mapped into a robust Retinex-based texture pipeline.
    del ksize, sigma, lambd, gamma
    enhanced = _enhance_texture_directly_gray(
        image,
        illumination_sigma=80.0,
        sigma_s=10,
        sigma_r=0.15,
        usm_sigma=2.0,
        usm_alpha=1.5,
        usm_beta=-0.5,
        clahe_clip=2.0,
        clahe_tile=(8, 8),
    )
    return blend_with_raw(image.convert('L'), enhanced, mix)


def enhance_visible_with_sobel(image, mix=0.10):
    enhanced = _enhance_texture_directly_gray(
        image,
        illumination_sigma=80.0,
        sigma_s=10,
        sigma_r=0.15,
        usm_sigma=2.0,
        usm_alpha=1.5,
        usm_beta=-0.5,
        clahe_clip=2.0,
        clahe_tile=(8, 8),
    )
    return blend_with_raw(image.convert('L'), enhanced, mix)
