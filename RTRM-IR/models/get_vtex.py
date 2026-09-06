import cv2
import numpy as np
from PIL import Image


def _to_gray_uint8(image):
    if isinstance(image, Image.Image):
        gray = np.array(image.convert('L'))
    else:
        gray = np.asarray(image)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _normalize_detail_to_u8(detail, low=1.0, high=99.0):
    lo, hi = np.percentile(detail, (float(low), float(high)))
    if hi <= lo:
        return np.full(detail.shape, 127, dtype=np.uint8)
    detail = np.clip(detail, lo, hi)
    detail = (detail - lo) / (hi - lo + 1e-6)
    return np.clip(detail * 255.0, 0, 255).astype(np.uint8)


def _fixed_gabor_wavelet_texture(image, blur_sigma=3.0, raw_keep=0.35, clahe_clip=1.8, fine_boost=0.65, use_clahe=True):
    """Fixed Gabor + Haar-wavelet texture preprocessing.

    This moves the non-learned texture extraction before the model so the
    texture branch does not stack a second Gabor/wavelet frontend internally.
    """
    gray = _to_gray_uint8(image)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        local = clahe.apply(gray)
    else:
        local = gray
    local_f = local.astype(np.float32)

    sigma_bg = max(1.2, float(blur_sigma) * 2.0)
    background = cv2.GaussianBlur(local_f, (0, 0), sigmaX=sigma_bg, sigmaY=sigma_bg)
    detail = local_f - background

    gabor_responses = []
    for lambd in (4.0, 8.0):
        sigma = 0.56 * lambd
        ksize = 2 * int(1.5 * lambd) + 1
        for orient_idx in range(6):
            theta = orient_idx * np.pi / 6.0
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=sigma,
                theta=theta,
                lambd=lambd,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            kernel = kernel - kernel.mean()
            kernel = kernel / (np.abs(kernel).sum() + 1e-6)
            response = cv2.filter2D(detail, cv2.CV_32F, kernel)
            gabor_responses.append(np.abs(response))
    gabor_detail = np.mean(gabor_responses, axis=0) if gabor_responses else np.abs(detail)

    wavelet_levels = []
    current = local_f
    for _ in range(2):
        low = cv2.pyrDown(current)
        up = cv2.pyrUp(low, dstsize=(current.shape[1], current.shape[0]))
        wavelet_levels.append(np.abs(current - up))
        current = low
    wavelet_detail = wavelet_levels[0]
    if len(wavelet_levels) > 1:
        wavelet_detail = wavelet_detail + cv2.resize(
            wavelet_levels[1],
            (wavelet_detail.shape[1], wavelet_detail.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    fine_boost = float(np.clip(fine_boost, 0.0, 1.0))
    texture = 0.45 * np.abs(detail) + 0.35 * gabor_detail + 0.20 * wavelet_detail
    texture = fine_boost * texture + (1.0 - fine_boost) * np.abs(detail)
    detail_u8 = _normalize_detail_to_u8(texture, low=0.2, high=99.8)

    raw_keep = float(np.clip(raw_keep, 0.0, 1.0))
    texture_keep = 1.0 - raw_keep
    blended = raw_keep * gray.astype(np.float32) + texture_keep * detail_u8.astype(np.float32)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode='L')


def _legacy_gaussian_texture(image, blur_sigma=3.0, raw_keep=0.35, clahe_clip=1.8, fine_boost=0.65, use_clahe=True):
    """Original Gaussian-difference texture preprocessing used for IR."""
    gray = _to_gray_uint8(image)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        local = clahe.apply(gray)
    else:
        local = gray

    local_f = local.astype(np.float32)
    sigma_mid = max(1.0, float(blur_sigma))
    sigma_fine = max(0.45, sigma_mid * 0.35)
    sigma_coarse = max(sigma_mid * 1.8, sigma_mid + 1.0)

    fine_base = cv2.GaussianBlur(local_f, (0, 0), sigmaX=sigma_fine, sigmaY=sigma_fine)
    mid_base = cv2.GaussianBlur(local_f, (0, 0), sigmaX=sigma_mid, sigmaY=sigma_mid)
    coarse_base = cv2.GaussianBlur(local_f, (0, 0), sigmaX=sigma_coarse, sigmaY=sigma_coarse)

    fine_detail = local_f - fine_base
    mid_detail = fine_base - mid_base
    coarse_detail = mid_base - coarse_base

    fine_boost = float(np.clip(fine_boost, 0.0, 1.0))
    mid_weight = 0.25 * (1.0 - fine_boost)
    detail = fine_boost * fine_detail + (1.0 - fine_boost - mid_weight) * mid_detail + mid_weight * coarse_detail
    detail_u8 = _normalize_detail_to_u8(detail, low=0.1, high=99.9)

    raw_keep = float(np.clip(raw_keep, 0.0, 1.0))
    texture_keep = 1.0 - raw_keep
    blended = raw_keep * gray.astype(np.float32) + texture_keep * detail_u8.astype(np.float32)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode='L')


def process_image(image, *args, blur_sigma=3.0, raw_keep=0.35, clahe_clip=1.8, fine_boost=0.65, mode='gabor_wavelet', use_clahe=True):
    """Texture-oriented preprocessing used by both train and test datasets.

    Existing call signatures are kept for compatibility:
    - process_image(image, base_name, kernel_size, sigma, frequency, theta)
    - process_image(image, kernel_size, sigma, frequency, theta)
    """
    if len(args) not in (4, 5):
        raise TypeError('process_image expects 5 or 6 positional args including image')
    mode = str(mode).lower()
    if mode in ('legacy', 'gaussian', 'ir_legacy'):
        return _legacy_gaussian_texture(
            image,
            blur_sigma=blur_sigma,
            raw_keep=raw_keep,
            clahe_clip=clahe_clip,
            fine_boost=fine_boost,
            use_clahe=use_clahe,
        )
    return _fixed_gabor_wavelet_texture(
        image,
        blur_sigma=blur_sigma,
        raw_keep=raw_keep,
        clahe_clip=clahe_clip,
        fine_boost=fine_boost,
        use_clahe=use_clahe,
    )
