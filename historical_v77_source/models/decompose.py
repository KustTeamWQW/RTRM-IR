import cv2
import numpy as np
from PIL import Image


def _to_gray_array(image):
    if isinstance(image, Image.Image):
        arr = np.array(image)
    else:
        arr = image.copy()
    if len(arr.shape) > 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def extract_radiation_component(ir_image, D0=30):
    """Extract a low/mid-frequency radiation component from an IR image."""
    ir_np = _to_gray_array(ir_image)

    gray = np.float32(ir_np)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=12, sigmaSpace=5)

    sigma = max(float(D0) / 4.0, 0.8)
    ksize = int(max(5, round(sigma * 6 + 1)))
    if ksize % 2 == 0:
        ksize += 1

    low = cv2.GaussianBlur(denoised, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    mid = cv2.GaussianBlur(denoised, (0, 0), sigmaX=max(0.8, sigma * 0.55), sigmaY=max(0.8, sigma * 0.55))
    radiation = 0.55 * low + 0.30 * mid + 0.15 * denoised

    result_np = cv2.normalize(radiation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return Image.fromarray(result_np, 'L')


def _frequency_highpass(image, cutoff):
    gray = _to_gray_array(image)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    y, x = np.ogrid[:rows, :cols]
    mask = 1 - np.exp(-((x - ccol) ** 2 + (y - crow) ** 2) / (2 * cutoff ** 2))
    mask = np.repeat(mask[:, :, np.newaxis], 2, axis=2)

    filtered_shift = dft_shift * mask
    idft_shift = np.fft.ifftshift(filtered_shift)
    idft = cv2.idft(idft_shift)

    magnitude = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
    result_np = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return Image.fromarray(result_np, 'L')


def extract_texture_structure(ir_image, D1=20):
    """Extract a high-frequency texture structure map from an IR image."""
    return _frequency_highpass(ir_image, D1)


def extract_texture_detail(visible_image, D2=35):
    """Extract a high-frequency texture detail map from a visible image."""
    return _frequency_highpass(visible_image, D2)


def to_pseudo_rgb(image):
    """Convert a single-channel image to pseudo-RGB."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            rgb_array = np.dstack([image] * 3)
            return Image.fromarray(rgb_array.astype(np.uint8), 'RGB')
    elif isinstance(image, Image.Image):
        if image.mode == 'L':
            return Image.merge('RGB', [image] * 3)
    return image
