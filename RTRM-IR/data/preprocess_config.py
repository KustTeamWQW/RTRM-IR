import numpy as np
from PIL import Image

from models.decompose import extract_radiation_component
from models.get_vtex import process_image


PREPROCESS_CACHE_VERSION = 'texture_residual_v10_consistent_ir_no_clahe_vis_clahe'
IR_TEXTURE_RAW_KEEP = 0.06
VISIBLE_TEXTURE_RAW_KEEP = 0.15
IR_TEXTURE_FINE_BOOST = 0.90
VISIBLE_TEXTURE_FINE_BOOST = 0.65


def to_pseudo_rgb(image):
    if image.mode != 'L':
        image = image.convert('L')
    gray_array = np.array(image)
    rgb_array = np.dstack((gray_array, gray_array, gray_array))
    return Image.fromarray(rgb_array.astype('uint8'), 'RGB')


def build_radiation_component(image):
    return extract_radiation_component(image)


def build_ir_texture(image):
    return process_image(
        image,
        (5, 5), 2.0, 0.15, np.pi / 12,
        raw_keep=IR_TEXTURE_RAW_KEEP,
        fine_boost=IR_TEXTURE_FINE_BOOST,
        mode='legacy',
        use_clahe=False
    )


def build_visible_texture(image):
    return process_image(
        image,
        (5, 5), 2.0, 0.15, np.pi / 12,
        raw_keep=VISIBLE_TEXTURE_RAW_KEEP,
        fine_boost=VISIBLE_TEXTURE_FINE_BOOST,
        mode='legacy',
        use_clahe=True
    )
