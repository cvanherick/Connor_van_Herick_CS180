from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from stack import (
    GAUSSIAN_SIGMA,
    STACK_LEVELS,
    gaussian_stack,
    laplacian_stack,
    load_image,
    signed_display,
)


PROJECT_DIR = Path(__file__).resolve().parent


def center_crop_square(image):
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top:top + side, left:left + side]


def prepare_image(filename, size=600):
    cropped = center_crop_square(load_image(filename))
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def vertical_mask(shape):
    mask = np.zeros(shape, dtype=float)
    mask[:, :mask.shape[1] // 2, :] = 1
    return mask


def semicircle_mask(shape, center=(0.5, 0.60), radius=(0.52, 0.54)):
    height, width = shape[:2]
    y, x = np.mgrid[:height, :width]
    cx = center[0] * width
    cy = center[1] * height
    rx = radius[0] * width
    ry = radius[1] * height
    upper_semicircle = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1
    upper_semicircle &= y <= cy
    return np.repeat(upper_semicircle[:, :, None].astype(float), 3, axis=2)


def petal_trace_mask(shape):
    """Fill the area above a traced, wavy boundary following the petal ring."""
    height, width = shape[:2]
    trace = np.array([
        (0.00, 0.50),
        (0.07, 0.53),
        (0.14, 0.47),
        (0.21, 0.53),
        (0.29, 0.47),
        (0.37, 0.52),
        (0.45, 0.47),
        (0.53, 0.52),
        (0.61, 0.47),
        (0.69, 0.53),
        (0.77, 0.47),
        (0.84, 0.53),
        (0.92, 0.47),
        (1.00, 0.50),
    ])
    boundary = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in trace]
    polygon = [(0, 0), (width - 1, 0)] + list(reversed(boundary))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 1)
    return np.repeat(mask[:, :, None].astype(float), 3, axis=2)


def scale_center_crop(image, scale):
    height, width = image.shape[:2]
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    top = (resized.shape[0] - height) // 2
    left = (resized.shape[1] - width) // 2
    return resized[top:top + height, left:left + width]


def remove_dark_background(foreground, background):
    """Keep bright explosion pixels and use the sunflower behind dark pixels."""
    brightness = np.max(foreground, axis=2)
    alpha = np.clip((brightness - 0.015) / 0.16, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 3)[:, :, None]
    return alpha * foreground + (1 - alpha) * background


def multiresolution_blend(left_image, right_image, mask):
    left_laplacian = laplacian_stack(gaussian_stack(left_image, STACK_LEVELS))
    right_laplacian = laplacian_stack(gaussian_stack(right_image, STACK_LEVELS))
    mask_gaussian = gaussian_stack(mask, STACK_LEVELS)

    blended_laplacian = [
        mask_level * left_level + (1 - mask_level) * right_level
        for left_level, right_level, mask_level in zip(
            left_laplacian,
            right_laplacian,
            mask_gaussian,
        )
    ]
    return blended_laplacian, np.clip(sum(blended_laplacian), 0, 1)


def save_pair(left_image, right_image, filename, left_title, right_title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, image, title in zip(
        axes,
        (left_image, right_image),
        (left_title, right_title),
    ):
        axis.imshow(np.clip(image, 0, 1))
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_mask(mask, filename):
    plt.imsave(PROJECT_DIR / filename, mask[:, :, 0], cmap="gray", vmin=0, vmax=1)


def save_blend_process(
    left_laplacian,
    right_laplacian,
    mask_stack,
    filename,
    left_title,
    right_title,
    figure_title,
):
    fig, axes = plt.subplots(len(left_laplacian), 3, figsize=(14, 18))
    for level, (left_level, right_level, mask_level) in enumerate(
        zip(left_laplacian, right_laplacian, mask_stack)
    ):
        combined = mask_level * left_level + (1 - mask_level) * right_level
        if level < len(left_laplacian) - 1:
            scale = max(
                np.max(np.abs(left_level)),
                np.max(np.abs(right_level)),
                np.max(np.abs(combined)),
                1e-8,
            )
            display_images = [
                0.5 + left_level / (2 * scale),
                0.5 + right_level / (2 * scale),
                0.5 + combined / (2 * scale),
            ]
            label = f"Detail level {level}"
        else:
            display_images = [
                np.clip(mask_level * left_level, 0, 1),
                np.clip((1 - mask_level) * right_level, 0, 1),
                np.clip(combined, 0, 1),
            ]
            label = "Low-frequency residual"

        for column, image in enumerate(display_images):
            axes[level, column].imshow(np.clip(image, 0, 1))
            axes[level, column].axis("off")
        axes[level, 0].set_ylabel(label, rotation=90, size=11)

    axes[0, 0].set_title(f"{left_title} contribution")
    axes[0, 1].set_title(f"{right_title} contribution")
    axes[0, 2].set_title("Combined level")
    fig.suptitle(figure_title)
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


lion = prepare_image("lion.png")
tiger = prepare_image("tiger.png")
mask = vertical_mask(lion.shape)
mask_stack = gaussian_stack(mask, STACK_LEVELS)
lion_laplacian = laplacian_stack(gaussian_stack(lion, STACK_LEVELS))
tiger_laplacian = laplacian_stack(gaussian_stack(tiger, STACK_LEVELS))
_, lion_tiger = multiresolution_blend(lion, tiger, mask)

save_pair(
    lion,
    tiger,
    "lion_tiger_prepared_inputs.png",
    "Prepared lion",
    "Prepared tiger",
)
save_mask(mask, "lion_tiger_vertical_mask.png")
save_blend_process(
    lion_laplacian,
    tiger_laplacian,
    mask_stack,
    "lion_tiger_laplacian_blend.png",
    "Lion",
    "Tiger",
    "Lion/Tiger multiresolution blend",
)
plt.imsave(PROJECT_DIR / "lion_tiger_blend.png", lion_tiger)


sunflower = prepare_image("sunflower.png")
explosion = prepare_image("explosion.png")
explosion = scale_center_crop(explosion, 1.30)
sunflower_mask = petal_trace_mask(sunflower.shape)
sunflower_mask_stack = gaussian_stack(sunflower_mask, STACK_LEVELS)
explosion_laplacian = laplacian_stack(gaussian_stack(explosion, STACK_LEVELS))
flower_laplacian = laplacian_stack(gaussian_stack(sunflower, STACK_LEVELS))
_, sunflower_explosion = multiresolution_blend(
    explosion,
    sunflower,
    sunflower_mask,
)

save_pair(
    sunflower,
    explosion,
    "sunflower_explosion_prepared_inputs.png",
    "Prepared sunflower",
    "Prepared explosion",
)
save_mask(sunflower_mask, "sunflower_explosion_petal_mask.png")
save_blend_process(
    explosion_laplacian,
    flower_laplacian,
    sunflower_mask_stack,
    "sunflower_explosion_laplacian_blend.png",
    "Explosion",
    "Sunflower",
    "Sunflower/explosion semicircle blend",
)
plt.imsave(PROJECT_DIR / "sunflower_explosion_blend.png", sunflower_explosion)
