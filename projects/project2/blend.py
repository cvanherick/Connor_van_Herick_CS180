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


def save_pair(left_image, right_image, filename):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, image, title in zip(
        axes,
        (left_image, right_image),
        ("Prepared lion", "Prepared tiger"),
    ):
        axis.imshow(np.clip(image, 0, 1))
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_mask(mask, filename):
    plt.imsave(PROJECT_DIR / filename, mask[:, :, 0], cmap="gray", vmin=0, vmax=1)


def save_blend_process(left_laplacian, right_laplacian, mask_stack, filename):
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

    axes[0, 0].set_title("Lion contribution")
    axes[0, 1].set_title("Tiger contribution")
    axes[0, 2].set_title("Combined level")
    fig.suptitle("Lion/Tiger multiresolution blend")
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

save_pair(lion, tiger, "lion_tiger_prepared_inputs.png")
save_mask(mask, "lion_tiger_vertical_mask.png")
save_blend_process(
    lion_laplacian,
    tiger_laplacian,
    mask_stack,
    "lion_tiger_laplacian_blend.png",
)
plt.imsave(PROJECT_DIR / "lion_tiger_blend.png", lion_tiger)
