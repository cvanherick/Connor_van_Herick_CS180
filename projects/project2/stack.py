from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
STACK_LEVELS = 4
GAUSSIAN_SIGMA = 8


def load_image(filename):
    image = plt.imread(PROJECT_DIR / filename).astype(float)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.max() > 1:
        image /= 255.0
    return image


def convolve_two_for_loops(image, kernel):
    filter_height, filter_width = kernel.shape
    image_height, image_width = image.shape
    pad_h = filter_height // 2
    pad_w = filter_width // 2
    padded_image = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="reflect",
    )
    output = np.zeros((image_height, image_width))
    kernel = np.flip(kernel)

    for i in range(image_height):
        for j in range(image_width):
            patch = padded_image[i:i + filter_height, j:j + filter_width]
            output[i, j] = np.sum(patch * kernel)
    return output


def blur_image(image, sigma=GAUSSIAN_SIGMA):
    gaussian_1d = cv2.getGaussianKernel(9, sigma)
    gaussian_2d = gaussian_1d @ gaussian_1d.T
    return np.stack(
        [convolve_two_for_loops(image[:, :, c], gaussian_2d) for c in range(image.shape[2])],
        axis=2,
    )


def gaussian_stack(image, counter):
    """Build same-size Gaussian levels without downsampling."""
    levels = [image]
    for _ in range(counter):
        levels.append(blur_image(levels[-1]))
    return levels


def laplacian_stack(gaussian_levels):
    """Store each Gaussian level's detail plus the final low-frequency residual."""
    details = [
        gaussian_levels[i] - gaussian_levels[i + 1]
        for i in range(len(gaussian_levels) - 1)
    ]
    return details + [gaussian_levels[-1]]


def signed_display(image):
    limit = max(float(np.max(np.abs(image))), 1e-8)
    return np.clip(0.5 + image / (2 * limit), 0, 1)


def save_gaussian_stack(stacks, filename):
    fig, axes = plt.subplots(2, len(stacks[0][1]), figsize=(18, 7))
    for row, (name, levels) in enumerate(stacks):
        for level, image in enumerate(levels):
            axes[row, level].imshow(np.clip(image, 0, 1))
            axes[row, level].set_title(f"{name}, level {level}")
            axes[row, level].axis("off")
    fig.suptitle("Gaussian stacks (all levels keep the original dimensions)")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_laplacian_stack(stacks, filename):
    fig, axes = plt.subplots(2, len(stacks[0][1]), figsize=(18, 7))
    for row, (name, levels) in enumerate(stacks):
        for level, image in enumerate(levels):
            display = signed_display(image) if level < len(levels) - 1 else np.clip(image, 0, 1)
            axes[row, level].imshow(display)
            title = "low-frequency residual" if level == len(levels) - 1 else f"detail level {level}"
            axes[row, level].set_title(f"{name}, {title}")
            axes[row, level].axis("off")
    fig.suptitle("Laplacian stacks")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_oraple_blend(orange_laplacian, apple_laplacian, mask_stack, filename):
    fig, axes = plt.subplots(5, 3, figsize=(14, 18))
    blended_laplacian = []

    for level, (orange_level, apple_level) in enumerate(
        zip(orange_laplacian, apple_laplacian)
    ):
        mask_level = mask_stack[level]
        apple_contribution = mask_level * apple_level
        orange_contribution = (1 - mask_level) * orange_level
        combined_level = apple_contribution + orange_contribution
        blended_laplacian.append(combined_level)

        if level < len(orange_laplacian) - 1:
            scale = max(
                np.max(np.abs(orange_level)),
                np.max(np.abs(apple_level)),
                np.max(np.abs(combined_level)),
                1e-8,
            )
            display_images = [
                0.5 + orange_level / (2 * scale),
                0.5 + apple_level / (2 * scale),
                0.5 + combined_level / (2 * scale),
            ]
            row_title = f"Detail level {level}"
        else:
            display_images = [
                np.clip(apple_contribution, 0, 1),
                np.clip(orange_contribution, 0, 1),
                np.clip(combined_level, 0, 1),
            ]
            row_title = "Low-frequency residual"

        for column, image in enumerate(display_images):
            axes[level, column].imshow(np.clip(image, 0, 1))
            axes[level, column].axis("off")
        axes[level, 0].set_ylabel(row_title, rotation=90, size=11)

    axes[0, 0].set_title("Orange contribution")
    axes[0, 1].set_title("Apple contribution")
    axes[0, 2].set_title("Combined level")
    fig.suptitle("Oraple Laplacian-stack blend")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return blended_laplacian


def create_oraple_outputs():
    orange = load_image("orange.jpeg")
    apple = load_image("apple.jpeg")
    orange_gaussian = gaussian_stack(orange, STACK_LEVELS)
    apple_gaussian = gaussian_stack(apple, STACK_LEVELS)
    orange_laplacian = laplacian_stack(orange_gaussian)
    apple_laplacian = laplacian_stack(apple_gaussian)

    save_gaussian_stack(
        [("Orange", orange_gaussian), ("Apple", apple_gaussian)],
        "gaussian_stacks.png",
    )
    save_laplacian_stack(
        [("Orange", orange_laplacian), ("Apple", apple_laplacian)],
        "laplacian_stacks.png",
    )

    mask = np.zeros_like(orange)
    mask[:, :mask.shape[1] // 2, :] = 1
    mask_stack = gaussian_stack(mask, STACK_LEVELS)
    blended_laplacian = save_oraple_blend(
        orange_laplacian,
        apple_laplacian,
        mask_stack,
        "oraple_laplacian_blend.png",
    )
    oraple = np.clip(sum(blended_laplacian), 0, 1)
    plt.imsave(PROJECT_DIR / "oraple.png", oraple)


if __name__ == "__main__":
    create_oraple_outputs()
