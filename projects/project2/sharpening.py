import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
climber_image = np.array(
    Image.open(
       PROJECT_DIR / "climbing.jpeg"
    ).convert("L")    
).astype(float)
tai_image = np.array(
    Image.open(
       PROJECT_DIR / "tai.png"
    ).convert("L")    
).astype(float)
 
def convolve_two_for_loops(image, kernel):
    filter_height, filter_width = kernel.shape
    image_height, image_width = image.shape
    
    pad_h = filter_height // 2
    pad_w = filter_width // 2

    padded_image = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant"
    )
    output = np.zeros((image_height, image_width))

    # Flip kernel for true convolution
    kernel = np.flip(kernel)
    for i in range(image_height):
            for j in range(image_width):
                patch = padded_image[
                    i:i + filter_height,
                    j:j + filter_width
                ]
    
                output[i, j] = np.sum(patch * kernel)
    
    return output
def unsharpen(alpha = 1):
    G_1d = cv2.getGaussianKernel(9, 2)
    G_2d = G_1d @ G_1d.T

    identity = np.zeros_like(G_2d)
    identity[4, 4] = 1

    unsharp_filter = (1 + alpha) * identity - alpha * G_2d
    return unsharp_filter


def save_grayscale(image, title, filename, vmin=0, vmax=255):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_sharpening_set(image, prefix, label):
    gaussian = cv2.getGaussianKernel(9, 2)
    gaussian_filter = gaussian @ gaussian.T
    blurred = convolve_two_for_loops(image, gaussian_filter)
    high_frequency = image - blurred
    high_frequency_limit = np.max(np.abs(high_frequency))

    save_grayscale(image, f"{label}: Original", f"{prefix}_original.png")
    save_grayscale(blurred, f"{label}: Gaussian Blur", f"{prefix}_blurred.png")
    save_grayscale(
        high_frequency,
        f"{label}: High-Frequency Residual",
        f"{prefix}_high_frequency.png",
        vmin=-high_frequency_limit,
        vmax=high_frequency_limit,
    )

    for alpha in (1, 2, 4):
        sharpened = np.clip(
            convolve_two_for_loops(image, unsharpen(alpha)),
            0,
            255,
        )
        save_grayscale(
            sharpened,
            f"{label}: Sharpened (alpha = {alpha})",
            f"{prefix}_sharpened_{alpha}.png",
        )


save_sharpening_set(tai_image, "taj", "Taj Mahal")
save_sharpening_set(climber_image, "climber", "Climber")
