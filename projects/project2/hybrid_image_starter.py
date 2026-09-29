import math
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as sktr

from align_image_code import align_images, match_img_size, rescale_images


PROJECT_DIR = Path(__file__).resolve().parent
MAX_IMAGE_SIZE = 700
INTERACTIVE_ALIGNMENT = os.environ.get("HYBRID_INTERACTIVE_ALIGNMENT", "1") == "1"


def load_image(filename):
    image = plt.imread(PROJECT_DIR / filename).astype(float)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.max() > 1:
        image /= 255.0

    height, width = image.shape[:2]
    scale = min(1.0, MAX_IMAGE_SIZE / max(height, width))
    if scale < 1:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image


def relative_points(image, points):
    height, width = image.shape[:2]
    return tuple((x * width, y * height) for x, y in points)


def recenter_with_edge_padding(image, row, column):
    height, width = image.shape[:2]
    row_pad = int(abs(2 * row + 1 - height))
    column_pad = int(abs(2 * column + 1 - width))
    pad_width = [
        (0 if row > (height - 1) / 2 else row_pad,
         0 if row < (height - 1) / 2 else row_pad),
        (0 if column > (width - 1) / 2 else column_pad,
         0 if column < (width - 1) / 2 else column_pad),
    ]
    if image.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(image, pad_width, mode="edge")


def align_with_points(im1, im2, im1_points, im2_points):
    points = relative_points(im1, im1_points) + relative_points(im2, im2_points)
    p1, p2, p3, p4 = points
    center1 = np.round(np.mean([p1, p2], axis=0))
    center2 = np.round(np.mean([p3, p4], axis=0))
    im1 = recenter_with_edge_padding(im1, center1[1], center1[0])
    im2 = recenter_with_edge_padding(im2, center2[1], center2[0])
    im1, im2 = rescale_images(im1, im2, points)
    theta1 = math.atan2(-(p2[1] - p1[1]), p2[0] - p1[0])
    theta2 = math.atan2(-(p4[1] - p3[1]), p4[0] - p3[0])
    im1 = sktr.rotate(
        im1,
        (theta2 - theta1) * 180 / np.pi,
        mode="edge",
        preserve_range=True,
    )
    return match_img_size(im1, im2)


def convolve_two_for_loops(image, kernel):
    filter_height, filter_width = kernel.shape
    image_height, image_width = image.shape
    pad_h = filter_height // 2
    pad_w = filter_width // 2
    padded_image = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )
    output = np.zeros((image_height, image_width))
    kernel = np.flip(kernel)

    for i in range(image_height):
        for j in range(image_width):
            patch = padded_image[i:i + filter_height, j:j + filter_width]
            output[i, j] = np.sum(patch * kernel)
    return output


def blur_image(image, sigma):
    gaussian_1d = cv2.getGaussianKernel(9, sigma)
    gaussian_2d = gaussian_1d @ gaussian_1d.T
    return cv2.filter2D(
        image,
        ddepth=-1,
        kernel=gaussian_2d,
        borderType=cv2.BORDER_CONSTANT,
    )


def hybrid_from_aligned(high_image, low_image, sigma_high, sigma_low, alpha=1):
    blurred_high = blur_image(high_image, sigma_high)
    high_frequency = alpha * (high_image - blurred_high)
    low_frequency = blur_image(low_image, sigma_low)
    hybrid = np.clip(low_frequency + high_frequency, 0, 1)
    return hybrid, high_frequency, low_frequency


def hybrid_image(
    high_image,
    low_image,
    sigma_high,
    sigma_low,
    high_points,
    low_points,
    alpha=1,
    label="image pair",
):
    high_aligned, low_aligned = align_pair(
        high_image,
        low_image,
        high_points,
        low_points,
        label,
    )
    hybrid, high_frequency, low_frequency = hybrid_from_aligned(
        high_aligned,
        low_aligned,
        sigma_high,
        sigma_low,
        alpha,
    )
    return hybrid, high_aligned, low_aligned, high_frequency, low_frequency


def save_image(image, title, filename, signed=False):
    display_image = np.asarray(image)
    if signed:
        limit = max(float(np.max(np.abs(display_image))), 1e-8)
        display_image = np.clip(0.5 + display_image / (2 * limit), 0, 1)
    else:
        display_image = np.clip(display_image, 0, 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    if display_image.ndim == 2:
        ax.imshow(display_image, cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(display_image)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def grayscale(image):
    return np.mean(image, axis=2) if image.ndim == 3 else image


def fourier_magnitude(image):
    gray = grayscale(image)
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray))) + 1e-8)


def save_fourier(image, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(fourier_magnitude(image), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_cutoff_experiment(high_aligned, low_aligned, filename, title):
    settings = [(4, 6), (6, 8), (8, 10)]
    fig, axes = plt.subplots(1, len(settings), figsize=(15, 4))
    for ax, (sigma_high, sigma_low) in zip(axes, settings):
        hybrid, _, _ = hybrid_from_aligned(
            high_aligned,
            low_aligned,
            sigma_high,
            sigma_low,
        )
        ax.imshow(hybrid)
        ax.set_title(f"σhigh={sigma_high}, σlow={sigma_low}")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def align_pair(im1, im2, im1_points, im2_points, label="image pair"):
    if INTERACTIVE_ALIGNMENT:
        print(f"Aligning {label}.")
        print("Click two corresponding points in the first image, then the same two points in the second image.")
        return align_images(im1, im2)
    return align_with_points(im1, im2, im1_points, im2_points)


# The point pairs are normalized (x, y) coordinates for the two eyes in each image.
nutmeg = load_image("nutmeg.jpg")
derek = load_image("DerekPicture.jpg")
nutmeg_points = ((0.43, 0.28), (0.54, 0.34))
derek_points = ((0.38, 0.33), (0.60, 0.33))

nutmeg_derek = hybrid_image(
    nutmeg,
    derek,
    sigma_high=6,
    sigma_low=8,
    high_points=nutmeg_points,
    low_points=derek_points,
    label="Nutmeg and Derek",
)
nutmeg_derek_hybrid, nutmeg_aligned, derek_aligned, nutmeg_high, derek_low = nutmeg_derek
save_image(nutmeg_aligned, "Nutmeg aligned", "nutmeg_aligned.png")
save_image(derek_aligned, "Derek aligned", "derek_aligned.png")
save_image(nutmeg_high, "Nutmeg high-pass filtered", "nutmeg_high_pass.png", signed=True)
save_image(derek_low, "Derek low-pass filtered", "derek_low_pass.png")
save_image(nutmeg_derek_hybrid, "Nutmeg + Derek hybrid", "nutmeg_derek_hybrid.png")
save_fourier(nutmeg_aligned, "Nutmeg input Fourier magnitude", "nutmeg_input_fourier.png")
save_fourier(derek_aligned, "Derek input Fourier magnitude", "derek_input_fourier.png")
save_fourier(nutmeg_high, "Nutmeg high-pass Fourier magnitude", "nutmeg_high_fourier.png")
save_fourier(derek_low, "Derek low-pass Fourier magnitude", "derek_low_fourier.png")
save_fourier(nutmeg_derek_hybrid, "Nutmeg + Derek hybrid Fourier magnitude", "nutmeg_derek_hybrid_fourier.png")
save_cutoff_experiment(
    nutmeg_aligned,
    derek_aligned,
    "nutmeg_derek_cutoff_experiment.png",
    "Nutmeg/Derek cutoff experiment",
)


# Additional examples: only the originals and final hybrid are displayed on the website.
connor = load_image("connor_smiling.png")
snow = load_image("young_snow.jpeg")
connor_snow, _, _, _, _ = hybrid_image(
    connor,
    snow,
    sigma_high=4,
    sigma_low=6,
    high_points=((0.44, 0.45), (0.57, 0.44)),
    low_points=((0.40, 0.44), (0.62, 0.44)),
    label="Connor and Snow",
)
save_image(connor_snow, "Connor + Snow hybrid", "connor_snow_hybrid.png")

tiger = load_image("tiger.png")
lion = load_image("lion.png")
tiger_lion, _, _, _, _ = hybrid_image(
    tiger,
    lion,
    sigma_high=5,
    sigma_low=7,
    high_points=((0.38, 0.42), (0.61, 0.42)),
    low_points=((0.39, 0.50), (0.61, 0.50)),
    label="Tiger and lion",
)
save_image(tiger_lion, "Tiger + Lion hybrid", "tiger_lion_hybrid.png")

messi = load_image("messi.png")
messi_points = ((0.42, 0.40), (0.53, 0.40))
lion_points = ((0.39, 0.50), (0.61, 0.50))
messi_aligned, lion_aligned = align_pair(
    messi,
    lion,
    messi_points,
    lion_points,
    label="Messi and lion",
)
messi_lion, messi_high, lion_low = hybrid_from_aligned(
    messi_aligned,
    lion_aligned,
    sigma_high=6,
    sigma_low=8,
)
save_image(messi_aligned, "Messi aligned", "messi_aligned.png")
save_image(lion_aligned, "Lion aligned", "lion_aligned.png")
save_image(messi_high, "Messi high-pass filtered", "messi_high_pass.png", signed=True)
save_image(lion_low, "Lion low-pass filtered", "lion_low_pass.png")
save_image(messi_lion, "Messi + Lion hybrid", "messi_lion_hybrid.png")
save_fourier(messi_aligned, "Messi input Fourier magnitude", "messi_input_fourier.png")
save_fourier(lion_aligned, "Lion input Fourier magnitude", "lion_input_fourier.png")
save_fourier(messi_high, "Messi high-pass Fourier magnitude", "messi_high_fourier.png")
save_fourier(lion_low, "Lion low-pass Fourier magnitude", "lion_low_fourier.png")
save_fourier(messi_lion, "Messi + Lion hybrid Fourier magnitude", "messi_lion_hybrid_fourier.png")
save_cutoff_experiment(
    messi_aligned,
    lion_aligned,
    "messi_lion_cutoff_experiment.png",
    "Messi/Lion cutoff experiment",
)
