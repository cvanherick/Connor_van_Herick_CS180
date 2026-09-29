"""First, let's recap what a convolution is. Implement it with four for loops, then two for loops. 
Implement padding, with zero fill values (implement "same" or "full" padding); 
convolution without padding will receive partial credit. Your implementation should flip the filter.
Compare it with a built-in convolution function scipy.signal.convolve2d! Then, take a picture of yourself 
(and read it as grayscale), write out a 9x9 box filter, and convolve the picture with the box filter. 
Do it with the finite difference operators Dx and Dy as well. Include the code snippets in the website!

What can you use for this section? This section is meant to be done with numpy only, simple array operations."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from PIL import Image
import cv2
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
personal_image = np.array(
    Image.open(
        PROJECT_DIR / "color_to_greyscale.jpeg"
    ).convert("L")
).astype(float)
camera_image = np.array(
    Image.open(
       PROJECT_DIR / "cameraman.png"
    ).convert("L")
).astype(float)


def save_grayscale(image, title, filename, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(PROJECT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def convolve_four_for_loops(image, kernel):
    image_height, image_width = image.shape
    filter_height, filter_width = kernel.shape
    pad_h = filter_height // 2
    pad_w = filter_width // 2
    padded_image = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant"
    )


    output = np.zeros((image_height, image_width))

    # convolution flips the kernel
    kernel = np.flip(kernel)

    for i in range(image_height):
        for j in range(image_width):

            total = 0

            for u in range(filter_height):
                for v in range(filter_width):
                    total += padded_image[i + u, j + v] * kernel[u, v]

            output[i, j] = total

    return output
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
box_filter = np.ones((9, 9)) / 81


Dx = np.array([[1, 0, -1]])

Dy = np.array([
    [1], [0],
    [-1]
])
blurred = convolve_two_for_loops(camera_image, box_filter)
dx_image = convolve_two_for_loops(camera_image, Dx)
dy_image = convolve_two_for_loops(camera_image, Dy)
scipy_blurred = convolve2d(
    camera_image,
    box_filter,
    mode="same",
    boundary="fill",
    fillvalue=0
)

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
comparison_images = [camera_image, blurred, scipy_blurred, dx_image, dy_image]
comparison_titles = ["Input", "2-Loop Box Filter", "SciPy Box Filter", "Dx", "Dy"]
for ax, result, title in zip(axes, comparison_images, comparison_titles):
    ax.imshow(result, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "convolution_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(np.allclose(blurred, scipy_blurred))

personal_box_filtered = convolve_two_for_loops(personal_image, box_filter)
personal_dx = convolve_two_for_loops(personal_image, Dx)
personal_dy = convolve_two_for_loops(personal_image, Dy)
save_grayscale(personal_image, "Personal Grayscale Input", "personal_grayscale.png")
save_grayscale(personal_box_filtered, "Personal Photo: 9x9 Box Filter", "personal_box_filter.png")
save_grayscale(personal_dx, "Personal Photo: Dx", "personal_dx.png")
save_grayscale(personal_dy, "Personal Photo: Dy", "personal_dy.png")

dx_camera = convolve_two_for_loops(camera_image, Dx)
dy_camera = convolve_two_for_loops(camera_image, Dy)

FINITE_DIFFERENCE_THRESHOLD = 65
GAUSSIAN_THRESHOLD = 20

save_grayscale(dx_camera, "Partial Derivative in X", "partial_derivative_x.png")
save_grayscale(dy_camera, "Partial Derivative in Y", "partial_derivative_y.png")

gradient_mag_camera = np.hypot(dx_camera, dy_camera)
save_grayscale(gradient_mag_camera, "Gradient Magnitude", "gradient_magnitude.png")

binary_edge = gradient_mag_camera > FINITE_DIFFERENCE_THRESHOLD
save_grayscale(binary_edge, "Binary Edge (threshold = 65)", "binary_edges.png", vmin=0, vmax=1)

G_1d = cv2.getGaussianKernel(9, 2)
G_2d = G_1d @ G_1d.T
camera_blurred = convolve_two_for_loops(camera_image, G_2d)
save_grayscale(camera_blurred, "Gaussian-blurred Cameraman", "gaussian_blurred.png")

# Apply Dx and Dy after Gaussian smoothing, as in the finite-difference method.
dx_G_camera = convolve_two_for_loops(camera_blurred, Dx)
dy_G_camera = convolve_two_for_loops(camera_blurred, Dy)
gaussian_gradient_mag = np.hypot(dx_G_camera, dy_G_camera)
gaussian_binary_edge = gaussian_gradient_mag > GAUSSIAN_THRESHOLD
save_grayscale(
    gaussian_gradient_mag,
    "Gradient Magnitude after Gaussian Smoothing",
    "gaussian_gradient_magnitude.png",
)
save_grayscale(
    gaussian_binary_edge,
    "Gaussian-smoothed Binary Edge (threshold = 20)",
    "gaussian_binary_edges.png",
    vmin=0,
    vmax=1,
)

# Build full derivative-of-Gaussian filters and apply each with one convolution.
# The full mode preserves the complete 9x9 * 1x3 and 9x9 * 3x1 supports.
DoG_x = convolve2d(G_2d, Dx, mode="full", boundary="fill", fillvalue=0)
DoG_y = convolve2d(G_2d, Dy, mode="full", boundary="fill", fillvalue=0)
save_grayscale(DoG_x, "Derivative of Gaussian: DoG x", "dog_x_filter.png")
save_grayscale(DoG_y, "Derivative of Gaussian: DoG y", "dog_y_filter.png")

dog_dx_camera = convolve_two_for_loops(camera_image, DoG_x)
dog_dy_camera = convolve_two_for_loops(camera_image, DoG_y)
dog_gradient_mag = np.hypot(dog_dx_camera, dog_dy_camera)
dog_binary_edge = dog_gradient_mag > GAUSSIAN_THRESHOLD
save_grayscale(dog_gradient_mag, "DoG Gradient Magnitude", "dog_gradient_magnitude.png")
save_grayscale(
    dog_binary_edge,
    "DoG Binary Edge (threshold = 20)",
    "dog_binary_edges.png",
    vmin=0,
    vmax=1,
)

dog_difference = np.abs(gaussian_gradient_mag - dog_gradient_mag)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
comparison_images = [gaussian_gradient_mag, dog_gradient_mag, dog_difference]
comparison_titles = [
    "Gaussian then Dx/Dy",
    "Single-convolution DoG",
    "Absolute difference",
]
for ax, result, title in zip(axes, comparison_images, comparison_titles):
    ax.imshow(result, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig(PROJECT_DIR / "dog_verification.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("Finite-difference threshold:", FINITE_DIFFERENCE_THRESHOLD)
print("Gaussian/DoG threshold:", GAUSSIAN_THRESHOLD)
print("Maximum Gaussian-vs-DoG gradient difference:", dog_difference.max())
print("DoG matches the smoothed gradient away from the zero-padding boundary:", np.allclose(
    gaussian_gradient_mag[5:-5, 5:-5], dog_gradient_mag[5:-5, 5:-5], atol=1e-6
))
