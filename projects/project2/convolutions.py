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
image = np.array(
    Image.open(
        "/Users/revivedbonsai63/Desktop/CS 180/CS180_Website/projects/project2/color_to_greyscale.jpeg"
    ).convert("L")
).astype(float)
camera_image = image = np.array(
    Image.open(
       "/Users/revivedbonsai63/Desktop/CS 180/CS180_Website/projects/project2/cameraman.png"
    ).convert("L")
).astype(float)
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
blurred = convolve_two_for_loops(image, box_filter)
dx_image = convolve_two_for_loops(image, Dx)
dy_image = convolve_two_for_loops(image, Dy)
scipy_blurred = convolve2d(
    image,
    box_filter,
    mode="same",
    boundary="fill",
    fillvalue=0
)

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
comparison_images = [image, blurred, scipy_blurred, dx_image, dy_image]
comparison_titles = ["Input", "2-Loop Box Filter", "SciPy Box Filter", "Dx", "Dy"]
for ax, result, title in zip(axes, comparison_images, comparison_titles):
    ax.imshow(result, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("/Users/revivedbonsai63/Desktop/CS 180/CS180_Website/projects/project2/convolution_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(np.allclose(blurred, scipy_blurred))
dx_camera = convolve_two_for_loops(camera_image, Dx)
dy_camera = convolve_two_for_loops(camera_image, Dy)
plt.imshow(dx_image, cmap="gray")
plt.title("Partial Derivative in X")
plt.axis("off")
plt.savefig("/Users/revivedbonsai63/Desktop/CS 180/CS180_Website/projects/project2/partial_derivative_x.png", dpi=180, bbox_inches="tight")
plt.show()

plt.imshow(dy_image, cmap="gray")
plt.title("Partial Derivative in Y")
plt.axis("off")
plt.savefig("/Users/revivedbonsai63/Desktop/CS 180/CS180_Website/projects/project2/partial_derivative_y.png", dpi=180, bbox_inches="tight")
plt.show()
gradient_mag_camera = np.sqrt(dx_camera ** 2 + dy_camera**2) 
plt.imshow(gradient_mag_camera, cmap="gray")
plt.title("Gradient Magnitude")
plt.axis("off")
plt.savefig("/Users/revivedbonsai63/Desktop/CS 180/CS180_Website/projects/project2/gradient_magnitude.png", dpi=180, bbox_inches="tight")
plt.show()
binary_edge = gradient_mag_camera > 65
plt.imshow(binary_edge, cmap="gray")
plt.title("Binary Edge")
plt.axis("off")
plt.savefig("/Users/revivedbonsai63/Desktop/CS 180/CS180_Website/projects/project2/binary_edges.png", dpi=180, bbox_inches="tight")
plt.show()
G_1d = cv2.getGaussianKernel(9, 2)
G_2d = G_1d @  G_1d.T
camera_blurred = convolve_two_for_loops(camera_image, G_2d)
plt.imshow(camera_blurred, cmap="gray")
plt.title("Blurred")
plt.axis("off")
plt.savefig("/Users/revivedbonsai63/Desktop/CS 180/CS180_Website/projects/project2/gaussian_blurred.png", dpi=180, bbox_inches="tight")
plt.show()
dx_G_camera = convolve_two_for_loops(camera_blurred, Dx)
dy_G_camera = convolve_two_for_loops(camera_blurred, Dy)
blurred_binary_edge = np.sqrt(dx_G_camera ** 2 + dy_G_camera**2) > 20
plt.imshow(blurred_binary_edge, cmap="gray")
plt.title("Blurred_binary edge")
plt.axis("off")
plt.savefig("/Users/revivedbonsai63/Desktop/CS 180/CS180_Website/projects/project2/gaussian_binary_edges.png", dpi=180, bbox_inches="tight")
plt.show()
