# Project 2 Code

This folder contains the Python code and image assets for the Project 2 write-up at [`index.html`](index.html). The scripts save their figures and result images into this same folder.

## Setup

Use Python 3 and install the required libraries:

```bash
python3 -m pip install numpy scipy matplotlib pillow opencv-python scikit-image
```

Run commands from this directory so input and output paths are easy to follow:

```bash
cd "CS180_Website/projects/project2"
```

## Scripts

- `convolutions.py` — implements four-loop and two-loop convolution, compares the result with `scipy.signal.convolve2d`, and generates the personal-image, finite-difference, Gaussian, and DoG figures. Inputs include `color_to_greyscale.jpeg` and `cameraman.png`.
- `sharpening.py` — builds Gaussian-blurred, high-frequency residual, and unsharp-mask results for the Taj Mahal and climber images (`tai.png` and `climbing.jpeg`).
- `hybrid_image_starter.py` — creates the Nutmeg/Derek, Connor/Snow, Tiger/Lion, and Messi/Lion hybrid results, along with Fourier-analysis and cutoff-experiment figures. It uses the corresponding source images in this folder. By default, image alignment opens an interactive point-selection step. To use the preset alignment points instead, run:

  ```bash
  HYBRID_INTERACTIVE_ALIGNMENT=0 python3 hybrid_image_starter.py
  ```

- `stack.py` — contains the same-size Gaussian and Laplacian stack helpers and, when run directly, generates the apple/orange (“oraple”) figures.
- `blend.py` — uses those stack helpers to generate the lion/tiger and sunflower/explosion multiresolution blends, masks, and Laplacian-stack visualizations.
- `align_image_code.py` — helper functions for interactive image alignment; it is imported by `hybrid_image_starter.py`.

Run a script with:

```bash
python3 convolutions.py
```

Replace `convolutions.py` with the script you want to run. Some scripts use explicit Python-loop convolution and can take a while to finish. They overwrite the matching generated image files in this directory when run.

## Notes

- Keep each script's source images in this directory; paths are resolved relative to the script rather than the shell's current directory.
- Do not use OpenCV or scikit-image pyramid helpers for the stack implementation; the stacks in `stack.py` are built by repeatedly filtering without downsampling.
- The generated PNGs are the figures embedded by the project page. Re-run the relevant script after changing its code or inputs, then refresh `index.html` as needed.
