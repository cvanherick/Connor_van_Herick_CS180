"""
CS180 (CS280A): Project 1 starter Python code.

These are suggested libraries. Instead of scikit-image, you could use
matplotlib and OpenCV to read, write, and display images.
"""

import numpy as np
import skimage as sk
import skimage.io as skio
from pathlib import Path
DATA_DIR = Path(__file__).parent / "CS180_fa2026_proj1_data"
OUT_DIR = Path(__file__).parent / "CS180_fa2026_merged_photos"
BASELINE_OUT_DIR = Path(__file__).parent / "CS180_fa2026_baseline_photos"
OFFSETS_FILE = Path(__file__).parent / "offsets.txt"
OUT_DIR.mkdir(exist_ok=True)
BASELINE_OUT_DIR.mkdir(exist_ok=True)
def auto_contrast(img, low_percentile=1, high_percentile=99):
    low = np.percentile(img, low_percentile)
    high = np.percentile(img, high_percentile)

    img = (img - low) / (high - low)
    return np.clip(img, 0, 1)
def edge_image(img):
    dy = np.diff(img, axis=0, append=img[-1:, :])
    dx = np.diff(img, axis=1, append=img[:, -1:])
    return np.sqrt(dx**2 + dy**2)
def cross_correlation(shifted, reference):
    a = shifted - np.mean(shifted)
    b = reference - np.mean(reference)

    score = np.sum(a * b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
    return score
def align(channel, reference, cleanup=True):
    def pyramid_helper(ch, ref, center=(0,0), radius=15):
        if min(ch.shape) > 300:

            ch_small = sk.transform.rescale(ch, 0.5, anti_aliasing=True)
            ref_small = sk.transform.rescale(ref, 0.5, anti_aliasing=True)
            small_dy, small_dx = pyramid_helper(ch_small, ref_small)
            center = (2 * small_dy, 2 * small_dx)
            radius = 5



        best = [-np.inf, 0 , 0]
        center_y, center_x = center
        for dy in range(center_y - radius, center_y + radius + 1):
            for dx in range(center_x - radius, center_x + radius + 1):
                shifted = np.roll(ch, shift=(dy, dx), axis=(0, 1))
                if cleanup:
                    score = cross_correlation(crop_center(edge_image(shifted)), crop_center(edge_image(ref)))
                else:
                    score = cross_correlation(shifted, ref)
                if (score > best[0]):
                    best[0], best[1], best[2] = score, dy, dx
        return best[1], best[2]
    dy, dx = pyramid_helper(channel, reference)
    return np.roll(channel, shift=(dy, dx), axis=(0, 1)), (dy, dx)
def crop_center(img, margin=0.25):
    h, w = img.shape[:2]
    y = int(h * margin)
    x = int(w * margin)
    return img[y:h-y, x:w-x]
def process_image(imname, out_dir, cleanup=True):
    # read in the image
    im = skio.imread(DATA_DIR / imname)

    # convert to double (might want to do this later on to save memory)
    im = sk.img_as_float(im)
    if im.ndim == 3:
        im = im[:, :, 0]

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height : 2 * height]
    r = im[2 * height : 3 * height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)
    ag, g_offset = align(g, b, cleanup)
    ar, r_offset = align(r, b, cleanup)
    with OFFSETS_FILE.open("a") as offsets:
        offsets.write(f"{imname}: G offset {g_offset}, R offset {r_offset}\n")

    # create a color image
    im_out = np.dstack([ar, ag, b])
    if cleanup:
        im_out = crop_center(im_out, margin=0.07)
        im_out = auto_contrast(im_out)
    # save the image
    fname = out_dir / f"{Path(imname).stem}_out.jpg"
    skio.imsave(fname, sk.img_as_ubyte(np.clip(im_out, 0, 1)))
    print(fname)

# name of the input file

OFFSETS_FILE.write_text("")
for image_path in DATA_DIR.iterdir():
    if image_path.suffix.lower() in [".jpg", ".tif", ".png"]:
        process_image(image_path.name, OUT_DIR)
        process_image(image_path.name, BASELINE_OUT_DIR, cleanup=False)
