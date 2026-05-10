from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np


def write_rects(path: Path, rects) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["x", "y", "width", "height"])
        if rects is None:
            return
        rects = np.asarray(rects).reshape(-1, 4)
        for rect in rects:
            x, y, w, h = [int(v) for v in rect]
            writer.writerow([x, y, w, h])


def draw_rects(image, rects) -> None:
    if rects is None:
        return
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    rects = np.asarray(rects).reshape(-1, 4)
    for i, rect in enumerate(rects):
        x, y, w, h = [int(v) for v in rect]
        cv2.rectangle(image, (x, y), (x + w, y + h), colors[i % len(colors)], 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="../resource/IMG_8364.jpeg")
    parser.add_argument("--light-on-dark", action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"failed to read image: {image_path}")

    start = time.perf_counter()
    letters, draw, chains = cv2.text.detectTextSWT(
        img, not args.light_on_dark
    )
    elapsed = time.perf_counter() - start

    stem = image_path.with_suffix("")
    debug_path = stem.with_name(stem.name + "_opencv_swt_debug.png")
    overlay_path = stem.with_name(stem.name + "_opencv_swt_overlay.png")
    letters_path = stem.with_name(stem.name + "_opencv_swt_letters.tsv")
    chains_path = stem.with_name(stem.name + "_opencv_swt_chains.tsv")

    if draw is not None:
        cv2.imwrite(str(debug_path), draw)
    overlay = img.copy()
    draw_rects(overlay, letters)
    cv2.imwrite(str(overlay_path), overlay)
    write_rects(letters_path, letters)
    write_rects(chains_path, chains)

    print(f"image: {image_path}")
    print(f"size: {img.shape[1]}x{img.shape[0]}")
    print(f"elapsed: {elapsed:.6f}s")
    print(f"letter boxes: {0 if letters is None else len(letters)}")
    chain_count = 0 if chains is None else np.asarray(chains).reshape(-1, 4).shape[0]
    print(f"chain boxes: {chain_count}")
    print(f"saved debug: {debug_path}")
    print(f"saved overlay: {overlay_path}")
    print(f"saved letters: {letters_path}")
    print(f"saved chains: {chains_path}")


if __name__ == "__main__":
    main()
