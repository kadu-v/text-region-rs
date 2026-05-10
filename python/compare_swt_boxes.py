from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


Rect = tuple[int, int, int, int]


def read_rects(path: Path) -> list[Rect]:
    with path.open(newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        return [
            (
                int(row["x"]),
                int(row["y"]),
                int(row["width"]),
                int(row["height"]),
            )
            for row in rows
        ]


def iou(a: Rect, b: Rect) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return 0.0 if union <= 0 else inter / union


def summarize(name: str, rust: list[Rect], opencv: list[Rect]) -> None:
    best = [max((iou(r, o) for o in opencv), default=0.0) for r in rust]
    reverse_best = [max((iou(o, r) for r in rust), default=0.0) for o in opencv]
    print(f"{name}:")
    print(f"  rust count: {len(rust)}")
    print(f"  opencv count: {len(opencv)}")
    if best:
        print(f"  rust->opencv mean best IoU: {mean(best):.4f}")
        print(f"  rust->opencv median best IoU: {median(best):.4f}")
        print(f"  rust boxes with IoU >= 0.5: {sum(v >= 0.5 for v in best)}")
    if reverse_best:
        print(f"  opencv boxes with IoU >= 0.5: {sum(v >= 0.5 for v in reverse_best)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stem", nargs="?", default="../resource/IMG_8364")
    args = parser.parse_args()

    stem = Path(args.stem)
    summarize(
        "letters",
        read_rects(stem.with_name(stem.name + "_swt_letters.tsv")),
        read_rects(stem.with_name(stem.name + "_opencv_swt_letters.tsv")),
    )
    summarize(
        "chains",
        read_rects(stem.with_name(stem.name + "_swt_chains.tsv")),
        read_rects(stem.with_name(stem.name + "_opencv_swt_chains.tsv")),
    )


if __name__ == "__main__":
    main()
