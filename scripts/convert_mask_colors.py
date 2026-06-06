#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thinsection_generator.mask_conversion import batch_convert_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert old RGB label-mask colors to the unified palette.")
    parser.add_argument("--src_dir", required=True)
    parser.add_argument("--dst_dir", required=True)
    parser.add_argument("--tolerance", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_convert_masks(args.src_dir, args.dst_dir, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
