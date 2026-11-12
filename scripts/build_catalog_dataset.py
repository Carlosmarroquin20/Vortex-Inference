#!/usr/bin/env python3
"""
Build the labelled catalog-enrichment dataset from public product listings.

Source: Amazon Berkeley Objects (ABO) listings, CC BY 4.0
(https://amazon-berkeley-objects.s3.amazonaws.com/index.html). The archive
is pinned by SHA-256 and the selection is deterministic, so the committed
evals/datasets/catalog.jsonl can be rebuilt byte for byte.

Each row pairs a messy input with labels taken from the listing's own
structured fields:

  input     title        the listing's English name, corrupted the way
                         marketplace feeds are: shouting, SKU codes and
                         promotional noise, stray punctuation, collapsed or
                         doubled spaces
            description  up to three of its bullet points, with HTML tags,
                         entities and the material and dimensions written
                         in one of several free-text forms
  expected  category     one of ten product types, lowercased
            brand        the brand field
            material     the first material value
            dimensions   length, width and height with their unit, as listed
            title        the listing's name, uncorrupted

The corruption is seeded from each item's ID, so it is reproducible, and it
never removes the information a label needs: the brand stays in the title,
and material and dimensions are always stated in the description.

Usage:
  python3 scripts/build_catalog_dataset.py --out evals/datasets/catalog.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import random
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from jsonschema import Draft202012Validator

ARCHIVE_URL = (
    "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar"
)
ARCHIVE_SHA256 = "b7f7ceacb328fa5ab6e143b88e1f948443a877cfc95b67ff09c8ebabd50644e3"

# Ten distinct product types, each with at least 70 complete English
# listings whose name contains the brand. ABO's English listings are mostly
# home goods; clothing and bags rarely carry dimensions and material.
CATEGORIES = {
    "CHAIR": "chair",
    "SOFA": "sofa",
    "RUG": "rug",
    "TABLE": "table",
    "LAMP": "lamp",
    "BED": "bed",
    "OTTOMAN": "ottoman",
    "PLANTER": "planter",
    "PILLOW": "pillow",
    "HOME_MIRROR": "mirror",
}
PER_CATEGORY = 20
SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "app" / "catalog" / "product.schema.json"
)
UNITS = {
    "inches": "in",
    "centimeters": "cm",
    "millimeters": "mm",
    "meters": "m",
    "feet": "ft",
}

NOISE = [
    "**NEW**",
    "HOT SALE!!",
    "FREE SHIPPING",
    "[In Stock]",
    "Best Seller",
    "2-Pack?",
    "~~",
]


def english(values: list[dict] | None) -> str | None:
    for value in values or []:
        if value.get("language_tag") == "en_US":
            return value["value"].strip()
    return None


def dimensions(listing: dict) -> dict | None:
    dims = listing.get("item_dimensions") or {}
    out = {}
    for axis in ("length", "width", "height"):
        entry = dims.get(axis)
        if not entry or entry.get("unit") not in UNITS:
            return None
        out[axis] = round(float(entry["value"]), 2)
        out["unit"] = UNITS[entry["unit"]]
    if len({UNITS[dims[a]["unit"]] for a in ("length", "width", "height")}) != 1:
        return None
    return out


def candidates(metadata_dir: Path):
    for path in sorted(metadata_dir.glob("listings_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                listing = json.loads(line)
                types = listing.get("product_type") or []
                if not types or types[0]["value"] not in CATEGORIES:
                    continue
                name = english(listing.get("item_name"))
                brand = english(listing.get("brand"))
                material = english(listing.get("material"))
                bullets = [
                    b["value"].strip()
                    for b in listing.get("bullet_point") or []
                    if b.get("language_tag") == "en_US"
                ]
                dims = dimensions(listing)
                if not (name and brand and material and bullets and dims):
                    continue
                material = material.replace("_", " ")
                # Label quality: the brand must appear in the name, where the
                # model has to find it; the material must be stated in the
                # listing's own text, and must not just echo the brand (ABO
                # tags Stone & Beam furniture "stone" whatever it is made of).
                text = " ".join([name, *bullets]).lower()
                if brand.lower() not in name.lower():
                    continue
                if material.lower() in brand.lower() or material.lower() not in text:
                    continue
                yield (
                    listing["item_id"],
                    types[0]["value"],
                    name,
                    brand,
                    material,
                    bullets,
                    dims,
                )


def corrupt_title(name: str, rng: random.Random) -> str:
    title = name
    style = rng.randrange(3)
    if style == 0:
        title = title.upper()
    elif style == 1:
        title = title.lower()
    sku = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(8))
    parts = [title, f"#{sku}"]
    if rng.random() < 0.7:
        parts.insert(rng.randrange(2), rng.choice(NOISE))
    title = " ".join(parts)
    title = re.sub(
        r" ", lambda _: rng.choice([" ", " ", "  ", " - ", ","]), title, count=3
    )
    return title.strip()


def describe_dimensions(dims: dict, rng: random.Random) -> str:
    ln, w, h, unit = dims["length"], dims["width"], dims["height"], dims["unit"]
    return rng.choice(
        [
            f"Dimensions: {ln} x {w} x {h} {unit}",
            f"Measures {ln}{unit} L / {w}{unit} W / {h}{unit} H",
            f"size (LxWxH): {ln}*{w}*{h} {unit}.",
            f"Product dims - length {ln} {unit}, width {w} {unit}, height {h} {unit}",
        ]
    )


def corrupt_description(
    bullets: list[str], material: str, dims: dict, rng: random.Random
) -> str:
    lines = bullets[:3]
    lines.append(
        rng.choice(
            [
                f"Material: {material}",
                f"made of {material}",
                f"MATERIAL - {material.upper()}",
            ]
        )
    )
    lines.append(describe_dimensions(dims, rng))
    rng.shuffle(lines)
    joiner = rng.choice(["<br>", " | ", "<li>", "\n\n", " ;; "])
    text = joiner.join(lines)
    if rng.random() < 0.5:
        text = html.escape(text, quote=False)
    if rng.random() < 0.5:
        text = f"<p>{text}</p>"
    return text


def labels(item: tuple) -> dict:
    _, product_type, name, brand, material, _, dims = item
    return {
        "category": CATEGORIES[product_type],
        "brand": brand,
        "material": material,
        "dimensions": dims,
        "title": name,
    }


def build(metadata_dir: Path) -> list[dict]:
    chosen: dict[str, list[tuple]] = {key: [] for key in CATEGORIES}
    # Order by a hash of the item ID: deterministic, and independent of how
    # listings happen to be ordered in the archive's files.
    # Labels are what a perfect answer looks like, so each must satisfy the
    # schema the worker enforces; a listing whose fields do not (a material
    # spelled out as a paragraph) is skipped rather than truncated.
    validator = Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    ranked = sorted(
        candidates(metadata_dir),
        key=lambda c: hashlib.sha256(c[0].encode()).hexdigest(),
    )
    for item in ranked:
        bucket = chosen[item[1]]
        if len(bucket) < PER_CATEGORY and validator.is_valid(labels(item)):
            bucket.append(item)
    rows = []
    for product_type, items in chosen.items():
        if len(items) < PER_CATEGORY:
            raise SystemExit(f"only {len(items)} complete listings of {product_type}")
        for item in items:
            item_id, _, name, brand, material, bullets, dims = item
            # Seeded per item for reproducibility; nothing here is secret.
            rng = random.Random(  # noqa: S311
                int(hashlib.sha256(item_id.encode()).hexdigest()[:16], 16)
            )
            rows.append(
                {
                    "id": f"abo-{item_id}",
                    "input": {
                        "title": corrupt_title(name, rng),
                        "description": corrupt_description(
                            bullets, material, dims, rng
                        ),
                    },
                    "expected": labels(item),
                }
            )
    return rows


def fetch(destination: Path) -> Path:
    archive = destination / "abo-listings.tar"
    urllib.request.urlretrieve(ARCHIVE_URL, archive)  # noqa: S310 - pinned https URL
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"archive checksum mismatch: {digest}")
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    return destination / "listings" / "metadata"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--metadata-dir",
        help="already extracted listings/metadata (skips the download)",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        metadata = Path(args.metadata_dir) if args.metadata_dir else fetch(Path(tmp))
        rows = build(metadata)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
