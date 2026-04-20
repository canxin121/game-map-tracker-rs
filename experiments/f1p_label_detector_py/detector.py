from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

TAG_NAMES = ["F1", "F2", "F3", "F4", "J", "P"]
EXPECTED_LABEL_COUNT = len(TAG_NAMES)
RAW_TEMPLATE_SIZE = (48, 32)
DEFAULT_MODEL_PATH = Path(__file__).with_name("artifacts").joinpath("f1p_tag_model.npz")
DEFAULT_POSITIVE_DIR = Path(__file__).with_name("samples").joinpath("positive")
DEFAULT_NEGATIVE_DIR = Path(__file__).with_name("samples").joinpath("negative")
DEFAULT_DEBUG_DIR = Path(__file__).with_name("debug")
DEFAULT_ASSET_TEST_DIR = Path(__file__).resolve().parents[2].joinpath("assets", "test")
SLOT_RAW_THRESHOLD = 0.90
PRESENT_MIN_RAW = 0.90
PRESENT_MEAN_RAW = 0.96


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center_x(self) -> float:
        return self.x + self.w * 0.5

    def clip(self, width: int, height: int) -> "Box":
        x1 = max(0, min(self.x, width - 1))
        y1 = max(0, min(self.y, height - 1))
        x2 = max(x1 + 1, min(self.x2, width))
        y2 = max(y1 + 1, min(self.y2, height))
        return Box(x1, y1, x2 - x1, y2 - y1)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class SlotDetection:
    tag: str
    found: bool
    box: Box | None
    raw_score: float


@dataclass
class DetectionResult:
    present: bool
    final_score: float
    mean_raw_score: float
    min_raw_score: float
    slot_results: list[SlotDetection]
    image_size: tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "present": self.present,
            "final_score": round(self.final_score, 4),
            "mean_raw_score": round(self.mean_raw_score, 4),
            "min_raw_score": round(self.min_raw_score, 4),
            "image_size": list(self.image_size),
            "slots": [
                {
                    "tag": slot.tag,
                    "found": slot.found,
                    "box": list(slot.box.as_tuple()) if slot.box else None,
                    "raw_score": round(slot.raw_score, 4),
                }
                for slot in self.slot_results
            ],
        }


@dataclass
class DetectorModel:
    target_size: tuple[int, int]
    anchor_boxes: list[Box]
    raw_templates: list[np.ndarray]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "target_size": np.array(self.target_size, dtype=np.int32),
            "anchor_boxes": np.array([box.as_tuple() for box in self.anchor_boxes], dtype=np.int32),
        }
        for index, templates in enumerate(self.raw_templates):
            payload[f"raw_{index}"] = templates.astype(np.uint8)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "DetectorModel":
        blob = np.load(path)
        target_width, target_height = blob["target_size"].tolist()
        anchor_boxes = [Box(*coords) for coords in blob["anchor_boxes"].tolist()]
        raw_templates = [blob[f"raw_{index}"].astype(np.uint8) for index in range(EXPECTED_LABEL_COUNT)]
        return cls(
            target_size=(int(target_width), int(target_height)),
            anchor_boxes=anchor_boxes,
            raw_templates=raw_templates,
        )


def rgb_luma(image_rgb: np.ndarray) -> np.ndarray:
    return image_rgb[..., 0] * 0.299 + image_rgb[..., 1] * 0.587 + image_rgb[..., 2] * 0.114


def channel_spread(image_rgb: np.ndarray) -> np.ndarray:
    return image_rgb.max(axis=2) - image_rgb.min(axis=2)


def color_distance(image_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    delta = image_rgb - target_rgb.reshape(1, 1, 3)
    return np.sqrt(np.sum(delta * delta, axis=2))


def lower_luma_mean(values: np.ndarray, ratio: float) -> np.ndarray:
    if values.size == 0:
        return np.array([60.0, 110.0, 188.0], dtype=np.float32)
    luma = values[:, 0] * 0.299 + values[:, 1] * 0.587 + values[:, 2] * 0.114
    take = max(1, int(round(len(values) * max(0.1, min(ratio, 1.0)))))
    order = np.argsort(luma)
    return values[order[:take]].mean(axis=0)


def percentile_or_default(values: np.ndarray, quantile: float, default: float) -> float:
    if values.size == 0:
        return default
    return float(np.quantile(values, quantile))


def morphological_cleanup(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def border_samples(image_rgb: np.ndarray) -> np.ndarray:
    edge = min(2, max(0, image_rgb.shape[0] - 1), max(0, image_rgb.shape[1] - 1))
    if edge == 0:
        return image_rgb.reshape(-1, 3)
    top = image_rgb[: edge + 1].reshape(-1, 3)
    bottom = image_rgb[-(edge + 1) :].reshape(-1, 3)
    left = image_rgb[:, : edge + 1].reshape(-1, 3)
    right = image_rgb[:, -(edge + 1) :].reshape(-1, 3)
    return np.concatenate([top, bottom, left, right], axis=0)


def derive_palette(image_rgb: np.ndarray) -> dict[str, float | np.ndarray]:
    border = border_samples(image_rgb)
    background = lower_luma_mean(border, 0.55).astype(np.float32)
    background_luma = float(background[0] * 0.299 + background[1] * 0.587 + background[2] * 0.114)

    luma = rgb_luma(image_rgb)
    bg_dist = color_distance(image_rgb, background)
    spread = channel_spread(image_rgb)
    rough_keep = ((luma >= background_luma + 32.0) & (bg_dist >= 18.0)) | (
        (luma >= background_luma + 20.0) & (spread <= 26.0)
    )
    rough_mask = morphological_cleanup((rough_keep.astype(np.uint8)) * 255)

    support_samples = image_rgb[rough_mask > 0]
    if support_samples.size == 0:
        support = np.array([244.0, 236.0, 220.0], dtype=np.float32)
    else:
        support = support_samples.mean(axis=0).astype(np.float32)
    support_luma = float(support[0] * 0.299 + support[1] * 0.587 + support[2] * 0.114)
    support_gap = float(max(np.linalg.norm(support - background), 1.0))
    support_radius = percentile_or_default(
        np.linalg.norm(support_samples - support, axis=1) if support_samples.size else np.array([], dtype=np.float32),
        0.93,
        20.0,
    )
    support_radius = max(support_radius, support_gap * 0.18, 14.0) + 6.0

    if support_samples.size == 0:
        ink_samples = np.empty((0, 3), dtype=np.float32)
    else:
        ink_samples = support_samples[
            (support_samples[:, 0] * 0.299 + support_samples[:, 1] * 0.587 + support_samples[:, 2] * 0.114)
            <= support_luma - 56.0
        ]
    if ink_samples.size == 0:
        ink = np.array([74.0, 76.0, 86.0], dtype=np.float32)
    else:
        ink = ink_samples.mean(axis=0).astype(np.float32)
    ink_luma = float(ink[0] * 0.299 + ink[1] * 0.587 + ink[2] * 0.114)
    ink_gap = float(max(np.linalg.norm(ink - support), 1.0))
    ink_radius = percentile_or_default(
        np.linalg.norm(ink_samples - ink, axis=1) if ink_samples.size else np.array([], dtype=np.float32),
        0.93,
        16.0,
    )
    ink_radius = max(ink_radius, ink_gap * 0.16, 10.0) + 6.0

    return {
        "background": background,
        "background_luma": background_luma,
        "support": support,
        "support_luma": support_luma,
        "support_gap": support_gap,
        "support_radius": support_radius,
        "ink": ink,
        "ink_luma": ink_luma,
        "ink_gap": ink_gap,
        "ink_radius": ink_radius,
    }


def detect_support_mask(image_rgb: np.ndarray, palette: dict[str, float | np.ndarray]) -> np.ndarray:
    background = np.asarray(palette["background"], dtype=np.float32)
    support = np.asarray(palette["support"], dtype=np.float32)
    background_luma = float(palette["background_luma"])
    support_luma = float(palette["support_luma"])
    support_gap = float(palette["support_gap"])
    support_radius = float(palette["support_radius"])

    luma = rgb_luma(image_rgb)
    support_dist = color_distance(image_rgb, support)
    background_dist = color_distance(image_rgb, background)
    spread = channel_spread(image_rgb)
    blue_excess = image_rgb[..., 2] - (image_rgb[..., 0] + image_rgb[..., 1]) * 0.5

    min_luma = background_luma + (support_luma - background_luma) * 0.28
    min_background_distance = max(support_gap * 0.12, 10.0)
    keep = (
        (luma >= min_luma)
        & (background_dist >= min_background_distance * 0.60)
        & (
            ((support_dist <= support_radius * 2.10) & (support_dist <= background_dist * 1.45 + 18.0))
            | (blue_excess <= 10.0)
            | ((luma >= min_luma - 10.0) & (spread <= 26.0))
        )
    )
    return morphological_cleanup((keep.astype(np.uint8)) * 255)


def filter_components(
    mask: np.ndarray,
    min_width: float,
    max_width: float,
    min_height: float,
    max_height: float,
    min_area_ratio: float = 0.0025,
    max_area_ratio: float = 0.24,
) -> list[Box]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    image_area = mask.shape[0] * mask.shape[1]
    components: list[Box] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        fill_ratio = area / max(1, w * h)
        if area < image_area * min_area_ratio or area > image_area * max_area_ratio:
            continue
        if w < min_width or w > max_width:
            continue
        if h < min_height or h > max_height:
            continue
        if fill_ratio < 0.10:
            continue
        components.append(Box(int(x), int(y), int(w), int(h)))
    components.sort(key=lambda box: box.center_x)
    return components


def seed_boxes_from_image(image_rgb: np.ndarray) -> list[Box]:
    palette = derive_palette(image_rgb)
    support_mask = detect_support_mask(image_rgb, palette)
    height, width = support_mask.shape
    boxes = filter_components(
        support_mask,
        min_width=width * 0.018,
        max_width=width * 0.26,
        min_height=height * 0.18,
        max_height=height * 1.0,
    )
    if len(boxes) != EXPECTED_LABEL_COUNT:
        return []
    return boxes


def preprocess_raw_crop(gray_crop: np.ndarray) -> np.ndarray:
    gray = cv2.resize(gray_crop, RAW_TEMPLATE_SIZE, interpolation=cv2.INTER_LINEAR)
    gray = cv2.equalizeHist(gray)
    return gray


def prepare_centered_template_bank(templates: np.ndarray) -> np.ndarray:
    prepared = templates.astype(np.float32).reshape(templates.shape[0], -1) / 255.0
    prepared = prepared - prepared.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(prepared, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return prepared / norms


def grayscale_similarity_score_prepared(image_gray: np.ndarray, prepared_templates: np.ndarray) -> float:
    image = image_gray.astype(np.float32).reshape(-1) / 255.0
    image = image - float(np.mean(image))
    image_norm = float(np.linalg.norm(image))
    if image_norm <= 1e-6:
        return 0.0
    normalized = image / image_norm
    return float(np.max(prepared_templates @ normalized))


def normalize_image(image_rgb: np.ndarray, target_size: tuple[int, int] | None) -> np.ndarray:
    if target_size is None:
        return image_rgb
    target_width, target_height = target_size
    if image_rgb.shape[1] == target_width and image_rgb.shape[0] == target_height:
        return image_rgb
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.array(handle.convert("RGB"), dtype=np.uint8)


def load_images(paths: Iterable[Path], target_size: tuple[int, int] | None = None) -> list[np.ndarray]:
    images = []
    for path in paths:
        image = load_rgb_image(path)
        images.append(normalize_image(image, target_size))
    return images


def anchor_boxes_from_seed_images(images: list[np.ndarray]) -> list[Box]:
    seed_sets = [seed_boxes_from_image(image) for image in images]
    seed_sets = [seed for seed in seed_sets if len(seed) == EXPECTED_LABEL_COUNT]
    if not seed_sets:
        raise RuntimeError("failed to derive any seed boxes from positive samples")

    anchors: list[Box] = []
    for slot_index in range(EXPECTED_LABEL_COUNT):
        xs = [seed[slot_index].x for seed in seed_sets]
        ys = [seed[slot_index].y for seed in seed_sets]
        ws = [seed[slot_index].w for seed in seed_sets]
        hs = [seed[slot_index].h for seed in seed_sets]
        anchors.append(
            Box(
                int(round(float(np.median(xs)))),
                int(round(float(np.median(ys)))),
                int(round(float(np.median(ws)))),
                int(round(float(np.median(hs)))),
            )
        )
    return anchors


def train_model(positive_dir: Path, model_path: Path) -> DetectorModel:
    positive_paths = sorted(positive_dir.glob("*.png"))
    if not positive_paths:
        raise RuntimeError(f"no positive PNG files found in {positive_dir}")

    raw_images = load_images(positive_paths)
    target_size = (raw_images[0].shape[1], raw_images[0].shape[0])
    images = [normalize_image(image, target_size) for image in raw_images]
    anchors = anchor_boxes_from_seed_images(images)

    raw_templates: list[list[np.ndarray]] = [[] for _ in range(EXPECTED_LABEL_COUNT)]

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        for slot_index, anchor in enumerate(anchors):
            gray_crop = gray[anchor.y : anchor.y2, anchor.x : anchor.x2]
            raw_templates[slot_index].append(preprocess_raw_crop(gray_crop))

    stacked_raw: list[np.ndarray] = []
    for slot_index, tag in enumerate(TAG_NAMES):
        if not raw_templates[slot_index]:
            raise RuntimeError(f"failed to build raw template bank for slot {slot_index} ({tag})")
        stacked_raw.append(np.stack(raw_templates[slot_index], axis=0))

    model = DetectorModel(
        target_size=target_size,
        anchor_boxes=anchors,
        raw_templates=stacked_raw,
    )
    model.save(model_path)
    return model


class F1PTagDetector:
    def __init__(self, model: DetectorModel):
        self.model = model
        self.anchor_slices = [
            (slice(anchor.y, anchor.y2), slice(anchor.x, anchor.x2)) for anchor in self.model.anchor_boxes
        ]
        self.prepared_raw_templates = [
            prepare_centered_template_bank(templates) for templates in self.model.raw_templates
        ]

    def _raw_scores(self, normalized_gray: np.ndarray) -> list[float]:
        scores: list[float] = []
        for slot_index, (rows, cols) in enumerate(self.anchor_slices):
            raw_crop = normalized_gray[rows, cols]
            scores.append(
                grayscale_similarity_score_prepared(
                    preprocess_raw_crop(raw_crop),
                    self.prepared_raw_templates[slot_index],
                )
            )
        return scores

    def detect(self, image_rgb: np.ndarray) -> DetectionResult:
        normalized = normalize_image(image_rgb, self.model.target_size)
        normalized_gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
        raw_scores = self._raw_scores(normalized_gray)
        mean_raw = float(np.mean(raw_scores)) if raw_scores else 0.0
        min_raw = min(raw_scores, default=0.0)
        final_score = 0.65 * mean_raw + 0.35 * min_raw
        present = min_raw >= PRESENT_MIN_RAW and mean_raw >= PRESENT_MEAN_RAW

        slot_results = [
            SlotDetection(
                tag=tag,
                found=raw_score >= SLOT_RAW_THRESHOLD,
                box=anchor,
                raw_score=raw_score,
            )
            for tag, anchor, raw_score in zip(TAG_NAMES, self.model.anchor_boxes, raw_scores)
        ]
        return DetectionResult(
            present=present,
            final_score=float(final_score),
            mean_raw_score=mean_raw,
            min_raw_score=float(min_raw),
            slot_results=slot_results,
            image_size=self.model.target_size,
        )


def draw_rounded_rect(image: np.ndarray, box: Box, color: tuple[int, int, int], radius: int) -> None:
    x1, y1, x2, y2 = box.x, box.y, box.x2 - 1, box.y2 - 1
    radius = max(1, min(radius, box.w // 2, box.h // 2))
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x1 + radius, y1 + radius), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x2 - radius, y1 + radius), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x1 + radius, y2 - radius), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (x2 - radius, y2 - radius), radius, color, thickness=-1, lineType=cv2.LINE_AA)


def random_background(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    start = rng.integers(0, 160, size=3, dtype=np.int32)
    end = np.clip(start + rng.integers(-50, 70, size=3, dtype=np.int32), 0, 255)
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    alpha = (0.7 * xx + 0.3 * yy).reshape(height, width, 1)
    base = start.reshape(1, 1, 3) * (1.0 - alpha) + end.reshape(1, 1, 3) * alpha
    noise = rng.normal(0.0, 18.0, size=(height, width, 3)).astype(np.float32)
    image = np.clip(base + noise, 0, 255).astype(np.uint8)

    for _ in range(rng.integers(4, 10)):
        color = tuple(int(value) for value in rng.integers(20, 235, size=3))
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        axes = (int(rng.integers(12, 80)), int(rng.integers(6, 28)))
        angle = float(rng.integers(0, 180))
        cv2.ellipse(image, center, axes, angle, 0, 360, color, thickness=-1, lineType=cv2.LINE_AA)

    image = cv2.GaussianBlur(image, (5, 5), sigmaX=0.0)
    return image


def draw_decoy_labels(rng: np.random.Generator, image: np.ndarray, anchors: list[Box]) -> None:
    wrong_sets = [
        ["Q", "E", "R", "T", "Y", "U"],
        ["F1", "F3", "F2", "F4", "K", "O"],
        ["1", "2", "3", "4", "5", "6"],
        ["A", "B", "C", "D", "E", "F"],
    ]
    labels = wrong_sets[int(rng.integers(0, len(wrong_sets)))]
    for index, anchor in enumerate(anchors):
        if rng.random() < 0.18:
            continue
        dx = int(rng.integers(-12, 13))
        dy = int(rng.integers(-2, 3))
        w = max(28, anchor.w + int(rng.integers(-3, 4)))
        h = max(20, anchor.h + int(rng.integers(-2, 3)))
        box = Box(anchor.x + dx, anchor.y + dy, w, h).clip(image.shape[1], image.shape[0])
        fill = tuple(int(value) for value in rng.integers(225, 252, size=3))
        draw_rounded_rect(image, box, fill, radius=6)
        text = labels[index]
        scale = 0.45 if len(text) == 1 else 0.42
        thickness = 1
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        org = (
            int(box.x + (box.w - text_size[0]) / 2),
            int(box.y + (box.h + text_size[1]) / 2) - 2,
        )
        ink = tuple(int(value) for value in rng.integers(10, 90, size=3))
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, ink, thickness, cv2.LINE_AA)


def generate_negative_samples(output_dir: Path, count: int, model: DetectorModel | None, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = model.target_size if model else (587, 36)
    anchors = model.anchor_boxes if model else [
        Box(9, 3, 44, 30),
        Box(122, 3, 44, 30),
        Box(228, 3, 44, 30),
        Box(336, 3, 44, 30),
        Box(443, 3, 39, 30),
        Box(542, 3, 38, 30),
    ]

    created: list[Path] = []
    for index in range(count):
        image = random_background(rng, width, height)
        mode = index % 4
        if mode == 1:
            draw_decoy_labels(rng, image, anchors[: rng.integers(3, 6)])
        elif mode == 2:
            draw_decoy_labels(rng, image, anchors)
            cv2.line(image, (0, int(rng.integers(8, height - 8))), (width - 1, int(rng.integers(8, height - 8))), (250, 250, 250), 2, cv2.LINE_AA)
        elif mode == 3:
            for _ in range(rng.integers(6, 10)):
                left = int(rng.integers(0, width - 25))
                top = int(rng.integers(0, height - 14))
                blob = Box(left, top, int(rng.integers(12, 60)), int(rng.integers(10, 26))).clip(width, height)
                draw_rounded_rect(image, blob, tuple(int(v) for v in rng.integers(215, 255, size=3)), radius=5)

        path = output_dir / f"no_map_random_{index:03d}.png"
        Image.fromarray(image).save(path)
        created.append(path)
    return created


def detector_from_model_path(model_path: Path, positive_dir: Path) -> F1PTagDetector:
    if model_path.exists():
        model = DetectorModel.load(model_path)
    else:
        model = train_model(positive_dir, model_path)
    return F1PTagDetector(model)


def sync_asset_test_samples(source_dir: Path, positive_dir: Path, negative_dir: Path) -> dict[str, int]:
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    positive_paths = sorted(source_dir.glob("has_map_*.png"))
    negative_paths = sorted(source_dir.glob("no_map_*.png"))
    for path in positive_paths:
        shutil.copy2(path, positive_dir / path.name)
    for path in negative_paths:
        shutil.copy2(path, negative_dir / path.name)

    return {
        "positive_copied": len(positive_paths),
        "negative_copied": len(negative_paths),
    }


def save_debug_render(path: Path, image_rgb: np.ndarray, result: DetectionResult) -> None:
    render = image_rgb.copy()
    for slot in result.slot_results:
        if slot.box is None:
            continue
        color = (72, 220, 96) if slot.found else (240, 70, 70)
        cv2.rectangle(render, (slot.box.x, slot.box.y), (slot.box.x2 - 1, slot.box.y2 - 1), color, 1, cv2.LINE_AA)
        label = f"{slot.tag}:{slot.raw_score:.2f}"
        cv2.putText(
            render,
            label,
            (slot.box.x, max(10, slot.box.y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        render,
        f"present={result.present} score={result.final_score:.3f}",
        (6, result.image_size[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    Image.fromarray(render).save(path)


def evaluate_dataset(
    detector: F1PTagDetector,
    positive_dir: Path,
    negative_dir: Path,
    debug_dir: Path | None,
) -> dict:
    positive_paths = sorted(positive_dir.glob("*.png"))
    negative_paths = sorted(negative_dir.glob("*.png"))
    if not positive_paths:
        raise RuntimeError(f"no positive PNG files found in {positive_dir}")
    if not negative_paths:
        raise RuntimeError(f"no negative PNG files found in {negative_dir}")

    debug_dir.mkdir(parents=True, exist_ok=True) if debug_dir else None

    positive_hits = 0
    negative_hits = 0
    sample_rows = []
    start = time.perf_counter()
    per_image_times_ms: list[float] = []

    for label, expected, paths in [("positive", True, positive_paths), ("negative", False, negative_paths)]:
        for path in paths:
            image = load_rgb_image(path)
            tick = time.perf_counter()
            result = detector.detect(image)
            elapsed_ms = (time.perf_counter() - tick) * 1000.0
            per_image_times_ms.append(elapsed_ms)

            if debug_dir:
                save_debug_render(debug_dir / f"{path.stem}_debug.png", normalize_image(image, detector.model.target_size), result)

            if expected and result.present:
                positive_hits += 1
            if (not expected) and result.present:
                negative_hits += 1

            sample_rows.append(
                {
                    "file": path.name,
                    "kind": label,
                    "expected": expected,
                    "predicted": result.present,
                    "final_score": round(result.final_score, 4),
                    "mean_raw_score": round(result.mean_raw_score, 4),
                    "min_raw_score": round(result.min_raw_score, 4),
                }
            )

    total_elapsed_ms = (time.perf_counter() - start) * 1000.0
    summary = {
        "positive_total": len(positive_paths),
        "positive_true_positive": positive_hits,
        "positive_recall": round(positive_hits / len(positive_paths), 4),
        "negative_total": len(negative_paths),
        "negative_false_positive": negative_hits,
        "negative_specificity": round((len(negative_paths) - negative_hits) / len(negative_paths), 4),
        "average_inference_ms": round(float(np.mean(per_image_times_ms)), 3),
        "median_inference_ms": round(float(np.median(per_image_times_ms)), 3),
        "dataset_elapsed_ms": round(total_elapsed_ms, 3),
        "samples": sample_rows,
    }
    return summary


def command_build_model(args: argparse.Namespace) -> int:
    model = train_model(args.positive_dir, args.model)
    print(f"model_saved={args.model}")
    print(f"target_size={model.target_size[0]}x{model.target_size[1]}")
    for tag, anchor in zip(TAG_NAMES, model.anchor_boxes):
        print(f"anchor[{tag}]={anchor.as_tuple()}")
    return 0


def command_generate_negatives(args: argparse.Namespace) -> int:
    model = DetectorModel.load(args.model) if args.model.exists() else None
    created = generate_negative_samples(args.output_dir, args.count, model, args.seed)
    print(f"generated={len(created)}")
    print(f"output_dir={args.output_dir}")
    return 0


def command_sync_assets(args: argparse.Namespace) -> int:
    summary = sync_asset_test_samples(args.source_dir, args.positive_dir, args.negative_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_detect(args: argparse.Namespace) -> int:
    detector = detector_from_model_path(args.model, args.positive_dir)
    image = load_rgb_image(args.image)
    result = detector.detect(image)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.debug_output:
        args.debug_output.parent.mkdir(parents=True, exist_ok=True)
        save_debug_render(args.debug_output, normalize_image(image, detector.model.target_size), result)
        print(f"debug_saved={args.debug_output}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    detector = detector_from_model_path(args.model, args.positive_dir)
    if not any(args.negative_dir.glob("*.png")) and args.generate_negatives_if_missing:
        generate_negative_samples(args.negative_dir, args.negative_count, detector.model, args.seed)

    summary = evaluate_dataset(detector, args.positive_dir, args.negative_dir, args.debug_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report_saved={args.report}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F1-P tag presence detector for the cropped probe band")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_model = subparsers.add_parser("build-model", help="train template banks from positive PNG samples")
    build_model.add_argument("--positive-dir", type=Path, default=DEFAULT_POSITIVE_DIR)
    build_model.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    build_model.set_defaults(func=command_build_model)

    generate = subparsers.add_parser("generate-negatives", help="generate random false PNG samples")
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_NEGATIVE_DIR)
    generate.add_argument("--count", type=int, default=80)
    generate.add_argument("--seed", type=int, default=17766681)
    generate.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    generate.set_defaults(func=command_generate_negatives)

    sync_assets = subparsers.add_parser("sync-assets", help="copy has_map/no_map PNGs from assets/test into experiment samples")
    sync_assets.add_argument("--source-dir", type=Path, default=DEFAULT_ASSET_TEST_DIR)
    sync_assets.add_argument("--positive-dir", type=Path, default=DEFAULT_POSITIVE_DIR)
    sync_assets.add_argument("--negative-dir", type=Path, default=DEFAULT_NEGATIVE_DIR)
    sync_assets.set_defaults(func=command_sync_assets)

    detect = subparsers.add_parser("detect", help="run the detector on a single image")
    detect.add_argument("--image", type=Path, required=True)
    detect.add_argument("--positive-dir", type=Path, default=DEFAULT_POSITIVE_DIR)
    detect.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    detect.add_argument("--debug-output", type=Path, default=None)
    detect.set_defaults(func=command_detect)

    evaluate = subparsers.add_parser("evaluate", help="evaluate positives and negatives and print metrics")
    evaluate.add_argument("--positive-dir", type=Path, default=DEFAULT_POSITIVE_DIR)
    evaluate.add_argument("--negative-dir", type=Path, default=DEFAULT_NEGATIVE_DIR)
    evaluate.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    evaluate.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    evaluate.add_argument("--report", type=Path, default=Path(__file__).with_name("artifacts").joinpath("evaluation_report.json"))
    evaluate.add_argument("--generate-negatives-if-missing", action="store_true")
    evaluate.add_argument("--negative-count", type=int, default=80)
    evaluate.add_argument("--seed", type=int, default=17766681)
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
