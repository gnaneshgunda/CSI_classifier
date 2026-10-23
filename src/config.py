"""
src/config.py
=============
Typed configuration layer for the CSI-HAR pipeline.

Usage
-----
    from src.config import load_config, ExperimentConfig
    cfg = load_config("configs/experiment.yaml")
    print(cfg.window.length)          # 150
    print(cfg.paths.manifest_csv)     # "derived/manifest.csv"

Design notes
------------
* Every section of experiment.yaml maps 1-to-1 to a frozen dataclass.
* ``load_config`` validates all values before returning – callers can assume
  the returned object is internally consistent.
* Paths are stored as ``pathlib.Path`` objects so downstream code never
  constructs path strings manually.
* The module is import-safe with no side effects at import time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Literal, Optional

import yaml


# ---------------------------------------------------------------------------
# Allowed literal sets (mirrors experiment.yaml vocabularies)
# ---------------------------------------------------------------------------

ALLOWED_ROOMS: frozenset[str] = frozenset({"A", "B"})
ALLOWED_SPLITS: frozenset[str] = frozenset({"dev", "val"})
ALLOWED_ACTIVITIES: frozenset[str] = frozenset(
    {"empty", "fall", "lying", "sit", "sitdown", "stand", "standup", "walk"}
)
ALLOWED_SHORT_CAPTURE_POLICIES: frozenset[str] = frozenset(
    {"reflect_pad", "exclude"}
)
ALLOWED_NORM_METHODS: frozenset[str] = frozenset(
    {"zscore", "minmax", "robust", "per_capture"}
)
ALLOWED_ENCODERS: frozenset[str] = frozenset({"cnn1d", "transformer"})
ALLOWED_ALIGN_METHODS: frozenset[str] = frozenset(
    {"procrustes", "coral", "none"}
)
ALLOWED_ROUTING_MODES: frozenset[str] = frozenset(
    {"predicted_hard", "predicted_soft", "oracle", "none"}
)
ALLOWED_FS_CLASSIFIERS: frozenset[str] = frozenset({"ncm", "lda"})
ALLOWED_SHRINKAGE: frozenset[str] = frozenset({"ledoit_wolf", "none"})
ALLOWED_DISTANCE_METRICS: frozenset[str] = frozenset({"cosine", "euclidean"})


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathsConfig:
    room_a_pcap_dir: Path
    room_a_val_dir: Path
    room_b_csv_dir: Path
    derived_dir: Path
    intermediate_dir: Path
    windows_dir: Path
    manifest_csv: Path
    windows_index_csv: Path
    checkpoints_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class SeedsConfig:
    global_seed: int
    fewshot_seeds: List[int]


@dataclass(frozen=True)
class WindowConfig:
    length: int
    n_subcarriers: int
    dev_stride: int
    val_stride: int
    short_capture_policy: str


@dataclass(frozen=True)
class CleaningConfig:
    min_valid_packets: int
    max_nan_fraction_per_packet: float
    amplitude_clip_max: float
    amplitude_floor: float
    max_zero_subcarrier_fraction: float


@dataclass(frozen=True)
class NormalizationConfig:
    method: str
    clip_after_norm: Optional[float]


@dataclass(frozen=True)
class SSLConfig:
    encoder: str
    projection_dim: int
    embedding_dim: int
    temperature: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    lr_min: float
    n_negatives: Optional[int]


@dataclass(frozen=True)
class AugmentConfig:
    gaussian_noise_std: float
    time_crop_min_frac: float
    subcarrier_mask_frac: float
    amplitude_scale_min: float
    amplitude_scale_max: float


@dataclass(frozen=True)
class AlignConfig:
    method: str


@dataclass(frozen=True)
class RoutingConfig:
    mode: str
    confidence_threshold: float


@dataclass(frozen=True)
class FewshotConfig:
    K_list: List[int]
    n_seeds: int
    distance_metric: str
    shrinkage: str
    classifier: str


@dataclass(frozen=True)
class ActivitiesConfig:
    room_a: List[str]
    room_b: List[str]
    shared: List[str]
    room_a_exclusive: List[str]
    room_b_exclusive: List[str]
    all: List[str]


@dataclass(frozen=True)
class RoomsConfig:
    all: List[str]


@dataclass(frozen=True)
class SplitsConfig:
    all: List[str]
    val_seed: int
    val_captures_per_class: int


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config object returned by ``load_config``."""
    paths: PathsConfig
    seeds: SeedsConfig
    window: WindowConfig
    cleaning: CleaningConfig
    normalization: NormalizationConfig
    ssl: SSLConfig
    augment: AugmentConfig
    align: AlignConfig
    routing: RoutingConfig
    fewshot: FewshotConfig
    activities: ActivitiesConfig
    rooms: RoomsConfig
    splits: SplitsConfig

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def config_hash(self) -> str:
        """SHA-256 of the serialised config (hex, first 16 chars).

        Used to tag derived windows so that a config change invalidates
        the cached window index without manual cleanup.
        """
        blob = json.dumps(asdict(self), default=str, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def activity_to_room(self, activity: str) -> Optional[str]:
        """Return the room that exclusively owns *activity*, or None if shared."""
        if activity in self.activities.room_a_exclusive:
            return "A"
        if activity in self.activities.room_b_exclusive:
            return "B"
        return None  # shared activity


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require(value: object, allowed: frozenset[str], field_name: str) -> None:
    """Raise ``ValueError`` if *value* is not in *allowed*."""
    if value not in allowed:
        raise ValueError(
            f"Config field '{field_name}' has value {value!r}; "
            f"allowed values: {sorted(allowed)}"
        )


def _require_positive(value: float | int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"Config field '{field_name}' must be > 0, got {value!r}")


def _require_fraction(value: float, field_name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(
            f"Config field '{field_name}' must be in [0, 1], got {value!r}"
        )


def _parse_paths(raw: dict) -> PathsConfig:
    return PathsConfig(
        room_a_pcap_dir=Path(raw["room_a_pcap_dir"]),
        room_a_val_dir=Path(raw["room_a_val_dir"]),
        room_b_csv_dir=Path(raw["room_b_csv_dir"]),
        derived_dir=Path(raw["derived_dir"]),
        intermediate_dir=Path(raw["intermediate_dir"]),
        windows_dir=Path(raw["windows_dir"]),
        manifest_csv=Path(raw["manifest_csv"]),
        windows_index_csv=Path(raw["windows_index_csv"]),
        checkpoints_dir=Path(raw["checkpoints_dir"]),
        reports_dir=Path(raw["reports_dir"]),
    )


def _parse_seeds(raw: dict) -> SeedsConfig:
    seeds = SeedsConfig(
        global_seed=int(raw["global_seed"]),
        fewshot_seeds=[int(s) for s in raw["fewshot_seeds"]],
    )
    _require_positive(seeds.global_seed + 1, "seeds.global_seed")  # allow 0
    return seeds


def _parse_window(raw: dict) -> WindowConfig:
    cfg = WindowConfig(
        length=int(raw["length"]),
        n_subcarriers=int(raw["n_subcarriers"]),
        dev_stride=int(raw["dev_stride"]),
        val_stride=int(raw["val_stride"]),
        short_capture_policy=str(raw["short_capture_policy"]),
    )
    _require_positive(cfg.length, "window.length")
    _require_positive(cfg.n_subcarriers, "window.n_subcarriers")
    _require_positive(cfg.dev_stride, "window.dev_stride")
    _require_positive(cfg.val_stride, "window.val_stride")
    _require(
        cfg.short_capture_policy,
        ALLOWED_SHORT_CAPTURE_POLICIES,
        "window.short_capture_policy",
    )
    return cfg


def _parse_cleaning(raw: dict) -> CleaningConfig:
    cfg = CleaningConfig(
        min_valid_packets=int(raw["min_valid_packets"]),
        max_nan_fraction_per_packet=float(raw["max_nan_fraction_per_packet"]),
        amplitude_clip_max=float(raw["amplitude_clip_max"]),
        amplitude_floor=float(raw["amplitude_floor"]),
        max_zero_subcarrier_fraction=float(raw["max_zero_subcarrier_fraction"]),
    )
    _require_positive(cfg.min_valid_packets, "cleaning.min_valid_packets")
    _require_fraction(cfg.max_nan_fraction_per_packet, "cleaning.max_nan_fraction_per_packet")
    _require_fraction(cfg.max_zero_subcarrier_fraction, "cleaning.max_zero_subcarrier_fraction")
    if cfg.amplitude_clip_max <= cfg.amplitude_floor:
        raise ValueError(
            "cleaning.amplitude_clip_max must be > cleaning.amplitude_floor"
        )
    return cfg


def _parse_normalization(raw: dict) -> NormalizationConfig:
    cfg = NormalizationConfig(
        method=str(raw["method"]),
        clip_after_norm=float(raw["clip_after_norm"]) if raw.get("clip_after_norm") is not None else None,
    )
    _require(cfg.method, ALLOWED_NORM_METHODS, "normalization.method")
    if cfg.clip_after_norm is not None and cfg.clip_after_norm <= 0:
        raise ValueError("normalization.clip_after_norm must be > 0 if set")
    return cfg


def _parse_ssl(raw: dict) -> SSLConfig:
    cfg = SSLConfig(
        encoder=str(raw["encoder"]),
        projection_dim=int(raw["projection_dim"]),
        embedding_dim=int(raw["embedding_dim"]),
        temperature=float(raw["temperature"]),
        epochs=int(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        lr=float(raw["lr"]),
        weight_decay=float(raw["weight_decay"]),
        lr_min=float(raw["lr_min"]),
        n_negatives=int(raw["n_negatives"]) if raw.get("n_negatives") is not None else None,
    )
    _require(cfg.encoder, ALLOWED_ENCODERS, "ssl.encoder")
    _require_positive(cfg.projection_dim, "ssl.projection_dim")
    _require_positive(cfg.embedding_dim, "ssl.embedding_dim")
    _require_positive(cfg.temperature, "ssl.temperature")
    _require_positive(cfg.epochs, "ssl.epochs")
    _require_positive(cfg.batch_size, "ssl.batch_size")
    _require_positive(cfg.lr, "ssl.lr")
    return cfg


def _parse_augment(raw: dict) -> AugmentConfig:
    cfg = AugmentConfig(
        gaussian_noise_std=float(raw["gaussian_noise_std"]),
        time_crop_min_frac=float(raw["time_crop_min_frac"]),
        subcarrier_mask_frac=float(raw["subcarrier_mask_frac"]),
        amplitude_scale_min=float(raw["amplitude_scale_min"]),
        amplitude_scale_max=float(raw["amplitude_scale_max"]),
    )
    _require_fraction(cfg.time_crop_min_frac, "augment.time_crop_min_frac")
    _require_fraction(cfg.subcarrier_mask_frac, "augment.subcarrier_mask_frac")
    if cfg.amplitude_scale_min > cfg.amplitude_scale_max:
        raise ValueError(
            "augment.amplitude_scale_min must be <= augment.amplitude_scale_max"
        )
    return cfg


def _parse_align(raw: dict) -> AlignConfig:
    cfg = AlignConfig(method=str(raw["method"]))
    _require(cfg.method, ALLOWED_ALIGN_METHODS, "align.method")
    return cfg


def _parse_routing(raw: dict) -> RoutingConfig:
    cfg = RoutingConfig(
        mode=str(raw["mode"]),
        confidence_threshold=float(raw["confidence_threshold"]),
    )
    _require(cfg.mode, ALLOWED_ROUTING_MODES, "routing.mode")
    _require_fraction(cfg.confidence_threshold, "routing.confidence_threshold")
    return cfg


def _parse_fewshot(raw: dict) -> FewshotConfig:
    cfg = FewshotConfig(
        K_list=[int(k) for k in raw["K_list"]],
        n_seeds=int(raw["n_seeds"]),
        distance_metric=str(raw["distance_metric"]),
        shrinkage=str(raw["shrinkage"]),
        classifier=str(raw["classifier"]),
    )
    if any(k <= 0 for k in cfg.K_list):
        raise ValueError("All values in fewshot.K_list must be > 0")
    _require_positive(cfg.n_seeds, "fewshot.n_seeds")
    _require(cfg.distance_metric, ALLOWED_DISTANCE_METRICS, "fewshot.distance_metric")
    _require(cfg.shrinkage, ALLOWED_SHRINKAGE, "fewshot.shrinkage")
    _require(cfg.classifier, ALLOWED_FS_CLASSIFIERS, "fewshot.classifier")
    return cfg


def _parse_activities(raw: dict) -> ActivitiesConfig:
    cfg = ActivitiesConfig(
        room_a=[str(a) for a in raw["room_a"]],
        room_b=[str(a) for a in raw["room_b"]],
        shared=[str(a) for a in raw["shared"]],
        room_a_exclusive=[str(a) for a in raw["room_a_exclusive"]],
        room_b_exclusive=[str(a) for a in raw["room_b_exclusive"]],
        all=[str(a) for a in raw["all"]],
    )
    # Validate every listed activity is in the master set
    for lst_name, lst in [
        ("room_a", cfg.room_a),
        ("room_b", cfg.room_b),
        ("shared", cfg.shared),
        ("room_a_exclusive", cfg.room_a_exclusive),
        ("room_b_exclusive", cfg.room_b_exclusive),
        ("all", cfg.all),
    ]:
        for act in lst:
            if act not in ALLOWED_ACTIVITIES:
                raise ValueError(
                    f"activities.{lst_name} contains unknown activity {act!r}; "
                    f"allowed: {sorted(ALLOWED_ACTIVITIES)}"
                )
    # Shared + exclusives must together equal the per-room lists
    if set(cfg.shared) | set(cfg.room_a_exclusive) != set(cfg.room_a):
        raise ValueError(
            "activities: shared ∪ room_a_exclusive must equal room_a"
        )
    if set(cfg.shared) | set(cfg.room_b_exclusive) != set(cfg.room_b):
        raise ValueError(
            "activities: shared ∪ room_b_exclusive must equal room_b"
        )
    return cfg


def _parse_rooms(raw: dict) -> RoomsConfig:
    cfg = RoomsConfig(all=[str(r) for r in raw["all"]])
    for r in cfg.all:
        _require(r, ALLOWED_ROOMS, "rooms.all")
    return cfg


def _parse_splits(raw: dict) -> SplitsConfig:
    cfg = SplitsConfig(
        all=[str(s) for s in raw["all"]],
        val_seed=int(raw["val_seed"]),
        val_captures_per_class=int(raw["val_captures_per_class"]),
    )
    for s in cfg.all:
        _require(s, ALLOWED_SPLITS, "splits.all")
    _require_positive(cfg.val_captures_per_class, "splits.val_captures_per_class")
    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """Load, parse, and validate *path* (YAML) into an ``ExperimentConfig``.

    Parameters
    ----------
    path:
        Filesystem path to ``experiment.yaml`` (or any override file).

    Returns
    -------
    ExperimentConfig
        Fully validated, frozen configuration object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any field fails validation.
    KeyError
        If a required field is missing from the YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    return ExperimentConfig(
        paths=_parse_paths(raw["paths"]),
        seeds=_parse_seeds(raw["seeds"]),
        window=_parse_window(raw["window"]),
        cleaning=_parse_cleaning(raw["cleaning"]),
        normalization=_parse_normalization(raw["normalization"]),
        ssl=_parse_ssl(raw["ssl"]),
        augment=_parse_augment(raw["augment"]),
        align=_parse_align(raw["align"]),
        routing=_parse_routing(raw["routing"]),
        fewshot=_parse_fewshot(raw["fewshot"]),
        activities=_parse_activities(raw["activities"]),
        rooms=_parse_rooms(raw["rooms"]),
        splits=_parse_splits(raw["splits"]),
    )
