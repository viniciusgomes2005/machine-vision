"""Configuration objects for seedling inspection pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CalibrationConfig:
    """Holds optional pixel-to-metric calibration parameters."""

    enabled: bool = False
    reference_object: Optional[str] = None
    reference_diameter_mm: Optional[float] = None


@dataclass
class SegmentationConfig:
    """Color and morphology thresholds for mask extraction."""

    blue_hue_range: Tuple[int, int] = (85, 130)
    blue_s_min: int = 70
    blue_v_min: int = 40

    black_v_max: int = 75
    black_s_max: int = 120

    white_v_min: int = 180
    white_s_max: int = 70

    plant_hue_range: Tuple[int, int] = (25, 95)
    plant_s_min: int = 25
    plant_v_min: int = 30

    min_component_area_px: int = 200

    open_kernel: int = 3
    close_kernel: int = 5


@dataclass
class MeasurementConfig:
    """Parameters controlling morphology measurements."""

    collar_search_above_px: int = 100
    collar_search_below_px: int = 20
    collar_center_window_px: int = 140

    length_skeleton_min_size_px: int = 300

    collar_band_height_px: int = 110
    collar_band_offset_px: int = 15
    collar_samples_min: int = 5

    eucalyptus_leaf_min_area_px: int = 450
    pine_leaf_min_area_px: int = 120

    watershed_peak_min_distance: int = 18
    watershed_peak_threshold_rel: float = 0.25


@dataclass
class AppConfig:
    """Top-level app configuration."""

    save_debug: bool = True
    save_annotated: bool = True

    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
