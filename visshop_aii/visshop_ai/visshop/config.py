"""Configuration centrale (seuils du cahier des charges + paramètres runtime)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import Behavior

# Seuils par comportement (cahier des charges, section 1)
BEHAVIOR_THRESHOLDS: Dict[Behavior, float] = {
    Behavior.POCKET: 0.72,
    Behavior.UNDER_CLOTHING: 0.78,
    Behavior.BACKPACK: 0.70,
    Behavior.HANDBAG: 0.75,
}

DIFFICULTY: Dict[Behavior, str] = {
    Behavior.POCKET: "Moyen",
    Behavior.UNDER_CLOTHING: "Eleve",
    Behavior.BACKPACK: "Moyen",
    Behavior.HANDBAG: "Eleve",
}


@dataclass
class FilterConfig:
    """Filtre anti-fausses alertes a 4 niveaux (cahier des charges, section 2)."""

    smoothing_window: int = 20        # niveau 2 : fenetre glissante (echantillons analyses)
    trigger_threshold: float = 0.68   # niveau 3 : seuil de declenchement (hysteresis)
    cancel_threshold: float = 0.35    # niveau 3 : seuil d'annulation
    cooldown_s: float = 3.0           # niveau 4 : une alerte max par 3 s et par personne
    min_samples: int = 8              # echantillons minimum avant tout declenchement
    min_shelf_dwell_s: float = 1.0    # niveau 1 : duree de presence minimale au rayon
    shelf_memory_s: float = 8.0       # niveau 1 : validite d'une visite au rayon
    staff_ratio: float = 0.35         # niveau 1 : part de pixels "uniforme" dans le torse
    staff_votes_window: int = 10
    staff_vote_fraction: float = 0.6


@dataclass
class ZoneConfig:
    """Polygones de zones. Coordonnees normalisees [0,1] si normalized=True."""

    shelf: List[List[List[float]]] = field(default_factory=list)
    fitting_room: List[List[List[float]]] = field(default_factory=list)
    normalized: bool = False

    def scaled(self, width: int, height: int) -> "ZoneConfig":
        if not self.normalized:
            return self

        def sc(polys):
            return [[[x * width, y * height] for x, y in poly] for poly in polys]

        return ZoneConfig(sc(self.shelf), sc(self.fitting_room), normalized=False)


@dataclass
class StaffConfig:
    """Plages HSV (OpenCV : H 0-179) de l'uniforme du personnel. Vide = desactive."""

    hsv_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = field(
        default_factory=list
    )


@dataclass
class RuntimeConfig:
    frame_skip: Optional[int] = None  # None = automatique (GPU 3 / CPU 5)
    adaptive_skip: bool = True
    max_poses_per_cycle: int = 8
    detection_conf: float = 0.4
    camera_id: str = "cam01"
    save_screenshots: bool = True
    bag_memory_s: float = 1.5         # un sac non detecte est reutilise pendant 1.5 s


@dataclass
class Config:
    thresholds: Dict[Behavior, float] = field(
        default_factory=lambda: dict(BEHAVIOR_THRESHOLDS)
    )
    filter: FilterConfig = field(default_factory=FilterConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    staff: StaffConfig = field(default_factory=StaffConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        cfg = cls()
        for key, val in data.get("thresholds", {}).items():
            cfg.thresholds[Behavior(key)] = float(val)
        cfg.filter = FilterConfig(**data.get("filter", {}))
        z = data.get("zones", {})
        cfg.zones = ZoneConfig(
            shelf=z.get("shelf", []),
            fitting_room=z.get("fitting_room", []),
            normalized=z.get("normalized", False),
        )
        cfg.staff = StaffConfig(
            hsv_ranges=[
                (tuple(lo), tuple(hi))
                for lo, hi in data.get("staff", {}).get("hsv_ranges", [])
            ]
        )
        cfg.runtime = RuntimeConfig(**data.get("runtime", {}))
        return cfg


def load_config(path: Optional[str] = None) -> Config:
    if not path:
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        return Config.from_dict(json.load(f))
