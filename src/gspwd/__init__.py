"""GSPWD: Generalized Spherical Plane-Wave Decomposition.

Direction-of-arrival estimation from spherical-harmonic-domain signals using optimised zonal
kernels and a joint orthogonal matching pursuit across harmonic orders.
"""

from .experiments import ExperimentConfig, run_experiment, run_trial
from .kernels import (
    gpulse,
    maxdfcoeff,
    zonal_multiobjective_coeff,
)
from .metrics import (
    amplitude_errors,
    detectionmetrics,
    ecm,
    le,
    lr,
    support_aware_amplitude_error,
)
from .omp import omp_multi_dict_shared
from .scenes import planewave, randomscene_rejection, scene_shd
from .sphere import acn_index, gridangs, reorderalms
from .srf import generalized_srf, generalizeddict, pwd_dictionaries

__version__ = "0.1.0.dev0"

__all__ = [
    "ExperimentConfig",
    "run_experiment",
    "run_trial",
    "gpulse",
    "maxdfcoeff",
    "zonal_multiobjective_coeff",
    "amplitude_errors",
    "detectionmetrics",
    "ecm",
    "le",
    "lr",
    "support_aware_amplitude_error",
    "omp_multi_dict_shared",
    "planewave",
    "randomscene_rejection",
    "scene_shd",
    "acn_index",
    "gridangs",
    "reorderalms",
    "generalized_srf",
    "generalizeddict",
    "pwd_dictionaries",
]
