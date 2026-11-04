"""Scene splits of NVIDIA MTMC_Tracking_2024 (spec 5), one camera per scene for E1.

Each scene's camera is its calibrated camera with the most ground-truth rows
(visible boxes) in ground_truth.txt.
"""

SPLITS = {
    "tune": (("071", "645"), ("072", "663"), ("073", "677"), ("074", "695"), ("075", "709")),
}
