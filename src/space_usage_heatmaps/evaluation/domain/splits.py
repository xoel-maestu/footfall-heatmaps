"""Scene splits of NVIDIA MTMC_Tracking_2024 (spec 5), one camera per scene for E1.

Each scene's camera is its calibrated camera with the most ground-truth rows
(visible boxes) in ground_truth.txt. Every camera of 076-080 is calibrated.
The sealed split is scored only to confirm what was accepted on tune: never
tune on it.
"""

SPLITS = {
    "tune": (("071", "645"), ("072", "663"), ("073", "677"), ("074", "695"), ("075", "709")),
    "sealed": (("076", "723"), ("077", "734"), ("078", "753"), ("079", "777"), ("080", "784")),
}
