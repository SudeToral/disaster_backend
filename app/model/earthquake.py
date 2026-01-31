import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "earthquake_detection_v1.pt")

CONF_THRES = 0.25

def _compute_coverage_ratio(result) -> float:
    """
    YOLO segmentation mask'lerinden coverage_ratio hesaplar:
    toplam mask pikseli / (H*W)
    """
    r = result
    if r.masks is None or r.masks.data is None:
        return 0.0


    m = r.masks.data
    n, h, w = m.shape
    covered = float(m.sum().item())
    total = float(h * w)
    return covered / total if total > 0 else 0.0


def predict(image_path: str, save: bool = False, overlay_path: str | None = None) -> dict:
    """
    Returns standardized dict:
    - is_disaster (bool)
    - coverage_ratio (float)
    - damaged_regions (int)
    - produced_overlay (bool)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    # Ultralytics save mekanizmasını kullanmak yerine:
    # save=True ise kendi overlay_path'imize yazmak için results[0].plot() kullanacağız.
    results = model(image_path, conf=CONF_THRES, save=False)

    r = results[0]
    damaged_regions = len(r.masks) if r.masks is not None else 0
    coverage_ratio = _compute_coverage_ratio(r)
    is_disaster = damaged_regions > 0

    produced_overlay = False
    if save and overlay_path is not None:
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        # r.plot() -> numpy array (BGR)
        plotted = r.plot()
        # OpenCV yoksa PIL ile yazalım:
        try:
            from PIL import Image
            import numpy as np
            img = Image.fromarray(plotted[..., ::-1])  # BGR -> RGB
            img.save(overlay_path)
            produced_overlay = True
        except Exception:
            produced_overlay = False

    return {
        "is_disaster": is_disaster,
        "coverage_ratio": coverage_ratio,
        "damaged_regions": damaged_regions,
        "details": {
            "conf_thres": CONF_THRES
        },
        "produced_overlay": produced_overlay
    }
