import os
from ultralytics import YOLO

# Dosyanın bulunduğu klasöre göre model yolunu çöz
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "flood_detection_v1.pt")

CONF_THRES = 0.25

_flood_model = None


def _get_model():
    """Lazy-load YOLO model (server açılırken değil, ilk istekte yüklenir)."""
    global _flood_model
    if _flood_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Flood model not found: {MODEL_PATH}")
        _flood_model = YOLO(MODEL_PATH)
    return _flood_model


def _compute_coverage_ratio(result) -> float:
    """
    YOLO segmentation mask'lerinden coverage_ratio hesaplar:
    toplam mask pikseli / (H*W)
    """
    r = result
    if r.masks is None or r.masks.data is None:
        return 0.0

    m = r.masks.data  # (N, H, W) torch tensor
    _, h, w = m.shape
    covered = float(m.sum().item())
    total = float(h * w)
    return covered / total if total > 0 else 0.0


def predict(image_path: str, save: bool = False, overlay_path: str | None = None) -> dict:
    """
    Flood segmentation inference.

    Returns standardized dict:
    - is_disaster (bool)
    - coverage_ratio (float)
    - damaged_regions (int)
    - details (dict)
    - produced_overlay (bool)
    """
    print(f"[FLOOD] Running model on image: {image_path}")

    model = _get_model()

    # Ultralytics'in kendi save mekanizmasını kullanmıyoruz:
    # overlay'i overlay_path'e kendimiz yazacağız.
    results = model(image_path, conf=CONF_THRES, save=False)

    r = results[0]

    damaged_regions = len(r.masks) if r.masks is not None else 0
    coverage_ratio = _compute_coverage_ratio(r)

    # Basit karar: en az 1 mask varsa disaster
    # İstersen coverage_ratio threshold ile daha sağlam yaparız
    is_disaster = damaged_regions > 0

    produced_overlay = False
    if save and overlay_path is not None:
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        plotted = r.plot()  # numpy array (BGR)
        try:
            from PIL import Image
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
            "conf_thres": CONF_THRES,
            "model_path": MODEL_PATH
        },
        "produced_overlay": produced_overlay
    }
