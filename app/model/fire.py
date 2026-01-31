import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# =========================
# MODEL CONFIG
# =========================
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "fire_detection_v1.h5")
TARGET_SIZE = (224, 224)  # (W, H) for cv2.resize
LABELS = ["nowildfire", "wildfire"]  # index 1 = wildfire

# PSEUDO-COVERAGE CONFIG (Grad-CAM)
# "Top 20% hottest pixels" => coverage
HEATMAP_TOP_QUANTILE = 0.80  # 0.80 => top 20%
HEATMAP_MIN_ENERGY = 1e-6    # safety

_fire_model = None
_last_conv_layer_name = None


# =========================
# MODEL (LAZY LOAD)
# =========================
def _get_model():
    global _fire_model, _last_conv_layer_name

    if _fire_model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Fire model not found: {MODEL_PATH}")

        _fire_model = load_model(MODEL_PATH, compile=False)

        conv_layers = [
            l.name for l in _fire_model.layers
            if isinstance(l, tf.keras.layers.Conv2D)
        ]
        if not conv_layers:
            raise RuntimeError("No Conv2D layer found for Grad-CAM.")
        _last_conv_layer_name = conv_layers[-1]

        print("[FIRE] Loaded model:", MODEL_PATH)
        print("[FIRE] Grad-CAM layer:", _last_conv_layer_name)

    return _fire_model


# =========================
# PREPROCESS (BGR->RGB, 224, /255)
# =========================
def _preprocess(image_path: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Image could not be read: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, TARGET_SIZE)
    img_input = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=0)  # (1,224,224,3)

    return img_bgr, img_input


def _predict_scores(model, img_input: np.ndarray) -> np.ndarray:
    """
    Handles models that may expect nested inputs (Expected [['input']]).
    Returns scores for a single sample: shape (K,)
    """
    try:
        out = model.predict(img_input, verbose=0)
    except Exception:
        out = model.predict([[img_input]], verbose=0)

    if isinstance(out, (list, tuple)):
        out = out[0]

    out = np.array(out)
    return out[0]  # (K,)


# =========================
# GRAD-CAM (nested input safe)
# =========================
def _get_gradcam_heatmap(img_input: np.ndarray, model, last_conv_layer_name: str, pred_index: int) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    def _forward(x):
        try:
            conv_out, preds = grad_model(x, training=False)
        except Exception:
            conv_out, preds = grad_model([[x]], training=False)

        if isinstance(conv_out, (list, tuple)):
            conv_out = conv_out[0]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        return conv_out, preds

    with tf.GradientTape() as tape:
        conv_out, preds = _forward(img_input)
        tape.watch(conv_out)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients are None. Check layer selection / model graph.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_out = conv_out[0]  # (H,W,C)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # (H,W)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()  # 0..1


def _heatmap_to_pseudo_coverage(heatmap_01: np.ndarray) -> float:
    """
    Percentile-based pseudo coverage:
    coverage = fraction of pixels in the top (1-quantile) hottest region.
    More stable than fixed threshold.
    """
    if heatmap_01 is None or heatmap_01.size == 0:
        return 0.0

    # If heatmap has almost no energy, don't produce fake coverage
    if float(np.max(heatmap_01)) < HEATMAP_MIN_ENERGY:
        return 0.0

    t = float(np.quantile(heatmap_01, HEATMAP_TOP_QUANTILE))  # e.g. 80th percentile
    mask = heatmap_01 >= t
    return float(mask.mean())


# =========================
# MAIN PREDICT
# =========================
def predict(image_path: str, save: bool = False, overlay_path: str | None = None) -> dict:
    """
    Fire classification + Grad-CAM product overlay.
    Product rule: overlay is produced ONLY if predicted label == 'wildfire' AND save=True.

    Also returns:
    - risk_score: wildfire probability/confidence
    - coverage_ratio: pseudo-coverage from Grad-CAM (only meaningful for wildfire)
    """
    print(f"[FIRE] Running model on image: {image_path}")
    print("[FIRE] save:", save, "overlay_path:", overlay_path)

    model = _get_model()
    img_bgr, img_input = _preprocess(image_path)

    # ---- Prediction ----
    scores = _predict_scores(model, img_input)
    class_idx = int(np.argmax(scores))
    confidence = float(scores[class_idx])

    label = LABELS[class_idx] if class_idx < len(LABELS) else str(class_idx)
    is_disaster = (label == "wildfire")

    # "risk_score" = wildfire olasılığı gibi kullanacağız
    # İki sınıf varsayımıyla wildfire score'u: scores[1]
    # Eğer modelin class order'ı değişirse burada güncelleriz.
    wildfire_score = float(scores[1]) if len(scores) > 1 else confidence
    risk_score = wildfire_score

    produced_overlay = False
    coverage_ratio = 0.0  # pseudo coverage (Grad-CAM)

    # ---- If wildfire -> compute Grad-CAM once (for coverage and possibly overlay)
    if is_disaster:
        try:
            heatmap = _get_gradcam_heatmap(
                img_input=img_input,
                model=model,
                last_conv_layer_name=_last_conv_layer_name,
                pred_index=class_idx,
            )
            coverage_ratio = _heatmap_to_pseudo_coverage(heatmap)

            # Only save overlay if requested
            if save and overlay_path:
                heatmap_vis = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
                heatmap_vis = np.uint8(255 * heatmap_vis)
                heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

                overlay_img = cv2.addWeighted(img_bgr, 0.6, heatmap_vis, 0.4, 0)

                text = f"{label} ({confidence*100:.2f}%) | cov~{coverage_ratio*100:.1f}%"
                cv2.putText(
                    overlay_img,
                    text,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                ok = cv2.imwrite(overlay_path, overlay_img)

                print("[FIRE] overlay_path:", overlay_path, "write_ok:", ok)
                print("[FIRE] overlay_exists_after:", os.path.exists(overlay_path),
                      "size:", os.path.getsize(overlay_path) if os.path.exists(overlay_path) else None)

                produced_overlay = bool(ok)

        except Exception as e:
            print("[FIRE] Grad-CAM/coverage/overlay failed:", str(e))
            coverage_ratio = 0.0
            produced_overlay = False

    else:
        if save and overlay_path:
            print("[FIRE] Overlay skipped (not wildfire).")

    return {
        "is_disaster": bool(is_disaster),
        "risk_score": float(risk_score),          # 0..1
        "coverage_ratio": float(coverage_ratio),  # 0..1 (pseudo spatial coverage)
        "damaged_regions": 1 if is_disaster else 0,
        "details": {
            "predicted_label": label,
            "confidence": float(confidence),
            "raw_scores": scores.tolist(),
            "model_type": "keras_classification_gradcam",
            "overlay_policy": "only_if_wildfire",
            "coverage_method": f"gradcam_top_{int((1-HEATMAP_TOP_QUANTILE)*100)}pct",
            "heatmap_top_quantile": HEATMAP_TOP_QUANTILE,
        },
        "produced_overlay": produced_overlay,
    }
