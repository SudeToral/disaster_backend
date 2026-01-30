import os
import random
from typing import Tuple, Dict

def get_mock_satellite_image(coordinates: Tuple[float, float]) -> str:
    """
    Emulates fetching a satellite image for given coordinates.
    Returns the path to a mock image from the 'images' folder.
    """
    images_folder = os.path.join(os.path.dirname(__file__), "../../images")
    if not os.path.exists(images_folder):
        raise FileNotFoundError(f"Images folder not found at {images_folder}")
    
    images = os.listdir(images_folder)
    if not images:
        raise FileNotFoundError("No images found in the 'images' folder.")
    
    # Randomly select an image to emulate satellite response
    selected_image = random.choice(images)
    return os.path.join(images_folder, selected_image)
# app/model/model.py
from typing import Dict

def predict_with_type(image_path: str, disaster_type: str, save: bool = False, overlay_path: str | None = None) -> Dict:
    disaster_type = disaster_type.lower().strip()

    if disaster_type == "deprem":
        from .earthquake import predict as predict_deprem
        return predict_deprem(image_path, save=save, overlay_path=overlay_path)

    elif disaster_type == "sel":
        from .flood import predict as predict_sel
        # flood predict imzan farklıysa:
        try:
            return predict_sel(image_path, save=save, overlay_path=overlay_path)
        except TypeError:
            return predict_sel(image_path)

    elif disaster_type == "yangin":
        from .fire import predict as predict_yangin
        try:
            return predict_yangin(image_path, save=save, overlay_path=overlay_path)
        except TypeError:
            return predict_yangin(image_path)

    else:
        raise ValueError(f"Unknown disaster type: {disaster_type}")
