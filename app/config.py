from app.store import data_store

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 30

FALLBACK_ENABLED = True


def load_hospitals():
    """Load all hospitals from the data store."""
    return data_store.get_all_hospitals()


def load_resources():
    """Load resource types and inventory from the data store."""
    return {
        "resource_types": data_store.resource_types,
        "inventory": data_store.resource_inventory,
    }


def load_hospitals_with_resources():
    """Load hospitals with merged resource inventory from the data store."""
    return data_store.get_hospitals_with_resources()


def detect_terrain(lat: float, lon: float) -> str:
    """Detect terrain type from coordinates using the data store."""
    region = data_store.get_region_by_coords(lat, lon)
    return region["terrain_type"] if region else "urban"
