from pathlib import Path
import os

CAM_LOCATION_TYPES = {
    1: "Low light area",
    2: "Medium light area", 
    3: "Bright light area", 
}
 
USER_ROLES = { 
    1: "Admin", 
    2: "Primary Admin", 
    3: "Super Admin", 
}

# --------------------------------
# EMBEDDING CONSTANTS
# --------------------------------

# --- EDIT / override with env vars ---
RTSP_STREAMS = [
   "rtsp://admin:rolex%40123@192.168.1.111:554/Streaming/channels/101"
    
]


YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
DEVICE = os.getenv("DEVICE", "cuda:0")
CONF_THRES = float(os.getenv("CONF_THRES", 0.80))
IOU_THRES = float(os.getenv("IOU_THRES", 0.40))


EMB_CSV = os.getenv("EMB_CSV", "live_embeddings_multi.csv")
CROPS_ROOT = os.getenv("CROPS_ROOT", "crops")
EMB_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:1234@192.168.1.136:5432/m_vision",
)


# locks / simple runtime globals live inside services/embedding_service.py


# ensure crops dir exists by default
Path(CROPS_ROOT).mkdir(parents=True, exist_ok=True) 