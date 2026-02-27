from pathlib import Path
import os
 
USER_ROLES = { 
    1: "Admin", 
    2: "Primary Admin", 
    3: "Super Admin", 
}

TARGET_TYPE = {
    1: "users" ,
    2: "departments" ,
    3: "site_hierarchies" ,
    4: "cameras" ,
    5: "access_groups" ,
    6: "members" ,
    7: "notifications" ,
    8: "site_location_access" ,
    9: "member_access"
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