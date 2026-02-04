from pydantic import BaseModel, Field, conint
from typing import Optional, List, Tuple 

# ---------- Schemas ----------
class MemberCameraRequest(BaseModel):
    member_id: conint(gt=0) = Field(..., description="Member ID to capture/extract for")
    camera_id: conint(gt=0) = Field(..., description="Camera ID to capture/extract from")
    name: str = Field("unknown", description="Member name for folder/DB labeling")


class StartResponse(BaseModel):
    status: str
    num_cams: int
    member_id: int
    camera_id: int
    name: str


class ExtractStartResponse(BaseModel):
    status: str
    member_id: int
    camera_id: int
    name: str


class ProgressResponse(BaseModel):
    member_id: int
    camera_id: int
    stage: str
    percent: int
    message: str
    total_body: int
    total_face: int
    done_body: int
    done_face: int


class StopRequest(BaseModel):
    # Allow stopping a specific member/camera job, or stop everything if omitted
    member_id: Optional[conint(gt=0)] = Field(None, description="Member ID to stop (optional)")
    camera_id: Optional[conint(gt=0)] = Field(None, description="Camera ID to stop (optional)")


class StopResponse(BaseModel):
    status: str
    member_id: Optional[int] = None
    camera_id: Optional[int] = None


class ExtractionStatus(BaseModel):
    running: bool
    num_cams: int
    rtsp_streams: List[str]
    member_id: Optional[int] = None
    camera_id: Optional[int] = None

