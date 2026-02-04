from pydantic import BaseModel, Field, validator
from typing import Optional


class CameraCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    ip_address: str = Field(..., min_length=7, max_length=16)
    site_location_id: int
    location_type: int = Field(..., description="1=low light, 2=high traffic")

    @validator("name")
    def name_cannot_be_blank(cls, v: str):
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @validator("ip_address")
    def ip_not_blank(cls, v: str):
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=64)
    ip_address: Optional[str] = Field(None, min_length=7, max_length=16)
    site_location_id: Optional[int] = None
    location_type: Optional[int] = None
    is_active: Optional[bool] = None

    @validator("name")
    def name_cannot_be_blank(cls, v: Optional[str]):
        if v is not None and not v.strip():
            raise ValueError("must not be empty")
        return v.strip() if v else v

    @validator("ip_address")
    def ip_not_blank(cls, v: Optional[str]):
        if v is not None and not v.strip():
            raise ValueError("must not be empty")
        return v.strip() if v else v

class SiteLocationMini(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True
 
class CameraOut(BaseModel):
    id: int
    name: str
    ip_address: str 
    location_type: int
    is_active: bool
    site_location: SiteLocationMini

    class Config:
        orm_mode = True
