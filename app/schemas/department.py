from pydantic import BaseModel, Field, validator
from typing import Optional


class DepartmentCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)

    @validator("name")
    def name_cannot_be_blank(cls, v: str):
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class DepartmentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    is_active: Optional[bool] = None

    @validator("name")
    def name_cannot_be_blank(cls, v: Optional[str]):
        if v is not None and not v.strip():
            raise ValueError("must not be empty")
        return v.strip() if v else v


class DepartmentOut(BaseModel):
    id: int
    name: str
    is_active: bool

    class Config:
        from_attributes = True
