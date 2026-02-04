from pydantic import BaseModel, Field, validator, conint
from typing import Optional, List


# -------------------------
# CREATE
# -------------------------
class MemberCreate(BaseModel):
    member_number: str = Field(..., min_length=1, max_length=16)
    first_name: str = Field(..., min_length=1, max_length=64)
    last_name: str = Field(..., min_length=1, max_length=64)
    department_id: Optional[int] = None
    is_active: Optional[bool] = True

    @validator("member_number", "first_name", "last_name")
    def field_cannot_be_blank(cls, v: str):
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


# -------------------------
# UPDATE
# -------------------------
class MemberUpdate(BaseModel):
    member_number: Optional[str] = Field(None, min_length=1, max_length=16)
    first_name: Optional[str] = Field(None, min_length=1, max_length=64)
    last_name: Optional[str] = Field(None, min_length=1, max_length=64)
    department_id: Optional[int] = None
    is_active: Optional[bool] = None

    @validator("member_number", "first_name", "last_name")
    def field_cannot_be_blank(cls, v: Optional[str]):
        if v is not None and not v.strip():
            raise ValueError("must not be empty")
        return v.strip() if v else v


# -------------------------
# DEPARTMENT (MINIMAL)
# -------------------------
class DepartmentOutMinimal(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


# -------------------------
# ACCESS GROUP (MINIMAL)
# -------------------------
class AccessGroupOutMinimal(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


# -------------------------
# OUTPUT
# -------------------------
class MemberOut(BaseModel):
    id: int
    member_number: str
    first_name: str
    last_name: str
    is_active: bool

    department_id: Optional[int]

    # relationships
    department: Optional[DepartmentOutMinimal] = None
    access_groups: List[AccessGroupOutMinimal] = []

    class Config:
        from_attributes = True 


