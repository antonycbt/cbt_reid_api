from pydantic import BaseModel, Field, validator
from typing import Optional, List
from app.schemas.base import TrimmedModel 
# -------------------------
# CREATE
# -------------------------
class SiteHierarchyCreate(TrimmedModel):
    name: str
    parent_site_hierarchy_id: Optional[int] = None
    is_active: bool = True


# -------------------------
# UPDATE
# -------------------------
class SiteHierarchyUpdate(TrimmedModel):
    name: Optional[str]
    parent_site_hierarchy_id: Optional[int]
    is_active: Optional[bool]


# -------------------------
# OUTPUT
# -------------------------
class SiteHierarchyNode(BaseModel):
    id: int
    name: str
    parent_site_hierarchy_id: Optional[int]
    is_active: bool
    children: List["SiteHierarchyNode"] = []
    is_locked: bool = False
    model_config = {"from_attributes": True}

SiteHierarchyNode.update_forward_refs()


class SiteHierarchyOut(SiteHierarchyNode):
    pass
