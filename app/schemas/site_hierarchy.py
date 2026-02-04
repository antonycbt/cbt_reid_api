from pydantic import BaseModel, Field, validator
from typing import Optional, List


class SiteHierarchyCreate(BaseModel):
    name: str
    parent_site_hierarchy_id: Optional[int] = None
    is_active: bool = True

class SiteHierarchyUpdate(BaseModel):
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

    model_config = {"from_attributes": True}  # important for from_orm

SiteHierarchyNode.update_forward_refs()

class SiteHierarchyOut(SiteHierarchyNode):
    pass
