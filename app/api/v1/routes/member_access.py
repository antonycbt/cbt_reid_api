from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from app.schemas.member_access import MemberAccessCreate, MemberAccessBulkCreate
from app.schemas.common import MessageResponse
from app.services.member_access_service import MemberAccessService 

router = APIRouter()
# -------- CREATE SINGLE --------
@router.post("", response_model=MessageResponse[None])
def create_member_access(payload: MemberAccessCreate):
    MemberAccessService.create_member_access(payload)
    return {"message": "Access assigned successfully"}


# -------- BULK ASSIGN --------
@router.post("/bulk", response_model=MessageResponse[dict])
def bulk_assign_member_access(payload: MemberAccessBulkCreate):
    created = MemberAccessService.bulk_assign_access(payload)
    return {
        "message": f"{created['created_count']} access group(s) assigned successfully",
        "data": created,
    }


# -------- LIST --------
@router.get("", response_model=MessageResponse[list])
def list_member_access(
    search: Optional[str] = Query(None),
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1)
):
    data, total = MemberAccessService.list_member_access(search, page, page_size)
    return {"message": "Access fetched successfully", "data": data, "total": total}


# -------- DELETE ALL ACCESS GROUPS FOR MEMBER ACCESS --------
@router.delete("/all", response_model=MessageResponse)
def delete_all_access_groups(member_id: int = Query(..., description="ID of the member access")):
    MemberAccessService.delete_all_for_site_location(member_id)
    return {"message": "All access groups removed"}


# -------- DELETE SINGLE ACCESS GROUP --------
@router.delete("/single", response_model=MessageResponse)
def delete_single_access_group(
    member_id: int = Query(..., description="ID of the member access"),
    access_group_id: int = Query(..., description="ID of the access group"),
):
    """
    Delete a single access group from a member access
    """
    try:
        MemberAccessService.delete_single_access(member_id, access_group_id)
        return {"message": "Access group removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
