from fastapi import APIRouter, Query, HTTPException
from typing import Optional

from app.schemas.site_location_access import SiteLocationAccessCreate, SiteLocationAccessBulkCreate
from app.schemas.common import MessageResponse
from app.services.site_location_access_service import SiteLocationAccessService 

router = APIRouter()
# -------- CREATE SINGLE --------
@router.post("", response_model=MessageResponse[None])
def create_site_location_access(payload: SiteLocationAccessCreate):
    SiteLocationAccessService.create_site_location_access(payload)
    return {"message": "Access assigned successfully"}


# -------- BULK ASSIGN --------
@router.post("/bulk", response_model=MessageResponse[dict])
def bulk_assign_site_location_access(payload: SiteLocationAccessBulkCreate):
    created = SiteLocationAccessService.bulk_assign_access(payload)
    return {
        "message": f"{created['created_count']} access group(s) assigned successfully",
        "data": created,
    }


# -------- LIST --------
@router.get("", response_model=MessageResponse[list])
def list_site_location_access(
    search: Optional[str] = Query(None),
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1)
):
    data, total = SiteLocationAccessService.list_site_location_access(search, page, page_size)
    return {"message": "Access fetched successfully", "data": data, "total": total}


# -------- DELETE ALL ACCESS GROUPS FOR SITE LOCATION --------
@router.delete("/all", response_model=MessageResponse)
def delete_all_access_groups(site_location_id: int = Query(..., description="ID of the site location")):
    SiteLocationAccessService.delete_all_for_site_location(site_location_id)
    return {"message": "All access groups removed"}


# -------- DELETE SINGLE ACCESS GROUP --------
@router.delete("/single", response_model=MessageResponse)
def delete_single_access_group(
    site_location_id: int = Query(..., description="ID of the site location"),
    access_group_id: int = Query(..., description="ID of the access group"),
):
    """
    Delete a single access group from a site location
    """
    try:
        SiteLocationAccessService.delete_single_access(site_location_id, access_group_id)
        return {"message": "Access group removed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
