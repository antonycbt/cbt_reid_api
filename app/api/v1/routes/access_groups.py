from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional

from app.schemas.access_group import (
    AccessGroupCreate,
    AccessGroupUpdate,
    AccessGroupOut,
)
from app.services.access_group_service import AccessGroupService
from app.schemas.common import MessageResponse
from sqlalchemy.orm import Session
from app.db.session import get_db
from typing import List

router = APIRouter()


@router.get(
    "/list_unlinked_access_groups_by_member",
    response_model=MessageResponse[List[AccessGroupOut]],
)
def list_unlinked_access_groups_by_member(
    member_id: Optional[int] = Query(None, description="Filter out linked access groups"),
    db: Session = Depends(get_db),
):
    access_groups = AccessGroupService.list_unlinked_access_groups_by_member(db, member_id)
    return {
        "message": "Access groups fetched successfully",
        "data": access_groups,
        "total": len(access_groups),
    }

@router.get(
    "/list_unlinked_access_groups_by_site_location",
    response_model=MessageResponse[List[AccessGroupOut]],
)
def list_unlinked_access_groups_by_site_location(
    site_location_id: Optional[int] = Query(None, description="Filter out linked access groups"),
    db: Session = Depends(get_db),
):
    access_groups = AccessGroupService.list_unlinked_access_groups_by_site_location(db, site_location_id)
    return {
        "message": "Access groups fetched successfully",
        "data": access_groups,
        "total": len(access_groups),
    }

# LIST access groups (search + pagination)
@router.get("", response_model=MessageResponse[list[AccessGroupOut]])
def list_access_groups(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=1000),  # ✅ FIX HERE
):
    groups, total = AccessGroupService.list_access_groups(
        search, page, page_size
    )
    return {
        "message": "Access groups fetched successfully",
        "data": groups,
        "total": total,
    }

# CREATE access group
@router.post("", response_model=MessageResponse[AccessGroupOut])
def create_access_group(payload: AccessGroupCreate):
    group = AccessGroupService.create_access_group(payload)
    return {
        "message": "Access group created successfully",
        "data": group,
    }

# GET access group by ID
@router.get("/{access_group_id}", response_model=MessageResponse[AccessGroupOut])
def get_access_group(access_group_id: int):
    group = AccessGroupService.get_access_group(access_group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Access group not found")

    return {
        "message": "Access group fetched successfully",
        "data": group,
    }

# UPDATE access group
@router.put("/{access_group_id}", response_model=MessageResponse[AccessGroupOut])
def update_access_group(
    access_group_id: int,
    payload: AccessGroupUpdate,
):
    group = AccessGroupService.update_access_group(
        access_group_id, payload
    )
    return {
        "message": "Access group updated successfully",
        "data": group,
    }

# HARD DELETE (permanent)
@router.delete("/{access_group_id}", response_model=MessageResponse[None])
def delete_access_group(access_group_id: int):
    success = AccessGroupService.delete_access_group(access_group_id)
    if not success:
        raise HTTPException(status_code=404, detail="Access group not found")

    return {
        "message": "Access group deleted permanently",
    }


