from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
from sqlalchemy.orm import Session

from app.schemas.access_group import (
    AccessGroupCreate,
    AccessGroupUpdate,
    AccessGroupOut
)
from app.schemas.member import (
    MemberOut
)
from app.services.access_group_service import AccessGroupService
from app.schemas.common import MessageResponse
from app.db.session import get_db

router = APIRouter()


# ---------- STATIC ROUTES FIRST ----------

@router.get(
    "/list_unlinked_members_by_access_groups",
    response_model=MessageResponse[List[MemberOut]],
)
def list_unlinked_members_by_access_groups(
    access_group_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    members = AccessGroupService.list_unlinked_members_by_access_groups(
        db, access_group_id
    )
    return {
        "message": "Members fetched successfully",
        "data": members,
        "total": len(members),
    }


@router.get("/all", response_model=MessageResponse[List[AccessGroupOut]])
def get_all_access_groups(db: Session = Depends(get_db)):
    groups, total = AccessGroupService.list_access_groups(
        search=None,
        page=0,
        page_size=1000000
    )
    return {
        "message": "Access groups fetched successfully",
        "data": groups,
        "total": len(groups),
    } 
  
# ---------- LIST (search + pagination) ----------

@router.get("", response_model=MessageResponse[List[AccessGroupOut]])
def list_access_groups(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=1000),
):
    groups, total = AccessGroupService.list_access_groups(search, page, page_size)
    return {
        "message": "Access groups fetched successfully",
        "data": groups,
        "total": total,
    }


# ---------- CREATE ----------

@router.post("", response_model=MessageResponse[AccessGroupOut])
def create_access_group(payload: AccessGroupCreate):
    group = AccessGroupService.create_access_group(payload)
    return {
        "message": "Access group created successfully",
        "data": group,
    }


# ---------- DYNAMIC ROUTES LAST ----------

@router.get("/{access_group_id}", response_model=MessageResponse[AccessGroupOut])
def get_access_group(access_group_id: int):
    group = AccessGroupService.get_access_group(access_group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Access group not found")

    return {
        "message": "Access group fetched successfully",
        "data": group,
    }


@router.put("/{access_group_id}", response_model=MessageResponse[AccessGroupOut])
def update_access_group(access_group_id: int, payload: AccessGroupUpdate):
    group = AccessGroupService.update_access_group(access_group_id, payload)
    return {
        "message": "Access group updated successfully",
        "data": group,
    }


@router.delete("/{access_group_id}", response_model=MessageResponse[None])
def delete_access_group(access_group_id: int):
    success = AccessGroupService.delete_access_group(access_group_id)
    if not success:
        raise HTTPException(status_code=404, detail="Access group not found")

    return {
        "message": "Access group deleted permanently",
    }
