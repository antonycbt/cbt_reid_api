from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.schemas.member import (
    MemberCreate,
    MemberUpdate,
    MemberOut,
)
from app.services.member_service import MemberService
from app.schemas.common import MessageResponse 
from typing import Optional 



router = APIRouter()


# LIST all site locations (for dropdowns, no pagination)
@router.get("/all")
def list_all_members(
    search: str | None = None,
    page: int = 0,
    page_size: int = 10,
):
    members, total = MemberService.list_members(
        search=search,
        page=page,
        page_size=page_size,
    )

    return {
        "data": members,
        "total": total,
    }
# -------------------------
# LIST members (search + pagination)
# -------------------------
@router.get("")
def list_members(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),     
    page_size: int = Query(10, ge=1, le=1000),
):
    members, total = MemberService.list_members(
        search, page, page_size
    )
    return {
        "message": "Members fetched successfully",
        "data": members,
        "total": total,
    }


# -------------------------
# CREATE member
# -------------------------
@router.post("", response_model=MessageResponse[MemberOut])
def create_member(payload: MemberCreate):
    member = MemberService.create_member(payload)
    return {
        "message": "Member created successfully",
        "data": member,
    }


# -------------------------
# GET member by ID
# -------------------------
@router.get("/{member_id}", response_model=MessageResponse[MemberOut])
def get_member(member_id: int):
    member = MemberService.get_member(member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    return {
        "message": "Member fetched successfully",
        "data": member,
    }
 
# -------------------------
# UPDATE member
# -------------------------
@router.put("/{member_id}", response_model=MessageResponse[MemberOut])
def update_member(
    member_id: int,
    payload: MemberUpdate,
):
    member = MemberService.update_member(
        member_id, payload
    )
    return {
        "message": "Member updated successfully",
        "data": member,
    }


# -------------------------
# HARD DELETE (permanent)
# -------------------------
@router.delete("/{member_id}", response_model=MessageResponse[None])
def delete_member(member_id: int):
    success = MemberService.delete_member(member_id)
    if not success:
        raise HTTPException(status_code=404, detail="Member not found")

    return {
        "message": "Member deleted permanently",
    }

