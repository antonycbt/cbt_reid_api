from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from app.schemas.access_group import (
    AccessGroupCreate,
    AccessGroupUpdate,
    AccessGroupOut,
    AccessGroupNode
)
from app.schemas.member import (
    MemberOut
)
from app.services.access_group_service import AccessGroupService
from app.schemas.common import MessageResponse
from app.db.session import get_db
from app.db.models import AccessGroup 

router = APIRouter()


# ---------- STATIC ROUTES FIRST ----------


@router.get("/tree", response_model=List[AccessGroupNode])
def get_access_group_tree(db: Session = Depends(get_db)):
    """
    Return nested access group tree.
    - Fetches all access groups (flat)
    - Builds a nested tree using parent_access_group_id
    """
    # load all access groups
    all_groups: List[AccessGroup] = db.query(AccessGroup).all()

    # build nested tree (plain dicts)
    tree = build_tree_from_flat(all_groups)

    return tree

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

def build_tree_from_flat(flat: List[AccessGroup]) -> List[Dict[str, Any]]:
    """
    Convert flat SQLAlchemy AccessGroup objects list into nested dict tree.
    Returns a list of root node dicts. Each node is a plain dict suitable for
    Pydantic serialization (matches AccessGroupNode).
    """
    node_map: Dict[int, Dict[str, Any]] = {}
    roots: List[Dict[str, Any]] = []

    # 1) create node_map entries
    for r in flat:
        node_map[r.id] = {
            "id": r.id,
            "name": r.name,
            "parent_access_group_id": r.parent_access_group_id,
            "is_active": bool(r.is_active),
            "children": [],
        }

    # 2) link children to parents (or add to roots)
    for r in flat:
        node = node_map[r.id]
        parent_id = r.parent_access_group_id
        if parent_id is not None and parent_id in node_map:
            node_map[parent_id]["children"].append(node)
        else:
            roots.append(node)

    return roots
