from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
from sqlalchemy.orm import Session, selectinload

from app.schemas.site_hierarchy import (
    SiteHierarchyCreate,
    SiteHierarchyUpdate,
    SiteHierarchyOut,
    SiteHierarchyNode,
)
from app.schemas.common import MessageResponse
from app.db.session import get_db
from app.db.models.site_hierarchy import SiteHierarchy
from app.services.site_hierarchy_service import SiteHierarchyService
from sqlalchemy import exists, and_

router = APIRouter()

# ---------------- TREE HELPER ----------------
from typing import List, Dict, Set
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, selectinload
from app.db.session import get_db
from app.db.models import SiteHierarchy, SiteLocation, Camera
from app.schemas.site_hierarchy import SiteHierarchyNode

router = APIRouter()

# ---------------- TREE HELPER ----------------
def build_tree_with_lock(
    nodes: List[SiteHierarchy],
    direct_locations_map: Dict[int, List[int]],
    used_location_ids: Set[int],
    location_meta_map: Dict[int, Dict],   # ✅ NEW
) -> List[SiteHierarchyNode]:

    node_map: Dict[int, SiteHierarchy] = {node.id: node for node in nodes}
    roots: List[SiteHierarchy] = []

    # reset children
    for node in nodes:
        node.children = []

    # assign children
    for node in nodes:
        if node.parent_site_hierarchy_id:
            parent = node_map.get(node.parent_site_hierarchy_id)
            if parent:
                parent.children.append(node)
        else:
            roots.append(node)

    # lock only direct leaf nodes (UNCHANGED)
    for node in nodes:
        direct_loc_ids = direct_locations_map.get(node.id, [])
        node._is_locked = any(loc_id in used_location_ids for loc_id in direct_loc_ids)

    # convert to Pydantic
    def orm_to_pydantic(node: SiteHierarchy) -> SiteHierarchyNode:
        pyd = SiteHierarchyNode.from_orm(node)

        # children
        pyd.children = [
            orm_to_pydantic(child)
            for child in getattr(node, "children", [])
        ]

        # existing lock (UNCHANGED)
        pyd.is_locked = bool(getattr(node, "_is_locked", False))

        # NEW: fetch from site_location (ONLY for leaf nodes)
        direct_loc_ids = direct_locations_map.get(node.id, [])
        is_leaf = len(direct_loc_ids) > 0

        if is_leaf:
            loc_meta = location_meta_map.get(node.id)
            if loc_meta:
                pyd.is_public = loc_meta.get("is_public", False)
                pyd.is_protected = loc_meta.get("is_protected", False)

        return pyd

    return [orm_to_pydantic(root) for root in roots]


# ---------------- ROUTE ----------------

@router.get("/tree", response_model=List[SiteHierarchyNode])
def get_site_hierarchy_tree(db: Session = Depends(get_db)):

    # load hierarchy
    nodes = db.query(SiteHierarchy).options(
        selectinload(SiteHierarchy.children)
    ).all()

    # 1) site_location mapping + meta
    loc_rows = db.query(
        SiteLocation.id,
        SiteLocation.site_hierarchy_id,
        SiteLocation.is_public,
        SiteLocation.is_protected
    ).all()

    direct_locations_map: Dict[int, List[int]] = {}
    location_meta_map: Dict[int, Dict] = {}

    for loc_id, hierarchy_id, is_public, is_protected in loc_rows:
        if hierarchy_id is None:
            continue

        # existing mapping
        direct_locations_map.setdefault(hierarchy_id, []).append(loc_id)

        # store meta (leaf data source)
        location_meta_map[hierarchy_id] = {
            "is_public": is_public,
            "is_protected": is_protected
        }

    # 2) used locations (UNCHANGED)
    used_rows = (
        db.query(Camera.site_location_id)
        .filter(Camera.site_location_id.isnot(None))
        .distinct()
        .all()
    )

    used_location_ids: Set[int] = {
        r[0] for r in used_rows if r[0] is not None
    }

    # 3) build tree (pass new map)
    return build_tree_with_lock(
        nodes,
        direct_locations_map,
        used_location_ids,
        location_meta_map   
    ) 

# LIST (pagination + search)
@router.get("", response_model=MessageResponse[List[SiteHierarchyOut]])
def list_site_hierarchies(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    used_by_camera = exists().where(
        and_(
            Camera.site_location_id == SiteLocation.id,
            SiteLocation.site_hierarchy_id == SiteHierarchy.id,
        )
    )

    query = db.query(SiteHierarchy).filter(~used_by_camera)

    if search:
        query = query.filter(SiteHierarchy.name.ilike(f"%{search}%"))

    total = query.count()
    sites = query.offset(page * page_size).limit(page_size).all()

    for site in sites:
        site.children = []

    return {
        "message": "Site hierarchies fetched successfully",
        "data": [SiteHierarchyOut.from_orm(s) for s in sites],
        "total": total,
    }

@router.get("/active_hierarchy", response_model=MessageResponse[List[SiteHierarchyOut]])
def list_active_site_hierarchies(
    db: Session = Depends(get_db)
):
    used_by_camera = exists().where(
        and_(
            Camera.site_location_id == SiteLocation.id,
            SiteLocation.site_hierarchy_id == SiteHierarchy.id,
        )
    )

    sites = (
        db.query(SiteHierarchy)
        .filter(
            ~used_by_camera,
            SiteHierarchy.is_active == True,
        )
        .all()
    )

    all_sites_map: dict[int, SiteHierarchy] = {s.id: s for s in db.query(SiteHierarchy).all()}

    def all_ancestors_active(site: SiteHierarchy) -> bool:
        current = site
        while current.parent_site_hierarchy_id is not None:
            parent = all_sites_map.get(current.parent_site_hierarchy_id)
            if parent is None or not parent.is_active:
                return False
            current = parent
        return True

    valid_sites = [s for s in sites if all_ancestors_active(s)]

    for site in valid_sites:
        site.children = []

    return {
        "message": "Active site hierarchies fetched successfully",
        "data": [SiteHierarchyOut.from_orm(s) for s in valid_sites],
        "total": len(valid_sites),
    }

# CREATE
@router.post("", response_model=MessageResponse[SiteHierarchyOut])
def create_site_hierarchy(
    payload: SiteHierarchyCreate,
    db: Session = Depends(get_db),
):
    site = SiteHierarchyService.create_site_hierarchy(db, payload)

    site.children = []

    return {
        "message": "Site hierarchy created successfully",
        "data": SiteHierarchyOut.from_orm(site),
    }



# GET BY ID
@router.get("/{site_hierarchy_id}", response_model=MessageResponse[SiteHierarchyOut])
def get_site_hierarchy(site_hierarchy_id: int, db: Session = Depends(get_db)):
    site = db.query(SiteHierarchy).options(selectinload(SiteHierarchy.children)).filter(SiteHierarchy.id == site_hierarchy_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site hierarchy not found")
    
    site.children = []
    return {
        "message": "Site hierarchy fetched successfully",
        "data": SiteHierarchyOut.from_orm(site),
    }


# UPDATE
@router.put("/{site_hierarchy_id}", response_model=MessageResponse[SiteHierarchyOut])
def update_site_hierarchy(
    site_hierarchy_id: int,
    payload: SiteHierarchyUpdate,
    db: Session = Depends(get_db),
):
    site = SiteHierarchyService.update_site_hierarchy(
        db,
        site_hierarchy_id,
        payload,
    )

    site.children = []

    return {
        "message": "Site hierarchy updated successfully",
        "data": SiteHierarchyOut.from_orm(site),
    } 

# DELETE
@router.delete("/{site_hierarchy_id}", response_model=MessageResponse[None])
def delete_site_hierarchy(site_hierarchy_id: int, db: Session = Depends(get_db)):

    site = db.query(SiteHierarchy).filter(
        SiteHierarchy.id == site_hierarchy_id
    ).first()

    if not site:
        raise HTTPException(status_code=404, detail="Site hierarchy not found")

    # find related site locations
    locations = db.query(SiteLocation).filter(
        SiteLocation.site_hierarchy_id == site_hierarchy_id
    ).all()

    location_ids = [l.id for l in locations]

    # check if any location used in camera
    in_use = db.query(Camera.id).filter(
        Camera.site_location_id.in_(location_ids)
    ).first()

    if in_use:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete. Site hierarchy is used by a camera."
        )

    # delete locations via ORM (important)
    for loc in locations:
        db.delete(loc)

    # delete hierarchy
    db.delete(site)

    db.commit()

    return {"message": "Site hierarchy deleted successfully"}

@router.get("/full_tree/{site_hierarchy_id}", response_model=MessageResponse[List[dict]])
def get_full_hierarchy_tree(site_hierarchy_id: int):
    tree = SiteHierarchyService.get_full_hierarchy_tree(site_hierarchy_id)
    return {
        "message": "Site hierarchy + locations tree fetched successfully",
        "data": tree
    }



