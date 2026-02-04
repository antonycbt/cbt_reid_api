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

router = APIRouter()

# ---------------- TREE HELPER ----------------
def build_tree(nodes: List[SiteHierarchy]) -> List[SiteHierarchyNode]:
    node_map = {node.id: node for node in nodes}
    roots = []

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

    # convert to Pydantic
    return [SiteHierarchyNode.from_orm(node) for node in roots]


# ---------------- ROUTES ----------------

# GET hierarchy tree
@router.get("/tree", response_model=List[SiteHierarchyNode])
def get_site_hierarchy_tree(db: Session = Depends(get_db)):
    nodes = db.query(SiteHierarchy).options(selectinload(SiteHierarchy.children)).all()
    return build_tree(nodes)


# LIST (pagination + search)
@router.get("", response_model=MessageResponse[List[SiteHierarchyOut]])
def list_site_hierarchies(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    query = db.query(SiteHierarchy)
    if search:
        query = query.filter(SiteHierarchy.name.ilike(f"%{search}%"))

    total = query.count()
    sites = query.offset(page * page_size).limit(page_size).all()

    # prevent lazy loading children errors
    for site in sites:
        site.children = []

    return {
        "message": "Site hierarchies fetched successfully",
        "data": [SiteHierarchyOut.from_orm(s) for s in sites],
        "total": total,
    }


# CREATE
@router.post("", response_model=MessageResponse[SiteHierarchyOut])
def create_site_hierarchy(payload: SiteHierarchyCreate, db: Session = Depends(get_db)):
    new_node = SiteHierarchy(
        name=payload.name,
        parent_site_hierarchy_id=payload.parent_site_hierarchy_id,
        is_active=payload.is_active,
    )
    db.add(new_node)
    db.commit()
    db.refresh(new_node)

    new_node.children = []
    return {
        "message": "Site hierarchy created successfully",
        "data": SiteHierarchyOut.from_orm(new_node),
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
def update_site_hierarchy(site_hierarchy_id: int, payload: SiteHierarchyUpdate, db: Session = Depends(get_db)):
    site = db.query(SiteHierarchy).filter(SiteHierarchy.id == site_hierarchy_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site hierarchy not found")
    
    if payload.name is not None:
        site.name = payload.name
    if payload.parent_site_hierarchy_id is not None:
        site.parent_site_hierarchy_id = payload.parent_site_hierarchy_id
    if payload.is_active is not None:
        site.is_active = payload.is_active

    db.commit()
    db.refresh(site)
    site.children = []
    return {
        "message": "Site hierarchy updated successfully",
        "data": SiteHierarchyOut.from_orm(site),
    }


# DELETE
@router.delete("/{site_hierarchy_id}", response_model=MessageResponse[None])
def delete_site_hierarchy(site_hierarchy_id: int, db: Session = Depends(get_db)):
    site = db.query(SiteHierarchy).filter(SiteHierarchy.id == site_hierarchy_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site hierarchy not found")

    # Recursive function to delete children
    def delete_children(site: SiteHierarchy):
        for child in site.children:  # assuming you have a relationship 'children'
            delete_children(child)
            db.delete(child)

    delete_children(site)
    db.delete(site)
    db.commit()
    return {"message": "Site hierarchy and its children deleted permanently"}

@router.get("/full_tree/{site_hierarchy_id}", response_model=MessageResponse[List[dict]])
def get_full_hierarchy_tree(site_hierarchy_id: int):
    tree = SiteHierarchyService.get_full_hierarchy_tree(site_hierarchy_id)
    return {
        "message": "Site hierarchy + locations tree fetched successfully",
        "data": tree
    }



