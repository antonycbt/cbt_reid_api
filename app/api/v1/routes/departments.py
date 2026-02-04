from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.schemas.department import (
    DepartmentCreate,
    DepartmentUpdate,
    DepartmentOut,
)
from app.services.department_service import DepartmentService
from app.schemas.common import MessageResponse
from typing import List

router = APIRouter()


@router.get(
    "/all",
    response_model=List[DepartmentOut],
)
def list_all_departments():
    return DepartmentService.list_all_departments()

# CREATE department
@router.post("", response_model=MessageResponse[DepartmentOut])
def create_department(payload: DepartmentCreate):
    department = DepartmentService.create_department(payload)
    return {
        "message": "Department created successfully",
        "data": department,
    }

# GET department by ID
@router.get("/{department_id}", response_model=MessageResponse[DepartmentOut])
def get_department(department_id: int):
    department = DepartmentService.get_department(department_id)
    if not department:
        raise HTTPException(status_code=404, detail="Department not found")

    return {
        "message": "Department fetched successfully",
        "data": department,
    }

# LIST departments (active only + search + pagination)
@router.get("")
def list_departments(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=100),
):
    departments, total = DepartmentService.list_departments(
        search, page, page_size
    )
    return {
        "message": "Departments fetched successfully",
        "data": departments,
        "total": total,
    }

# UPDATE department
@router.put("/{department_id}", response_model=MessageResponse[DepartmentOut])
def update_department(department_id: int, payload: DepartmentUpdate):
    department = DepartmentService.update_department(department_id, payload)
    return {
        "message": "Department updated successfully",
        "data": department,
    }

# HARD DELETE (permanent)
@router.delete("/{department_id}", response_model=MessageResponse[None])
def delete_department(department_id: int):
    success = DepartmentService.delete_department(department_id)
    if not success:
        raise HTTPException(status_code=404, detail="Department not found")

    return {
        "message": "Department deleted permanently",
    }


