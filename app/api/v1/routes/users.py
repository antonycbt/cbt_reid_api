from fastapi import APIRouter, HTTPException
from typing import List

from app.schemas.user import UserCreate, UserUpdate, UserOut
from app.services.user_service import UserService
from app.schemas.common import MessageResponse 
from typing import Optional 
from fastapi import Query 
from app.core.constants import USER_ROLES

router = APIRouter()

# GET user roles
@router.get("/roles", response_model=MessageResponse[List[dict]])
def get_user_roles():
    roles_list = [{"id": k, "name": v} for k, v in USER_ROLES.items()]
    return {
        "message": "Roles fetched successfully",
        "data": roles_list
    }


#CREATE user
@router.post("", response_model=MessageResponse[UserOut])
def create_user(payload: UserCreate):
    user = UserService.create_user(payload)
    return {
        "message": "User created successfully",
        "data": user
    }

#GET user by ID
@router.get("/{user_id}", response_model=MessageResponse[UserOut])
def get_user(user_id: int):
    user = UserService.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "message": "User fetched successfully",
        "data": user
    }

#LIST users (active only)
@router.get("")
def list_users(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=100),
):
    users, total = UserService.list_users(search, page, page_size)
    return {
        "message": "Users fetched successfully",
        "data": users,
        "total": total,
    }


#UPDATE user
@router.put("/{user_id}", response_model=MessageResponse[UserOut])
def update_user(user_id: int, payload: UserUpdate):
    user = UserService.update_user(user_id, payload)
    return {
        "message": "User updated successfully",
        "data": user
    }


#HARD DELETE 
@router.delete("/{user_id}", response_model=MessageResponse[None])
def delete_user(user_id: int):
    success = UserService.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "message": "User deleted permanently"
    }


