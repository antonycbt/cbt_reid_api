from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.auth_service import AuthService
from app.schemas.auth import LoginRequest, LoginResponse

router = APIRouter()

@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    try:
        user = AuthService.login(db, payload.email, payload.password)
        return LoginResponse.from_user(user)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))