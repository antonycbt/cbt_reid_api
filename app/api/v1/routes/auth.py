from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.auth_service import AuthService
from app.schemas.auth import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse

router = APIRouter()

@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    try:
        user = AuthService.login(db, payload.email, payload.password)
        return LoginResponse.from_user(user)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    

@router.post("/register", response_model=RegisterResponse, status_code=201)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    try:
        user = AuthService.register(db, payload)
        return user
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))