from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.auth_service import AuthService
from app.schemas.auth import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail

router = APIRouter()

@router.post("/login")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    try:
        user = AuthService.login(db, payload.email, password=payload.password)

        # Log the login activity
        detail = ActivityDetail(
            action="login",
            entity="user",
            changes={},
            meta={
                "actor_id": user.id,
                "display_name": f"{user.first_name} {user.last_name}",
            },
        )
        ActivityLogService.log(
            db=db,
            actor_id=user.id,
            target_type=1,
            target_id=user.id,
            detail=detail,
        )
        db.commit()

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