from app.db.session import SessionLocal
from app.repositories.user_repo import UserRepository
from app.schemas.user import UserCreate, UserUpdate
from app.db.models.user import User
from app.core.security import hash_password
from fastapi import HTTPException, status

#Create User
class UserService:
    @staticmethod
    def create_user(payload: UserCreate):
        db = SessionLocal()
        try:
            # check email uniqueness
            if UserRepository.get_by_email(db, payload.email):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already exists"
                )

            # HASH PASSWORD
            payload.password = hash_password(payload.password)

            return UserRepository.create(db, payload)

        finally:
            db.close()


    #Get User
    @staticmethod
    def get_user(user_id: int) -> User | None:
        db = SessionLocal()
        try:
            return UserRepository.get_by_id(db, user_id)
        finally:
            db.close()

    @staticmethod
    def list_users(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return UserRepository.list(db, search, page, page_size)
        finally:
            db.close()

    #Update User
    @staticmethod
    def update_user(user_id: int, payload: UserUpdate) -> User | None:
        db = SessionLocal()
        try:
            user = UserRepository.get_by_id(db, user_id)
            if not user:
                return None

            return UserRepository.update(db, user, payload)
        finally:
            db.close()
                
    # Hard delete user
    @staticmethod
    def delete_user(user_id: int) -> bool:
        db = SessionLocal()
        try:
            user = UserRepository.get_by_id(db, user_id)
            if not user:
                return False

            UserRepository.delete(db, user)
            return True
        finally:
            db.close()


