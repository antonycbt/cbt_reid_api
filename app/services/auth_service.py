from sqlalchemy.orm import Session
from app.repositories.user_repo import UserRepository
from app.core.security import verify_password
from app.db.models.user import User

class AuthService:

    @staticmethod
    def login(db: Session, email: str, password: str) -> User:
        user = UserRepository.get_by_email(db, email)

        if not user:
            raise ValueError("Invalid email or password")

        if not user.is_active:
            raise ValueError("User is inactive")

        if not verify_password(password, user.password):
            raise ValueError("Invalid email or password")

        UserRepository.update_last_login(db, user)

        return user