from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from app.db.session import SessionLocal
from app.repositories.department_repo import DepartmentRepository
from app.schemas.department import DepartmentCreate, DepartmentUpdate
from app.db.models.department import Department


class DepartmentService:

    @staticmethod
    def create_department(payload: DepartmentCreate) -> Department:
        db = SessionLocal()
        try:
            return DepartmentRepository.create(db, payload)

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Department already exists"
            )

        finally:
            db.close()


    # Get Department by ID
    @staticmethod
    def get_department(department_id: int) -> Department | None:
        db = SessionLocal()
        try:
            return DepartmentRepository.get_by_id(db, department_id)
        finally:
            db.close()

    # List Departments (search + pagination)
    @staticmethod
    def list_departments(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return DepartmentRepository.list(db, search, page, page_size)
        finally:
            db.close()

    # Update Department
    @staticmethod
    def update_department(department_id: int, payload: DepartmentUpdate) -> Department:
        db = SessionLocal()
        try:
            department = DepartmentRepository.get_by_id(db, department_id)
            if not department:
                raise HTTPException(status_code=404, detail="Department not found")

            return DepartmentRepository.update(db, department, payload)

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Department name already exists"
            )

        finally:
            db.close()


    # Hard delete Department
    @staticmethod
    def delete_department(department_id: int) -> bool:
        db = SessionLocal()
        try:
            department = DepartmentRepository.get_by_id(db, department_id)
            if not department:
                return False

            DepartmentRepository.delete(db, department)
            return True
        finally:
            db.close()

    @staticmethod
    def list_all_departments():
        db = SessionLocal()
        try:
            return DepartmentRepository.list_all_active(db)
        finally:
            db.close()        

