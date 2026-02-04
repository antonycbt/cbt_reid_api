from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.db.models.department import Department
from app.schemas.department import DepartmentCreate, DepartmentUpdate
from typing import List


class DepartmentRepository:

    # Create Department
    @staticmethod
    def create(db: Session, payload: DepartmentCreate) -> Department:
        department = Department(
            name=payload.name,
        )
        db.add(department)
        db.commit()
        db.refresh(department)
        return department

    # Get department by ID
    @staticmethod
    def get_by_id(db: Session, department_id: int) -> Department | None:
        return db.get(Department, department_id)

    # List departments (active + search + pagination)
    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list[Department], int]:

        stmt = select(Department)

        # search filter
        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.where(
                func.lower(Department.name).like(search_term)
            )

        # total count (before pagination)
        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        # ORDER: active first, inactive last, then name
        stmt = stmt.order_by(
            Department.is_active.desc(),
            func.lower(Department.name).asc()
        )

        # pagination
        stmt = stmt.offset(page * page_size).limit(page_size)

        departments = db.execute(stmt).scalars().all()
        return departments, total


    # Update department
    @staticmethod
    def update(
        db: Session,
        department: Department,
        payload: DepartmentUpdate,
    ) -> Department:
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(department, field, value)

        db.commit()
        db.refresh(department)
        return department

    # Hard delete (permanent)
    @staticmethod
    def delete(db: Session, department: Department) -> None:
        db.delete(department)
        db.commit()

    @staticmethod
    def list_all_active(db: Session) -> List[Department]:
        stmt = (
            select(Department)
            .where(Department.is_active.is_(True))
            .order_by(func.lower(Department.name).asc())
        )
        return db.execute(stmt).scalars().all()