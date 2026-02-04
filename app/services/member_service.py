from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from app.db.session import SessionLocal
from app.repositories.member_repo import MemberRepository
from app.schemas.member import (
    MemberUpdate,
)
from app.db.models.member import Member
from sqlalchemy.orm import Session
from sqlalchemy import select,func

class MemberService:

    # -------------------------
    # CREATE member
    # -------------------------
    @staticmethod
    def create_member(payload):
        db = SessionLocal()
        try:
            # unique member_number check
            if MemberRepository.exists_with_member_number(
                db,
                payload.member_number,
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Member number already exists"
                )

            return MemberRepository.create(db, payload)

        finally:
            db.close()

    # -------------------------
    # UPDATE member
    # -------------------------
    @staticmethod
    def update_member(
        member_id: int,
        payload: MemberUpdate,
    ):
        db = SessionLocal()
        try:
            member = MemberRepository.get_by_id(db, member_id)
            if not member:
                raise HTTPException(
                    status_code=404,
                    detail="Member not found"
                )

            # optional uniqueness check on update
            if (
                payload.member_number
                and MemberRepository.exists_with_member_number(
                    db,
                    payload.member_number,
                    exclude_id=member_id,
                )
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Member number already exists"
                )

            return MemberRepository.update(db, member, payload)

        finally:
            db.close()

    # -------------------------
    # GET member by ID
    # -------------------------
    @staticmethod
    def get_member(member_id: int) -> Member | None:
        db = SessionLocal()
        try:
            return MemberRepository.get_by_id(db, member_id)
        finally:
            db.close()

    # -------------------------
    # LIST members (search + pagination)
    # -------------------------
    @staticmethod
    def list_members(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return MemberRepository.list(
                db=db,
                search=search,
                page=page,
                page_size=page_size,
            )
        finally:
            db.close()


    # -------------------------
    # HARD DELETE member
    # -------------------------
    @staticmethod
    def delete_member(member_id: int) -> bool:
        db = SessionLocal()
        try:
            member = MemberRepository.get_by_id(db, member_id)
            if not member:
                return False

            MemberRepository.delete(db, member)
            return True

        finally:
            db.close()

    @staticmethod
    def list_all_site_locations(db: Session):
        # fetch active members
        stmt = select(Member).where(Member.is_active.is_(True))
        members = db.execute(stmt).scalars().all()

        # total count
        total = db.execute(
            select(func.count(Member.id))
            .where(Member.is_active.is_(True))
        ).scalar()

        return members, total  