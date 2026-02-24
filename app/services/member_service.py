from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from app.db.session import SessionLocal
from app.db.models.department import Department
from app.repositories.member_repo import MemberRepository
from app.schemas.member import (
    MemberUpdate
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
    
    # -------------------------
    # BULK IMPORT
    # -------------------------

    @staticmethod
    def bulk_import_members_from_rows(rows: list):
        db = SessionLocal()
        try:
            # ── Build department name → id map ─────────────────────────

            dept_name_to_id = MemberRepository.get_department_name_map(db)

            # ── Get existing member numbers ────────────────────────────
            raw_numbers = [
                str(row[0]).strip() for row in rows
                if row and row[0]
            ]
            existing_numbers = MemberRepository.get_existing_member_numbers(db, raw_numbers)

            # ── Process rows ───────────────────────────────────────────
            members_to_create = []
            skipped = []

            for i, row in enumerate(rows, start=2):  # start=2 → row 1 is header
                if not row or not row[0]:
                    continue

                member_number = str(row[0]).strip() if row[0] else None
                first_name    = str(row[1]).strip() if row[1] else None
                last_name     = str(row[2]).strip() if row[2] else None
                dept_raw      = str(row[3]).strip() if row[3] else None

                # Required fields check
                if not member_number or not first_name:
                    skipped.append({
                        "row": i,
                        "member_number": member_number or "N/A",
                        "reason": "Member Number or First Name is missing",
                    })
                    continue

                # Duplicate member number check
                if member_number.lower() in existing_numbers:
                    skipped.append({
                        "row": i,
                        "member_number": member_number,
                        "reason": f"Member number '{member_number}' already exists",
                    })
                    continue

                # Department name → id resolution
                department_id = None
                if dept_raw:
                    department_id = dept_name_to_id.get(dept_raw.lower())
                    if department_id is None:
                        skipped.append({
                            "row": i,
                            "member_number": member_number,
                            "reason": f"Department '{dept_raw}' does not exist",
                        })
                        continue

                members_to_create.append(Member(
                    member_number=member_number,
                    first_name=first_name,
                    last_name=last_name,
                    department_id=department_id,
                    is_active=True,
                ))

                # Track within-batch duplicates
                existing_numbers.add(member_number.lower())

            # ── Bulk insert ────────────────────────────────────────────
            added_count = 0
            if members_to_create:
                db.add_all(members_to_create)
                db.commit()
                added_count = len(members_to_create)

            # ── Build response ─────────────────────────────────────────
            total_skipped = len(skipped)
            message = (
                f"{added_count} {'entry' if added_count == 1 else 'entries'} added successfully. "
                f"{total_skipped} {'entry' if total_skipped == 1 else 'entries'} skipped."
                if total_skipped
                else f"{added_count} {'entry' if added_count == 1 else 'entries'} added successfully."
            )

            return {
                "message": message,
                "added_count": added_count,
                "skipped_count": total_skipped,
                "skipped": skipped,
            }

        finally:
            db.close()