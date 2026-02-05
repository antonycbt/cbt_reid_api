from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func, exists, and_, or_
from typing import List

from app.db.models.member import Member
from app.db.models.member_embedding import MemberEmbedding
from app.schemas.member import MemberCreate, MemberUpdate 


class MemberRepository:

    # -------------------------
    # CREATE
    # -------------------------
    @staticmethod
    def create(db: Session, payload: MemberCreate) -> Member:
        member = Member(
            member_number=payload.member_number,
            first_name=payload.first_name,
            last_name=payload.last_name,
            department_id=payload.department_id,
            is_active=payload.is_active,
        )
        db.add(member)
        db.commit()
 
        return (
            db.query(Member)
            .options(
                joinedload(Member.department),
                joinedload(Member.access_groups),
            )
            .filter(Member.id == member.id)
            .one()
        )

    # -------------------------
    # GET BY ID
    # -------------------------
    @staticmethod
    def get_by_id(db: Session, member_id: int) -> Member | None:
        return (
            db.query(Member)
            .options(
                joinedload(Member.department),
                joinedload(Member.access_groups),
            )
            .filter(Member.id == member_id)
            .one_or_none()
        )

    # -------------------------
    # DUPLICATION CHECK
    # -------------------------
    @staticmethod
    def exists_with_member_number(
        db: Session,
        member_number: str,
        exclude_id: int | None = None,
    ) -> bool:
        stmt = select(Member).where(
            func.lower(Member.member_number) == member_number.lower()
        )

        if exclude_id:
            stmt = stmt.where(Member.id != exclude_id)

        return db.execute(stmt).scalars().first() is not None

    # -------------------------
    # UPDATE
    # -------------------------
    @staticmethod
    def update(
        db: Session,
        member: Member,
        payload: MemberUpdate,
    ) -> Member:

        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(member, field, value)

        db.commit()

        # CRITICAL: re-fetch with relationships eagerly loaded
        return (
            db.query(Member)
            .options(
                joinedload(Member.department),
                joinedload(Member.access_groups),
            )
            .filter(Member.id == member.id)
            .one()
        )

    # -------------------------
    # LIST (search + pagination)
    # -------------------------
    
    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list[tuple[Member, bool]], int]:
        has_embeddings_expr = exists(
            select(1).where(
                and_(
                    MemberEmbedding.member_id == Member.id,
                    MemberEmbedding.body_embedding.isnot(None),
                    MemberEmbedding.face_embedding.isnot(None),
                    MemberEmbedding.back_body_embedding.isnot(None),
                    MemberEmbedding.body_embeddings_raw.isnot(None),
                    MemberEmbedding.face_embeddings_raw.isnot(None),
                    MemberEmbedding.back_body_embeddings_raw.isnot(None),
                )
            )
        ).label("has_embeddings")

        # build the main query
        stmt = (
            select(Member, has_embeddings_expr)
            .options(
                joinedload(Member.department),
                joinedload(Member.access_groups),
            )
        )

        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.where(
                func.lower(Member.first_name).like(search_term)
                | func.lower(Member.last_name).like(search_term)
                | func.lower(Member.member_number).like(search_term)
            )

        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        stmt = (
            stmt.order_by(
                Member.is_active.desc(),
                func.lower(Member.first_name).asc(),
            )
            .offset(page * page_size)
            .limit(page_size)
        )

        rows = db.execute(stmt).unique().all()

        members = []
        for member, has_embeddings in rows:
            member.has_embeddings = has_embeddings  # attach flag
            members.append(member)

        return members, total
 

    # -------------------------
    # LIST ALL ACTIVE (for dropdowns)
    # -------------------------
    @staticmethod
    def list_all_active(db: Session) -> List[Member]:
        stmt = (
            select(Member)
            .where(Member.is_active.is_(True))
            .order_by(
                func.lower(Member.first_name).asc(),
                func.lower(Member.last_name).asc(),
            )
        )
        return db.execute(stmt).scalars().all()

    # -------------------------
    # DELETE (HARD)
    # -------------------------
    @staticmethod
    def delete(db: Session, member: Member) -> None:
        db.delete(member)
        db.commit()
