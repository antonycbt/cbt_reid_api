from sqlalchemy.orm import Session
from sqlalchemy import select, insert, delete, func
from app.db.models.associations import member_access
from sqlalchemy.orm import joinedload
from app.db.models import Member

class MemberAccessRepository:

    # -------- CREATE (single) --------
    @staticmethod
    def create(
        db: Session,
        member_id: int,
        access_group_id: int,
    ):
        db.execute(
            insert(member_access).values(
                member_id=member_id,
                access_group_id=access_group_id,
            )
        )
        db.commit()

    # -------- EXISTS --------
    @staticmethod
    def exists(
        db: Session,
        member_id: int,
        access_group_id: int,
    ) -> bool:
        stmt = select(member_access).where(
            member_access.c.member_id == member_id,
            member_access.c.access_group_id == access_group_id,
        )
        return db.execute(stmt).first() is not None

    # -------- BULK CREATE --------
    @staticmethod
    def bulk_create(
        db: Session,
        member_id: int,
        access_group_ids: list[int],
    ) -> int:
        # find existing access groups
        existing_stmt = select(
            member_access.c.access_group_id
        ).where(
            member_access.c.member_id == member_id,
            member_access.c.access_group_id.in_(access_group_ids),
        )

        existing_ids = {
            row[0] for row in db.execute(existing_stmt).all()
        }

        new_rows = [
            {
                "member_id": member_id,
                "access_group_id": ag_id,
            }
            for ag_id in access_group_ids
            if ag_id not in existing_ids
        ]

        if not new_rows:
            return 0

        db.execute(insert(member_access), new_rows)
        db.commit()
        return len(new_rows)

    # -------- LIST --------
    @staticmethod
    def list(db, search: str | None, page: int = 0, page_size: int = 10): 
        query = db.query(Member).options(joinedload(Member.access_groups))

        if search:
            query = query.filter(
                (Member.first_name + " " + Member.last_name).ilike(f"%{search}%")
            )

        total = query.count()
        members = query.offset(page * page_size).limit(page_size).all()

        result = []
        for member in members:
            full_name = f"{member.first_name} {member.last_name}".strip()

            result.append({
                "member_id": member.id,
                "member_access_name": full_name,
                "access_groups": [
                    {"id": ag.id, "name": ag.name} for ag in member.access_groups
                ],
            })

        return result, total


    # -------- DELETE --------
    @staticmethod
    def delete(
        db: Session,
        member_id: int,
        access_group_id: int,
    ):
        db.execute(
            delete(member_access).where(
                member_access.c.member_id == member_id,
                member_access.c.access_group_id == access_group_id,
            )
        )
        db.commit()
