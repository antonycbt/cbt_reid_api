from sqlalchemy.orm import Session
from sqlalchemy import select, insert, delete, func
from app.db.models.associations import member_access
from sqlalchemy.orm import joinedload, with_loader_criteria
from app.db.models import Member, AccessGroup

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

        # Pre-fetch all access groups for ancestor traversal
        all_access_groups = {ag.id: ag for ag in db.query(AccessGroup).all()}

        def all_access_group_ancestors_active(access_group_id: int) -> bool:
            current = all_access_groups.get(access_group_id)
            while current and current.parent_access_group_id is not None:
                parent = all_access_groups.get(current.parent_access_group_id)
                if parent is None or not parent.is_active:
                    return False
                current = parent
            return True

        query = (
            db.query(Member)
            .options(joinedload(Member.access_groups))
        )

        if search:
            query = query.filter(
                (Member.first_name + " " + Member.last_name).ilike(f"%{search}%")
            )

        total = query.count()
        members = query.offset(page * page_size).limit(page_size).all()

        access_group_map: dict[int, dict] = {}

        for member in members:
            if not member.is_active:
                continue

            full_name = " ".join(
                part for part in [member.first_name, member.last_name] if part
            )

            for ag in member.access_groups:
                if not ag.is_active or not all_access_group_ancestors_active(ag.id):
                    continue

                if ag.id not in access_group_map:
                    access_group_map[ag.id] = {
                        "access_group_id": ag.id,
                        "access_group_name": ag.name,
                        "members": [],
                    }

                access_group_map[ag.id]["members"].append({
                    "member_id": member.id,
                    "member_name": full_name,
                })

        result = list(access_group_map.values())
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
