from typing import Optional, Tuple, List
from datetime import datetime

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func, and_, or_

from app.db.models import NormalizedData, Member, Camera


class NormalizedReportRepository:
    def __init__(self, db: Session):
        self.db = db

    def fetch_report(
        self,
        page: int,
        page_size: int,
        start_ts: Optional[datetime],
        end_ts: Optional[datetime],
        site_location_id: Optional[int],   # ← changed
        member_id: Optional[int],
        movement_type: Optional[int],
        search: Optional[str],
        min_match: Optional[int],
    ) -> Tuple[List[NormalizedData], int]:

        filters = []

        if site_location_id is not None:
            filters.append(Camera.site_location_id == site_location_id)   # ← changed

        if member_id is not None:
            filters.append(NormalizedData.member_id == member_id)

        if movement_type is not None:
            filters.append(NormalizedData.movement_type == movement_type)

        if start_ts is not None:
            filters.append(NormalizedData.movement_ts >= start_ts)

        if end_ts is not None:
            filters.append(NormalizedData.movement_ts <= end_ts)

        if min_match is not None:
            filters.append(NormalizedData.average_match_value >= min_match)

        if search:
            filters.append(
                or_(
                    NormalizedData.guest_temp_id.ilike(f"%{search}%"),
                    Member.first_name.ilike(f"%{search}%"),
                    Member.last_name.ilike(f"%{search}%"),
                )
            )

        # COUNT — always join Camera now (needed for site_location_id filter)
        count_stmt = (
            select(func.count())
            .select_from(NormalizedData)
            .join(Camera, NormalizedData.camera_id == Camera.id)
            .join(Member, isouter=True)
        )
        if filters:
            count_stmt = count_stmt.where(and_(*filters))

        total = self.db.execute(count_stmt).scalar() or 0

        # DATA
        offset = (page - 1) * page_size
        query = (
            select(NormalizedData)
            .options(
                selectinload(NormalizedData.member),
                selectinload(NormalizedData.camera),
            )
            .join(Camera, NormalizedData.camera_id == Camera.id)   # ← explicit join
            .join(Member, isouter=True)
        )

        if filters:
            query = query.where(and_(*filters))

        query = (
            query.order_by(NormalizedData.movement_ts.desc())
            .offset(offset)
            .limit(page_size)
        )

        rows = self.db.execute(query).scalars().all()
        return rows, total