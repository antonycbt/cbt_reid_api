from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.orm import Session

from app.repositories.report_repo import NormalizedReportRepository
from app.core.constants import MOVEMENT_TYPES


class NormalizedReportService:
    def __init__(self, db: Session):
        self.repo = NormalizedReportRepository(db)

    def get_report(
        self,
        page: int,
        page_size: int,
        start_ts: Optional[datetime],
        end_ts: Optional[datetime],
        site_location_id: Optional[int],
        member_id: Optional[int],
        movement_type: Optional[int],
        search: Optional[str],
        min_match: Optional[int],
    ) -> Dict[str, Any]:

        rows, total = self.repo.fetch_report(
            page,
            page_size,
            start_ts,
            end_ts,
            site_location_id,
            member_id,
            movement_type,
            search,
            min_match,
        )

        serialized: List[Dict[str, Any]] = []

        for r in rows:
            # ✅ Member name
            member_name = None
            if r.member:
                member_name = f"{r.member.first_name} {r.member.last_name or ''}".strip()

            # ✅ Camera info
            camera_name = None
            camera_location = None
            if r.camera:
                camera_name = r.camera.name
                camera_location = r.camera.site_location

            # ✅ Movement label
            movement_label = MOVEMENT_TYPES.get(r.movement_type)

            serialized.append({
                "member": member_name,
                "guest_temp_id": r.guest_temp_id,
                "camera": camera_name,
                "location": camera_location,
                "movement_type": movement_label,
                "movement_ts": r.movement_ts.isoformat() if r.movement_ts else None,
                "average_match_value": r.average_match_value,
            })

        return {
            "rows": serialized,
            "total": total,
        }