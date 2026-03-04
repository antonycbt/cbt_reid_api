from typing import Optional, Dict, Any, List
from datetime import datetime

from app.db.session import SessionLocal
from app.repositories.report_repo import NormalizedReportRepository
from app.core.constants import MOVEMENT_TYPES


class NormalizedReportService:

    @staticmethod
    def list_site_locations(search: Optional[str] = None) -> List[Dict]:
        db = SessionLocal()
        try:
            return NormalizedReportRepository.list_site_locations(db, search=search)
        finally:
            db.close()

    @staticmethod
    def list_active_member_names(
        search: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        db = SessionLocal()
        try:
            members = NormalizedReportRepository.list_all_active(
                db, search=search, limit=limit
            )
            return [
                {
                    "id": m.id,
                    "name": f"{m.first_name} {m.last_name or ''}".strip(),
                }
                for m in members
            ]
        finally:
            db.close()

    @staticmethod
    def get_report(
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
        db = SessionLocal()
        try:
            rows, total = NormalizedReportRepository.fetch_report(
                db, page, page_size, start_ts, end_ts,
                site_location_id, member_id, movement_type, search, min_match,
            )

            serialized: List[Dict[str, Any]] = []
            for r in rows:
                member_name = None
                if r.member:
                    member_name = f"{r.member.first_name} {r.member.last_name or ''}".strip()

                camera_name = None
                camera_location = None
                if r.camera:
                    camera_name = r.camera.name
                    # site_location_rel gives us the SiteLocation object
                    # SiteLocation.name is a @property → site_hierarchy.name
                    if r.camera.site_location_rel:
                        camera_location = r.camera.site_location_rel.name

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

            return {"rows": serialized, "total": total}
        finally:
            db.close()