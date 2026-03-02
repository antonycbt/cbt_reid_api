from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.schemas.common import MessageResponse
from app.services.report_service import NormalizedReportService
from app.db.session import get_db

router = APIRouter()


@router.get("/member_presence_report", response_model=MessageResponse[dict])
def get_normalized_report(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    start_ts: Optional[datetime] = Query(None),
    end_ts: Optional[datetime] = Query(None),
    camera_id: Optional[int] = Query(None),
    member_id: Optional[int] = Query(None),
    movement_type: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    min_match: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    service = NormalizedReportService(db)

    result = service.get_report(
        page,
        page_size,
        start_ts,
        end_ts,
        camera_id,
        member_id,
        movement_type,
        search,
        min_match,
    )

    return {
        "message": "Report fetched successfully",
        "data": result,
    }