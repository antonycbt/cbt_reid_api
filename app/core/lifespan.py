from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.pipeline_args import build_pipeline_argv
from app.services.tracking.pipeline import parse_args
from app.services.tracking.service import DetectionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    argv = build_pipeline_argv()
    args = parse_args(argv)

    svc = DetectionService(args)
    svc.start()

    app.state.detection_service = svc
    try:
        yield
    finally:
        svc.stop()
        app.state.detection_service = None
