from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from sqlalchemy import create_engine

from app.core.pipeline_args import build_pipeline_argv
from app.services.tracking.pipeline import parse_args
from app.services.tracking.service import DetectionService
from app.services.normalization.normalization_worker import NormalizationWorker
from app.services.normalization.normalization_service import NormalizationService
from app.repositories.raw_repository import RawRepository
from app.repositories.normalized_repository import NormalizedRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Starts the tracking pipeline when Uvicorn starts, stops on shutdown.

    - If DB mode and no --src IDs were provided, the runner will auto-start
      all active cameras from DB during svc.start().
    """

    svc: Optional[DetectionService] = None
    normalizer: Optional[NormalizationWorker] = None

    try:
        argv = build_pipeline_argv()
        args = parse_args(argv)

        svc = DetectionService(args)
        svc.start()

        # Expose service to routers
        app.state.detection_service = svc

        # ---- Normalizer startup (non-fatal) ----
        try:
            db_url = getattr(args, "db_url", "") or os.environ.get("DATABASE_URL", "")
            enable_normalizer = bool(getattr(args, "enable_normalizer", True))

            if db_url and enable_normalizer:
                engine = create_engine(db_url, pool_pre_ping=True)
                raw_repo = RawRepository(engine)
                raw_repo.ensure_raw_data_registry()
                norm_repo = NormalizedRepository(engine)
                norm_service = NormalizationService(raw_repo, norm_repo)

                normalizer = NormalizationWorker(
                    norm_service,
                    poll_interval=float(getattr(args, "normalizer_poll_interval", 5.0)),
                )
                normalizer.start()

                app.state.normalizer = normalizer
                app.state.normalizer_engine = engine
                print(f"***************************** NORMALIZATION IS RUNNING........................")

        except Exception as e:
            print(f"[lifespan] normalizer start failed: {e}")

        # Debug info
        app.state.pipeline_argv = argv
        app.state.pipeline_args = args

        yield

    finally:
        # ---- Graceful shutdown ----
        if normalizer is not None:
            try:
                normalizer.stop()
            except Exception as e:
                print(f"[lifespan] normalizer stop failed: {e}")

        if getattr(app.state, "normalizer_engine", None) is not None:
            try:
                app.state.normalizer_engine.dispose()
            except Exception:
                pass
            app.state.normalizer_engine = None

        if svc is not None:
            try:
                svc.stop()
            except Exception as e:
                print(f"[lifespan] stop failed: {e}")

        app.state.detection_service = None
        app.state.normalizer = None
        app.state.pipeline_argv = None
        app.state.pipeline_args = None
