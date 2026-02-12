from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    DATABASE_URL: str
    pipeline_args: str = Field(..., env="PIPELINE_ARGS")
    RTSP_USER: str = ""
    RTSP_PASS: str = ""
    RTSP_PORT: str = ""
    RTSP_PATH: str = ""
    RTSP_SCHEME: str = "rtsp"
    RTSP_STREAM: str = ""
    RTSP_URL_TEMPLATE: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="forbid",  # prevent unknown fields
    )

settings = Settings()


import os
print("----------------------------------OS sees PIPELINE_ARGS =", os.environ.get("PIPELINE_ARGS"))

