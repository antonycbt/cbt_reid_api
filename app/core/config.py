from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str

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
    )

settings = Settings()

RTSP_USERNAME = settings.RTSP_USER
RTSP_PASSWORD = settings.RTSP_PASS
RTSP_PORT = settings.RTSP_PORT
RTSP_PATH = settings.RTSP_PATH
RTSP_SCHEME = settings.RTSP_SCHEME
RTSP_STREAM = settings.RTSP_STREAM
RTSP_URL_TEMPLATE = settings.RTSP_URL_TEMPLATE
