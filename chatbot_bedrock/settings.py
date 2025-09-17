import os
from typing import Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class BedrockSettings(BaseModel):
    aws_default_region: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )
    bedrock: BedrockSettings = BedrockSettings()


def get_settings() -> Settings:
    settings = Settings()

    # Middleware to set env variables for boto3
    for key, value in settings.bedrock.model_dump().items():
        if value is not None:
            os.environ[key.upper()] = value
    return settings
