"""
Application settings and configuration management.

This module loads and validates environment variables, providing a centralized
configuration interface for the entire application.
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Pydantic v2 compatibility
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field, field_validator
    PYDANTIC_V2 = True
except ImportError:
    # Pydantic v1 fallback
    try:
        from pydantic import BaseSettings, Field, validator
        PYDANTIC_V2 = False
    except ImportError:
        # Fallback to basic implementation
        from pydantic import BaseModel as BaseSettings, Field
        validator = lambda *args, **kwargs: lambda f: f
        PYDANTIC_V2 = False

# Find project root (where .env file should be)
# This file is in src/stock_sentiment/config/, so go up 3 levels to project root
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent.parent
_env_file = _project_root / ".env"

# Load environment variables from .env file in project root
# This must happen before Pydantic BaseSettings classes are instantiated
# We load with dotenv first, then Pydantic can also read from the file
_env_loaded = False
if _env_file.exists():
    load_dotenv(dotenv_path=_env_file, override=True)
    _env_loaded = True
    # Also set as absolute path for Pydantic v2
    _env_file_abs = str(_env_file.resolve())
else:
    # Fallback: try current directory and parent directories
    load_dotenv(override=True)
    _env_file_abs = str(Path(".env").resolve()) if Path(".env").exists() else ".env"


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI configuration settings."""
    
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    deployment_name: str = Field(default="gpt-4", description="Deployment name")
    api_version: str = Field(default="2023-05-15", description="API version")
    embedding_deployment: Optional[str] = Field(
        default=None,
        description="Embedding deployment name"
    )
    
    if PYDANTIC_V2:
        # Pydantic v2 reads from os.environ (already loaded by dotenv above)
        model_config = SettingsConfigDict(
            env_prefix="AZURE_OPENAI_",
            case_sensitive=False,
            extra="ignore"
        )
        
        @field_validator("endpoint")
        @classmethod
        def validate_endpoint(cls, v):
            """Validate that endpoint is a valid URL."""
            if not v.startswith("http"):
                raise ValueError("Azure OpenAI endpoint must be a valid URL")
            return v.rstrip("/")
    else:
        @validator("endpoint")
        def validate_endpoint(cls, v):
            """Validate that endpoint is a valid URL."""
            if not v.startswith("http"):
                raise ValueError("Azure OpenAI endpoint must be a valid URL")
            return v.rstrip("/")
        
        class Config:
            """Pydantic v1 configuration."""
            env_prefix = "AZURE_OPENAI_"
            case_sensitive = False


class RedisSettings(BaseSettings):
    """Redis cache configuration settings."""
    
    host: str = Field(..., description="Redis host")
    port: int = Field(default=6380, description="Redis port")
    password: str = Field(..., description="Redis password")
    ssl: bool = Field(default=True, description="Enable SSL")
    connection_string: Optional[str] = Field(
        default=None,
        description="Redis connection string"
    )
    
    if PYDANTIC_V2:
        # Pydantic v2 reads from os.environ (already loaded by dotenv above)
        model_config = SettingsConfigDict(
            env_prefix="REDIS_",
            case_sensitive=False,
            extra="ignore"
        )
        
        @field_validator("ssl", mode="before")
        @classmethod
        def parse_ssl(cls, v):
            """Parse SSL setting from string or boolean."""
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes")
            return bool(v)
    else:
        @validator("ssl", pre=True)
        def parse_ssl(cls, v):
            """Parse SSL setting from string or boolean."""
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes")
            return bool(v)
        
        class Config:
            """Pydantic v1 configuration."""
            env_prefix = "REDIS_"
            case_sensitive = False


class AppSettings(BaseSettings):
    """Application-wide settings."""
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Cache TTLs (in seconds)
    cache_ttl_sentiment: int = Field(default=86400, description="Sentiment cache TTL")  # 24 hours
    cache_ttl_stock: int = Field(default=3600, description="Stock data cache TTL")  # 1 hour (increased from 5 min)
    cache_ttl_news: int = Field(default=7200, description="News cache TTL")  # 2 hours (increased from 30 min)
    
    # RAG settings
    rag_top_k: int = Field(default=3, description="Number of similar articles to retrieve")
    rag_similarity_threshold: float = Field(default=0.01, description="Minimum similarity score for RAG retrieval (0.0-1.0). Lower values return more articles but may include less relevant ones. For RRF scores, use 0.01-0.03. For cosine similarity, use 0.3-0.7.")
    
    if PYDANTIC_V2:
        # Pydantic v2 reads from os.environ (already loaded by dotenv above)
        model_config = SettingsConfigDict(
            env_prefix="APP_",
            case_sensitive=False,
            extra="ignore"
        )
        
        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v):
            """Validate log level."""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v.upper() not in valid_levels:
                raise ValueError(f"Log level must be one of {valid_levels}")
            return v.upper()
    else:
        @validator("log_level")
        def validate_log_level(cls, v):
            """Validate log level."""
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v.upper() not in valid_levels:
                raise ValueError(f"Log level must be one of {valid_levels}")
            return v.upper()
        
        class Config:
            """Pydantic v1 configuration."""
            env_prefix = "APP_"
            case_sensitive = False


class Settings:
    """
    Main settings class that aggregates all configuration.
    
    This class provides a single point of access for all application settings,
    with proper validation and type checking.
    """
    
    def __init__(self):
        """Initialize settings from environment variables."""
        try:
            # For Pydantic v2, ensure env vars are available
            import os
            if PYDANTIC_V2:
                # Manually construct from environment variables
                self.azure_openai = AzureOpenAISettings(
                    endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
                    embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
                )
            else:
                self.azure_openai = AzureOpenAISettings()
        except Exception as e:
            raise ValueError(
                f"Azure OpenAI configuration error: {e}. "
                "Please check your .env file."
            )
        
        try:
            import os
            if PYDANTIC_V2:
                # Manually construct from environment variables
                redis_host = os.getenv("REDIS_HOST")
                redis_password = os.getenv("REDIS_PASSWORD")
                if redis_host and redis_password:
                    self.redis = RedisSettings(
                        host=redis_host,
                        port=int(os.getenv("REDIS_PORT", "6380")),
                        password=redis_password,
                        ssl=os.getenv("REDIS_SSL", "true").lower() in ("true", "1", "yes"),
                        connection_string=os.getenv("REDIS_CONNECTION_STRING")
                    )
                else:
                    self.redis = None
            else:
                self.redis = RedisSettings()
        except Exception as e:
            # Redis is optional, so we allow it to fail
            self.redis = None
        
        self.app = AppSettings()
    
    def is_redis_available(self) -> bool:
        """Check if Redis is configured and available."""
        return self.redis is not None
    
    def is_rag_available(self) -> bool:
        """Check if RAG is configured and available."""
        return (
            self.azure_openai.embedding_deployment is not None
            and self.azure_openai.embedding_deployment != ""
        )
    
    def is_azure_openai_available(self) -> bool:
        """Check if Azure OpenAI is configured and available."""
        return (
            self.azure_openai.endpoint is not None
            and self.azure_openai.endpoint != ""
            and self.azure_openai.api_key is not None
            and self.azure_openai.api_key != ""
            and self.azure_openai.deployment_name is not None
            and self.azure_openai.deployment_name != ""
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance (singleton pattern).
    
    Returns:
        Settings: The application settings instance
        
    Raises:
        ValueError: If required configuration is missing
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

