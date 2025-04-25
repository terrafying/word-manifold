"""
Configuration management for word-manifold using Pydantic.

This module provides strongly typed configuration management with validation
and environment variable support.
"""

from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import torch
from pathlib import Path

class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    file: Optional[Path] = None
    max_size: str = "10MB"
    backup_count: int = 5

    class Config:
        env_prefix = "WM_LOG_"

class SecuritySettings(BaseSettings):
    """Security and authentication settings."""
    enable_cors: bool = True
    allowed_origins: List[str] = ["*"]
    require_auth: bool = False
    api_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    
    class Config:
        env_prefix = "WM_SECURITY_"

class ServerSettings(BaseSettings):
    """Server configuration settings."""
    host: str = "localhost"
    port: int = 5000
    workers: int = 4
    debug: bool = False
    timeout: int = 30
    reload: bool = False
    
    class Config:
        env_prefix = "WM_SERVER_"

class ModelSettings(BaseSettings):
    """ML model configuration settings."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"
    spacy_model: str = "en_core_web_lg"
    device: str = Field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 32
    max_sequence_length: int = 512
    cache_dir: Optional[Path] = None
    
    class Config:
        env_prefix = "WM_MODEL_"
    
    @validator("device")
    def validate_device(cls, v: str) -> str:
        """Validate and normalize device specification."""
        if v == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if v == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return v

class CacheSettings(BaseSettings):
    """Cache configuration settings."""
    enable: bool = True
    type: str = "memory"  # memory, redis, filesystem
    max_size: int = 10000
    ttl: int = 3600  # seconds
    redis_url: Optional[str] = None
    filesystem_path: Optional[Path] = None
    
    class Config:
        env_prefix = "WM_CACHE_"

class Settings(BaseSettings):
    """Main configuration settings."""
    app_name: str = "word-manifold"
    version: str = "0.1.0"
    debug: bool = False
    environment: str = "development"
    
    # Nested configurations
    logging: LoggingSettings = LoggingSettings()
    security: SecuritySettings = SecuritySettings()
    server: ServerSettings = ServerSettings()
    model: ModelSettings = ModelSettings()
    cache: CacheSettings = CacheSettings()
    
    # Additional settings
    output_dir: Path = Path("output")
    temp_dir: Path = Path("/tmp/word-manifold")
    
    class Config:
        env_prefix = "WM_"
        
    def configure_logging(self) -> None:
        """Configure logging based on settings."""
        import logging
        import logging.handlers
        
        # Create formatter
        formatter = logging.Formatter(self.logging.format)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.logging.level)
        
        # Always add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler if configured
        if self.logging.file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.logging.file,
                maxBytes=self.logging.max_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

# Create global settings instance
settings = Settings() 