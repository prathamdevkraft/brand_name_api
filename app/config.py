#!/usr/bin/env python3
"""
Configuration for AI-Enhanced Brand Identifier API
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings with AI configurations"""
    
    # API Configuration
    api_title: str = "AI-Enhanced Brand Identifier API"
    api_version: str = "1.0.0"
    api_description: str = "Upload HTML files to identify pharmaceutical brand names using AI"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # File Upload Limits
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    max_batch_size: int = Field(default=10, description="Max number of files in batch")
    allowed_extensions: list = Field(default=[".html", ".htm"], description="Allowed file extensions")
    
    # AI Service Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key for LLM extraction", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
    openai_max_tokens: int = Field(default=10, description="Max tokens for OpenAI response")
    openai_temperature: float = Field(default=0.1, description="Temperature for OpenAI model")
    
    # spaCy Configuration
    spacy_model: str = Field(default="en_core_web_sm", description="spaCy model name")
    spacy_enabled: bool = Field(default=True, description="Enable spaCy NER extraction")
    
    # Transformers Configuration
    transformers_model: str = Field(
        default="dbmdz/bert-large-cased-finetuned-conll03-english",
        description="Transformers NER model"
    )
    transformers_enabled: bool = Field(default=True, description="Enable Transformers NER")
    transformers_threshold: float = Field(default=0.8, description="Confidence threshold for entities")
    
    # Cache Configuration
    cache_enabled: bool = Field(default=True, description="Enable in-memory caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, description="Max cache entries")
    
    # Performance Configuration
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    ai_timeout: int = Field(default=10, description="AI service timeout in seconds")
    max_concurrent_ai_requests: int = Field(default=5, description="Max concurrent AI requests")
    
    # Extraction Method Weights (for ensemble voting)
    method_weights: dict = Field(
        default={
            'filename': 0.9,
            'patterns': 0.8,
            'structure': 0.7,
            'openai_llm': 1.0,
            'spacy_ner': 0.85,
            'transformers_ner': 0.9,
            'text_frequency': 0.6
        },
        description="Weights for different extraction methods"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    # Security Configuration
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: list = Field(default=["*"], description="Allowed CORS origins")
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    
    # Health Check Configuration
    health_check_enabled: bool = Field(default=True, description="Enable health check endpoint")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def reload_settings():
    """Reload settings (useful for testing)"""
    global _settings
    _settings = None
    return get_settings()

# Validation functions
def validate_ai_services(settings: Settings) -> dict:
    """Validate AI service availability"""
    status = {
        'openai': False,
        'spacy': False,
        'transformers': False,
        'errors': []
    }
    
    # Check OpenAI
    if settings.openai_api_key:
        try:
            import openai
            openai.api_key = settings.openai_api_key
            status['openai'] = True
        except ImportError:
            status['errors'].append("OpenAI library not installed")
        except Exception as e:
            status['errors'].append(f"OpenAI setup error: {e}")
    else:
        status['errors'].append("OpenAI API key not provided")
    
    # Check spaCy
    if settings.spacy_enabled:
        try:
            import spacy
            spacy.load(settings.spacy_model)
            status['spacy'] = True
        except ImportError:
            status['errors'].append("spaCy library not installed")
        except OSError:
            status['errors'].append(f"spaCy model '{settings.spacy_model}' not found")
        except Exception as e:
            status['errors'].append(f"spaCy setup error: {e}")
    
    # Check Transformers
    if settings.transformers_enabled:
        try:
            from transformers import pipeline
            pipeline("ner", model=settings.transformers_model)
            status['transformers'] = True
        except ImportError:
            status['errors'].append("Transformers library not installed")
        except Exception as e:
            status['errors'].append(f"Transformers setup error: {e}")
    
    return status

# Example configuration for different environments
class DevelopmentSettings(Settings):
    """Development environment settings"""
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"
    cache_enabled: bool = False

class ProductionSettings(Settings):
    """Production environment settings"""
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    rate_limit_enabled: bool = True
    cors_origins: list = ["https://yourdomain.com"]

class TestingSettings(Settings):
    """Testing environment settings"""
    debug: bool = True
    cache_enabled: bool = False
    openai_api_key: str = "test_key"
    spacy_enabled: bool = False
    transformers_enabled: bool = False

def get_settings_for_environment(env: str = "development") -> Settings:
    """Get settings for specific environment"""
    env = env.lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()