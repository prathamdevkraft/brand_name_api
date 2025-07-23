#!/usr/bin/env python3
"""
Simplified configuration for Brand Identifier API
"""

import os
from typing import Optional

class Settings:
    """Simple settings class without Pydantic complexities"""
    
    def __init__(self):
        # API Configuration
        self.api_title = "AI-Enhanced Brand Identifier API"
        self.api_version = "1.0.0"
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Server Configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", 8000))
        
        # File Upload Limits
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))
        self.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 10))
        self.allowed_extensions = [".html", ".htm"]
        
        # AI Service Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        self.openai_max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", 10))
        self.openai_temperature = float(os.getenv("OPENAI_TEMPERATURE", 0.1))
        
        # spaCy Configuration
        self.spacy_model = os.getenv("SPACY_MODEL", "xx_ent_wiki_sm")
        self.spacy_enabled = os.getenv("SPACY_ENABLED", "true").lower() == "true"
        
        # Cache Configuration
        self.cache_enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", 3600))
        
        # Performance Configuration
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", 30))
        self.ai_timeout = int(os.getenv("AI_TIMEOUT", 10))

# Global settings instance
_settings = None

def get_settings():
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings