#!/usr/bin/env python3
"""
Pydantic models for request/response schemas
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

class BrandResponse(BaseModel):
    """Response model for brand identification"""
    request_id: str = Field(..., description="Unique request identifier")
    brand_name: str = Field(..., description="Identified brand name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    method: str = Field(..., description="Extraction method used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req_123456789",
                "brand_name": "Biktarvy",
                "confidence": 0.95,
                "method": "ensemble",
                "processing_time_ms": 245,
                "filename": "biktarvy_english.html",
                "file_size": 15420,
                "metadata": {
                    "contributing_methods": ["filename", "patterns", "ai"],
                    "detected_language": "en"
                }
            }
        }

class BatchBrandResponse(BaseModel):
    """Response model for batch brand identification"""
    total_files: int = Field(..., description="Total number of files processed")
    successful: int = Field(..., description="Number of successfully processed files")
    failed: int = Field(..., description="Number of failed files")
    results: List[BrandResponse] = Field(..., description="Individual results")
    total_processing_time_ms: int = Field(..., description="Total processing time")

class TextIdentificationRequest(BaseModel):
    """Request model for text-based identification"""
    html_content: str = Field(..., description="HTML content as string")
    filename: Optional[str] = Field(default="uploaded.html", description="Optional filename")
    
    @validator('html_content')
    def validate_html_content(cls, v):
        if not v or not v.strip():
            raise ValueError("HTML content cannot be empty")
        if len(v) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("HTML content too large (max 10MB)")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "html_content": "<!DOCTYPE html><html><head><title>Biktarvy Information</title></head><body>...</body></html>",
                "filename": "biktarvy_info.html"
            }
        }

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: Optional[float] = Field(default=None, description="Service uptime in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1640995200.0,
                "version": "1.0.0",
                "uptime_seconds": 3600.5
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: float = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request ID if available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "File format not supported",
                "timestamp": 1640995200.0,
                "request_id": "req_123456789"
            }
        }

class BrandIdentificationResult(BaseModel):
    """Internal model for brand identification results"""
    brand: str
    confidence: float
    method: str
    details: Optional[Dict[str, Any]] = None
    
class APIInfo(BaseModel):
    """API information model"""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Brand Identifier API",
                "version": "1.0.0",
                "description": "Upload HTML files to identify pharmaceutical brand names",
                "endpoints": {
                    "POST /identify": "Identify brand from single HTML file",
                    "POST /identify-batch": "Identify brands from multiple HTML files",
                    "POST /identify-text": "Identify brand from HTML content",
                    "GET /health": "Health check endpoint",
                    "GET /docs": "API documentation"
                }
            }
        }