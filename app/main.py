#!/usr/bin/env python3
"""
Simple FastAPI Brand Identifier API
Upload HTML files and get brand names back
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import List, Optional
import time
import uuid

from app.models import BrandResponse, ErrorResponse, HealthResponse
from app.services import AIBrandIdentifierService
from app.simple_config import get_settings

# Initialize FastAPI app
app = FastAPI(
    title="Brand Identifier API",
    description="Upload HTML files to identify pharmaceutical brand names",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = get_settings()
brand_service = AIBrandIdentifierService(settings)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Brand Identifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0"
    )

@app.post("/identify", response_model=BrandResponse)
async def identify_brand(file: UploadFile = File(...)):
    """
    Upload an HTML file and get the brand name identified
    
    Args:
        file: HTML file to analyze
        
    Returns:
        BrandResponse with identified brand name and confidence
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith(('.html', '.htm')):
            raise HTTPException(
                status_code=400, 
                detail="Only HTML files are supported (.html, .htm)"
            )
        
        # Check file size (max 10MB)
        file_content = await file.read()
        if len(file_content) > settings.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Max size: {settings.max_file_size} bytes"
            )
        
        # Save file temporarily and process
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.html', delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Identify brand (using async AI service)
            result = await brand_service.identify_brand(tmp_file_path, file.filename)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return BrandResponse(
                request_id=request_id,
                brand_name=result['brand'],
                confidence=result['confidence'],
                method=result['method'],
                processing_time_ms=processing_time,
                filename=file.filename,
                file_size=len(file_content),
                metadata=result.get('details', {})
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    except HTTPException:
        raise
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.post("/identify-batch", response_model=List[BrandResponse])
async def identify_brands_batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple HTML files and get brand names identified
    
    Args:
        files: List of HTML files to analyze
        
    Returns:
        List of BrandResponse objects
    """
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max batch size: {settings.max_batch_size}"
        )
    
    results = []
    
    for file in files:
        try:
            # Process each file individually
            result = await identify_brand(file)
            results.append(result)
        except HTTPException as e:
            # Add error result for failed files
            results.append(BrandResponse(
                request_id=str(uuid.uuid4()),
                brand_name="ERROR",
                confidence=0.0,
                method="error",
                processing_time_ms=0,
                filename=file.filename or "unknown",
                file_size=0,
                metadata={"error": e.detail}
            ))
    
    return results

@app.post("/identify-text", response_model=BrandResponse)
async def identify_brand_from_text(
    html_content: str,
    filename: Optional[str] = "uploaded.html"
):
    """
    Identify brand from HTML content as text
    
    Args:
        html_content: HTML content as string
        filename: Optional filename for context
        
    Returns:
        BrandResponse with identified brand name
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        if len(html_content) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"Content too large. Max size: {settings.max_file_size} bytes"
            )
        
        # Save content to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(html_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Identify brand (using async AI service)
            result = await brand_service.identify_brand(tmp_file_path, filename)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return BrandResponse(
                request_id=request_id,
                brand_name=result['brand'],
                confidence=result['confidence'],
                method=result['method'],
                processing_time_ms=processing_time,
                filename=filename,
                file_size=len(html_content.encode('utf-8')),
                metadata=result.get('details', {})
            )
            
        finally:
            # Clean up
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing content: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=time.time()
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )