"""
Analysis routes for FastAPI

Provides endpoints for managing analyses, jobs, and results

Author: Phase 12 Implementation
Date: October 2025
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
from datetime import datetime

from pathlib import Path

from database import (
    get_db,
    get_current_user,
    get_current_active_user,
    User,
    AnalysisType,
    JobStatus,
    create_analysis,
    get_analysis,
    get_analyses,
    get_public_analyses,
    update_analysis,
    delete_analysis,
    create_job,
    get_job_by_job_id,
    get_jobs,
    update_job_status,
    create_result,
    get_results,
    create_notification,
    create_audit_log,
    get_user_stats
)


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/api/v1", tags=["Analysis"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalysisCreate(BaseModel):
    """Analysis creation request"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    type: str = Field(..., description="synthetic, real_data, inference, batch, custom")
    config: Dict[str, Any] = Field(..., description="Analysis configuration")
    tags: Optional[List[str]] = None


class AnalysisResponse(BaseModel):
    """Analysis response"""
    id: int
    user_id: int
    name: str
    description: Optional[str]
    type: str
    config: Dict[str, Any]
    status: str
    progress: float
    is_public: bool
    tags: List[str]
    created_at: str
    updated_at: Optional[str]
    completed_at: Optional[str]
    
    class Config:
        from_attributes = True


class AnalysisUpdate(BaseModel):
    """Analysis update request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None


class JobResponse(BaseModel):
    """Job response"""
    id: int
    job_id: str
    job_type: str
    status: str
    progress: float
    error_message: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_seconds: Optional[float]
    
    class Config:
        from_attributes = True


class ResultResponse(BaseModel):
    """Result response"""
    id: int
    job_id: int
    result_type: str
    data: Dict[str, Any]
    result_metadata: Dict[str, Any]
    confidence_score: Optional[float]
    created_at: str
    
    class Config:
        from_attributes = True


class UserStatsResponse(BaseModel):
    """User statistics response"""
    analyses_count: int
    jobs_count: int
    results_count: int


# ============================================================================
# Analysis Endpoints
# ============================================================================

@router.post("/analyses", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def create_new_analysis(
    analysis_data: AnalysisCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new analysis
    
    Creates a new analysis session for the current user.
    """
    # Validate analysis type
    try:
        analysis_type = AnalysisType(analysis_data.type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid analysis type. Must be one of: {[t.value for t in AnalysisType]}"
        )
    
    # Create analysis
    analysis = create_analysis(
        db=db,
        user_id=current_user.id,
        name=analysis_data.name,
        type=analysis_type,
        config=analysis_data.config,
        description=analysis_data.description,
        tags=analysis_data.tags
    )
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="create_analysis",
        resource_type="analysis",
        resource_id=analysis.id
    )
    
    # Create notification
    create_notification(
        db=db,
        user_id=current_user.id,
        title="Analysis Created",
        message=f"Analysis '{analysis.name}' has been created",
        type="success",
        analysis_id=analysis.id
    )
    
    return analysis


@router.get("/analyses", response_model=List[AnalysisResponse])
async def list_analyses(
    type: Optional[str] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List user's analyses
    
    Returns a paginated list of analyses for the current user.
    """
    # Parse filters
    analysis_type = None
    if type:
        try:
            analysis_type = AnalysisType(type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid analysis type. Must be one of: {[t.value for t in AnalysisType]}"
            )
    
    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {[s.value for s in JobStatus]}"
            )
    
    # Get analyses
    analyses = get_analyses(
        db=db,
        user_id=current_user.id,
        type=analysis_type,
        status=job_status,
        skip=skip,
        limit=limit
    )
    
    return analyses


@router.get("/analyses/public", response_model=List[AnalysisResponse])
async def list_public_analyses(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List public analyses
    
    Returns a paginated list of public analyses.
    """
    analyses = get_public_analyses(db=db, skip=skip, limit=limit)
    return analyses


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_by_id(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis by ID
    
    Returns detailed information about a specific analysis.
    FIXED P0 SECURITY: Now properly checks ownership before returning data.
    """
    analysis = get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # P0 FIX: Check ownership or public access
    # This prevents IDOR vulnerability where users could access other users' private analyses
    if analysis.user_id != current_user.id and not analysis.is_public:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this analysis"
        )
    
    return analysis


@router.put("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def update_analysis_by_id(
    analysis_id: int,
    update_data: AnalysisUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update analysis
    
    Updates an existing analysis.
    """
    analysis = get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # Check ownership
    if analysis.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this analysis"
        )
    
    # Update fields
    update_dict = {}
    if update_data.name is not None:
        update_dict["name"] = update_data.name
    if update_data.description is not None:
        update_dict["description"] = update_data.description
    if update_data.is_public is not None:
        update_dict["is_public"] = update_data.is_public
    if update_data.tags is not None:
        update_dict["tags"] = update_data.tags
    
    updated_analysis = update_analysis(db, analysis_id, **update_dict)
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="update_analysis",
        resource_type="analysis",
        resource_id=analysis_id,
        changes=update_dict
    )
    
    return updated_analysis


@router.delete("/analyses/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis_by_id(
    analysis_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete analysis
    
    Deletes an analysis and all related data.
    """
    analysis = get_analysis(db, analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # Check ownership
    if analysis.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this analysis"
        )
    
    # Delete analysis
    success = delete_analysis(db, analysis_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete analysis"
        )
    
    # Create audit log
    create_audit_log(
        db=db,
        user_id=current_user.id,
        action="delete_analysis",
        resource_type="analysis",
        resource_id=analysis_id
    )
    
    return None


# ============================================================================
# Job Endpoints
# ============================================================================

@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    analysis_id: Optional[int] = None,
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List user's jobs
    
    Returns a paginated list of jobs for the current user.
    """
    # Parse status filter
    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {[s.value for s in JobStatus]}"
            )
    
    # Get jobs
    jobs = get_jobs(
        db=db,
        user_id=current_user.id,
        analysis_id=analysis_id,
        status=job_status,
        skip=skip,
        limit=limit
    )
    
    return jobs


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_by_id(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get job by ID
    
    Returns detailed information about a specific job.
    """
    job = get_job_by_job_id(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    # Check ownership
    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this job"
        )
    
    return job


# ============================================================================
# Result Endpoints
# ============================================================================

@router.get("/results", response_model=List[ResultResponse])
async def list_results(
    job_id: Optional[int] = None,
    analysis_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    List results
    
    Returns a paginated list of results.
    """
    # Verify ownership if filtering by job or analysis
    if job_id:
        from database import get_job
        job = get_job(db, job_id)
        if not job or job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access these results"
            )
    
    if analysis_id:
        analysis = get_analysis(db, analysis_id)
        if not analysis or analysis.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access these results"
            )
    
    # Get results
    results = get_results(
        db=db,
        job_id=job_id,
        analysis_id=analysis_id,
        skip=skip,
        limit=limit
    )
    
    return results


# ============================================================================
# Statistics Endpoints
# ============================================================================

@router.get("/stats", response_model=UserStatsResponse)
async def get_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user statistics
    
    Returns statistics about the current user's activity.
    """
    stats = get_user_stats(db, current_user.id)
    return stats
