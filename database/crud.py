"""
CRUD operations for database models

Provides Create, Read, Update, Delete operations for all models

Author: Phase 12 Implementation
Date: October 2025
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from typing import Optional, List, Dict
from datetime import datetime

from .models import (
    User, Analysis, Job, Result, ApiKey, 
    Notification, AuditLog, SharedLink,
    UserRole, JobStatus, AnalysisType
)
from .auth import hash_password


# ============================================================================
# User CRUD
# ============================================================================

def create_user(
    db: Session,
    email: str,
    username: str,
    password: str,
    full_name: Optional[str] = None,
    role: UserRole = UserRole.USER
) -> User:
    """Create a new user"""
    hashed_password = hash_password(password)
    user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        full_name=full_name,
        role=role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()


def get_users(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    role: Optional[UserRole] = None,
    is_active: Optional[bool] = None
) -> List[User]:
    """Get list of users with filters"""
    query = db.query(User)
    
    if role:
        query = query.filter(User.role == role)
    if is_active is not None:
        query = query.filter(User.is_active == is_active)
    
    return query.offset(skip).limit(limit).all()


def update_user(db: Session, user_id: int, **kwargs) -> Optional[User]:
    """Update user fields"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None
    
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    
    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, user_id: int) -> bool:
    """Delete user (soft delete - deactivate)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return False
    
    user.is_active = False
    db.commit()
    return True


# ============================================================================
# Analysis CRUD
# ============================================================================

def create_analysis(
    db: Session,
    user_id: int,
    name: str,
    type: AnalysisType,
    config: dict,
    description: Optional[str] = None,
    tags: Optional[list] = None
) -> Analysis:
    """Create a new analysis"""
    analysis = Analysis(
        user_id=user_id,
        name=name,
        type=type,
        config=config,
        description=description,
        tags=tags or []
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def get_analysis(db: Session, analysis_id: int) -> Optional[Analysis]:
    """Get analysis by ID"""
    return db.query(Analysis).filter(Analysis.id == analysis_id).first()


def get_analyses(
    db: Session,
    user_id: Optional[int] = None,
    type: Optional[AnalysisType] = None,
    status: Optional[JobStatus] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Analysis]:
    """Get list of analyses with filters"""
    query = db.query(Analysis)
    
    if user_id:
        query = query.filter(Analysis.user_id == user_id)
    if type:
        query = query.filter(Analysis.type == type)
    if status:
        query = query.filter(Analysis.status == status)
    
    return query.order_by(desc(Analysis.created_at)).offset(skip).limit(limit).all()


def get_public_analyses(db: Session, skip: int = 0, limit: int = 100) -> List[Analysis]:
    """Get public analyses"""
    return db.query(Analysis).filter(
        Analysis.is_public == True
    ).order_by(desc(Analysis.created_at)).offset(skip).limit(limit).all()


def update_analysis(db: Session, analysis_id: int, **kwargs) -> Optional[Analysis]:
    """Update analysis fields"""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        return None
    
    for key, value in kwargs.items():
        if hasattr(analysis, key):
            setattr(analysis, key, value)
    
    db.commit()
    db.refresh(analysis)
    return analysis


def delete_analysis(db: Session, analysis_id: int) -> bool:
    """Delete analysis and all related data"""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        return False
    
    db.delete(analysis)
    db.commit()
    return True


# ============================================================================
# Job CRUD
# ============================================================================

def create_job(
    db: Session,
    user_id: int,
    job_type: str,
    job_id: str,
    parameters: dict,
    analysis_id: Optional[int] = None
) -> Job:
    """Create a new job"""
    job = Job(
        user_id=user_id,
        analysis_id=analysis_id,
        job_type=job_type,
        job_id=job_id,
        parameters=parameters
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: int) -> Optional[Job]:
    """Get job by ID"""
    return db.query(Job).filter(Job.id == job_id).first()


def get_job_by_job_id(db: Session, job_id: str) -> Optional[Job]:
    """Get job by job_id (UUID)"""
    return db.query(Job).filter(Job.job_id == job_id).first()


def get_jobs(
    db: Session,
    user_id: Optional[int] = None,
    analysis_id: Optional[int] = None,
    status: Optional[JobStatus] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Job]:
    """Get list of jobs with filters"""
    query = db.query(Job)
    
    if user_id:
        query = query.filter(Job.user_id == user_id)
    if analysis_id:
        query = query.filter(Job.analysis_id == analysis_id)
    if status:
        query = query.filter(Job.status == status)
    
    return query.order_by(desc(Job.created_at)).offset(skip).limit(limit).all()


def update_job_status(
    db: Session,
    job_id: str,
    status: JobStatus,
    progress: Optional[float] = None,
    error_message: Optional[str] = None
) -> Optional[Job]:
    """Update job status"""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        return None
    
    job.status = status
    if progress is not None:
        job.progress = progress
    if error_message:
        job.error_message = error_message
    
    # Update timestamps
    if status == JobStatus.RUNNING and not job.started_at:
        job.started_at = datetime.utcnow()
    if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        job.completed_at = datetime.utcnow()
        if job.started_at:
            duration = (datetime.utcnow() - job.started_at).total_seconds()
            job.duration_seconds = duration
    
    db.commit()
    db.refresh(job)
    return job


def delete_job(db: Session, job_id: int) -> bool:
    """Delete job"""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return False
    
    db.delete(job)
    db.commit()
    return True


# ============================================================================
# Result CRUD
# ============================================================================

def create_result(
    db: Session,
    job_id: int,
    result_type: str,
    data: dict,
    analysis_id: Optional[int] = None,
    result_metadata: Optional[dict] = None,
    file_path: Optional[str] = None
) -> Result:
    """Create a new result"""
    result = Result(
        job_id=job_id,
        analysis_id=analysis_id,
        result_type=result_type,
        data=data,
        result_metadata=result_metadata or {},
        file_path=file_path
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def get_result(db: Session, result_id: int) -> Optional[Result]:
    """Get result by ID"""
    return db.query(Result).filter(Result.id == result_id).first()


def get_results(
    db: Session,
    job_id: Optional[int] = None,
    analysis_id: Optional[int] = None,
    result_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Result]:
    """Get list of results with filters"""
    query = db.query(Result)
    
    if job_id:
        query = query.filter(Result.job_id == job_id)
    if analysis_id:
        query = query.filter(Result.analysis_id == analysis_id)
    if result_type:
        query = query.filter(Result.result_type == result_type)
    
    return query.order_by(desc(Result.created_at)).offset(skip).limit(limit).all()


def delete_result(db: Session, result_id: int) -> bool:
    """Delete result"""
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        return False
    
    db.delete(result)
    db.commit()
    return True


# ============================================================================
# Notification CRUD
# ============================================================================

def create_notification(
    db: Session,
    user_id: int,
    title: str,
    message: str,
    type: str = "info",
    job_id: Optional[int] = None,
    analysis_id: Optional[int] = None
) -> Notification:
    """Create a new notification"""
    notification = Notification(
        user_id=user_id,
        title=title,
        message=message,
        type=type,
        job_id=job_id,
        analysis_id=analysis_id
    )
    db.add(notification)
    db.commit()
    db.refresh(notification)
    return notification


def get_notifications(
    db: Session,
    user_id: int,
    is_read: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Notification]:
    """Get user notifications"""
    query = db.query(Notification).filter(Notification.user_id == user_id)
    
    if is_read is not None:
        query = query.filter(Notification.is_read == is_read)
    
    return query.order_by(desc(Notification.created_at)).offset(skip).limit(limit).all()


def mark_notification_read(db: Session, notification_id: int) -> Optional[Notification]:
    """Mark notification as read"""
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        return None
    
    notification.is_read = True
    notification.read_at = datetime.utcnow()
    db.commit()
    db.refresh(notification)
    return notification


def mark_all_notifications_read(db: Session, user_id: int) -> int:
    """Mark all user notifications as read"""
    count = db.query(Notification).filter(
        Notification.user_id == user_id,
        Notification.is_read == False
    ).update({
        "is_read": True,
        "read_at": datetime.utcnow()
    })
    db.commit()
    return count


# ============================================================================
# Audit Log CRUD
# ============================================================================

def create_audit_log(
    db: Session,
    user_id: Optional[int],
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[int] = None,
    changes: Optional[dict] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> AuditLog:
    """Create audit log entry"""
    log = AuditLog(
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        changes=changes,
        ip_address=ip_address,
        user_agent=user_agent
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_audit_logs(
    db: Session,
    user_id: Optional[int] = None,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
) -> List[AuditLog]:
    """Get audit logs with filters"""
    query = db.query(AuditLog)
    
    if user_id:
        query = query.filter(AuditLog.user_id == user_id)
    if action:
        query = query.filter(AuditLog.action == action)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    
    return query.order_by(desc(AuditLog.created_at)).offset(skip).limit(limit).all()


# ============================================================================
# Shared Link CRUD
# ============================================================================

def create_shared_link(
    db: Session,
    analysis_id: int,
    user_id: int,
    token: str,
    can_edit: bool = False,
    max_uses: Optional[int] = None,
    expires_at: Optional[datetime] = None
) -> SharedLink:
    """Create a shared link"""
    link = SharedLink(
        analysis_id=analysis_id,
        user_id=user_id,
        token=token,
        can_edit=can_edit,
        max_uses=max_uses,
        expires_at=expires_at
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    return link


def get_shared_link(db: Session, token: str) -> Optional[SharedLink]:
    """Get shared link by token"""
    return db.query(SharedLink).filter(
        SharedLink.token == token,
        SharedLink.is_active == True
    ).first()


def increment_shared_link_usage(db: Session, token: str) -> Optional[SharedLink]:
    """Increment shared link usage counter"""
    link = db.query(SharedLink).filter(SharedLink.token == token).first()
    if not link:
        return None
    
    link.use_count += 1
    link.last_accessed = datetime.utcnow()
    
    # Deactivate if max uses reached
    if link.max_uses and link.use_count >= link.max_uses:
        link.is_active = False
    
    db.commit()
    db.refresh(link)
    return link


def revoke_shared_link(db: Session, token: str) -> bool:
    """Revoke (deactivate) a shared link"""
    link = db.query(SharedLink).filter(SharedLink.token == token).first()
    if not link:
        return False
    
    link.is_active = False
    db.commit()
    return True


# ============================================================================
# Statistics and Analytics
# ============================================================================

def get_user_stats(db: Session, user_id: int) -> dict:
    """Get user statistics"""
    analyses_count = db.query(Analysis).filter(Analysis.user_id == user_id).count()
    jobs_count = db.query(Job).filter(Job.user_id == user_id).count()
    results_count = db.query(Result).join(Job).filter(Job.user_id == user_id).count()
    
    return {
        "analyses_count": analyses_count,
        "jobs_count": jobs_count,
        "results_count": results_count
    }


def get_system_stats(db: Session) -> dict:
    """Get system-wide statistics"""
    users_count = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    analyses_count = db.query(Analysis).count()
    jobs_count = db.query(Job).count()
    
    pending_jobs = db.query(Job).filter(Job.status == JobStatus.PENDING).count()
    running_jobs = db.query(Job).filter(Job.status == JobStatus.RUNNING).count()
    completed_jobs = db.query(Job).filter(Job.status == JobStatus.COMPLETED).count()
    
    return {
        "users": users_count,
        "active_users": active_users,
        "analyses": analyses_count,
        "jobs": jobs_count,
        "pending_jobs": pending_jobs,
        "running_jobs": running_jobs,
        "completed_jobs": completed_jobs
    }
