"""
Database package for Gravitational Lensing Analysis Platform

Provides database models, authentication, CRUD operations, and connection management

Author: Phase 12 Implementation
Date: October 2025
"""

from .models import (
    Base,
    User,
    Analysis,
    Job,
    Result,
    ApiKey,
    Notification,
    AuditLog,
    SharedLink,
    UserRole,
    JobStatus,
    AnalysisType,
)

from .database import (
    engine,
    SessionLocal,
    get_db,
    get_db_context,
    init_db,
    drop_db,
    check_db_connection,
    get_db_info,
)

from .auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    authenticate_user,
    get_current_user,
    get_current_active_user,
    get_current_admin_user,
    get_current_researcher_user,
    generate_api_key,
    create_api_key,
    revoke_api_key,
)

from .crud import (
    # User CRUD
    create_user,
    get_user,
    get_user_by_email,
    get_user_by_username,
    get_users,
    update_user,
    delete_user,
    
    # Analysis CRUD
    create_analysis,
    get_analysis,
    get_analyses,
    get_public_analyses,
    update_analysis,
    delete_analysis,
    
    # Job CRUD
    create_job,
    get_job,
    get_job_by_job_id,
    get_jobs,
    update_job_status,
    delete_job,
    
    # Result CRUD
    create_result,
    get_result,
    get_results,
    delete_result,
    
    # Notification CRUD
    create_notification,
    get_notifications,
    mark_notification_read,
    mark_all_notifications_read,
    
    # Audit Log CRUD
    create_audit_log,
    get_audit_logs,
    
    # Shared Link CRUD
    create_shared_link,
    get_shared_link,
    increment_shared_link_usage,
    revoke_shared_link,
    
    # Statistics
    get_user_stats,
    get_system_stats,
)

__all__ = [
    # Models
    "Base",
    "User",
    "Analysis",
    "Job",
    "Result",
    "ApiKey",
    "Notification",
    "AuditLog",
    "SharedLink",
    "UserRole",
    "JobStatus",
    "AnalysisType",
    
    # Database
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_context",
    "init_db",
    "drop_db",
    "check_db_connection",
    "get_db_info",
    
    # Authentication
    "hash_password",
    "verify_password",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "authenticate_user",
    "get_current_user",
    "get_current_active_user",
    "get_current_admin_user",
    "get_current_researcher_user",
    "generate_api_key",
    "create_api_key",
    "revoke_api_key",
    
    # CRUD Operations
    "create_user",
    "get_user",
    "get_user_by_email",
    "get_user_by_username",
    "get_users",
    "update_user",
    "delete_user",
    "create_analysis",
    "get_analysis",
    "get_analyses",
    "get_public_analyses",
    "update_analysis",
    "delete_analysis",
    "create_job",
    "get_job",
    "get_job_by_job_id",
    "get_jobs",
    "update_job_status",
    "delete_job",
    "create_result",
    "get_result",
    "get_results",
    "delete_result",
    "create_notification",
    "get_notifications",
    "mark_notification_read",
    "mark_all_notifications_read",
    "create_audit_log",
    "get_audit_logs",
    "create_shared_link",
    "get_shared_link",
    "increment_shared_link_usage",
    "revoke_shared_link",
    "get_user_stats",
    "get_system_stats",
]
