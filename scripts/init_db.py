"""
Database initialization script

Creates all tables and optionally creates an admin user

Usage:
    python scripts/init_db.py
    python scripts/init_db.py --create-admin
    python scripts/init_db.py --admin-email admin@example.com --admin-password securepass

Author: Phase 12 Implementation
Date: October 2025
"""


# Removed hacky sys.path.insert - package is installed


from database import (
    init_db,
    check_db_connection,
    get_db_context,
    create_user,
    get_user_by_email,
    UserRole
)
import argparse


def main():
    """Initialize database and optionally create admin user"""
    parser = argparse.ArgumentParser(description="Initialize database")
    parser.add_argument(
        "--create-admin",
        action="store_true",
        help="Create an admin user"
    )
    parser.add_argument(
        "--admin-email",
        default="admin@lensing.com",
        help="Admin email address"
    )
    parser.add_argument(
        "--admin-username",
        default="admin",
        help="Admin username"
    )
    parser.add_argument(
        "--admin-password",
        default=None,
        help="Admin password (or set ADMIN_PASSWORD env var)"
    )
    parser.add_argument(
        "--admin-name",
        default="Administrator",
        help="Admin full name"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Gravitational Lensing Platform - Database Initialization")
    print("=" * 60)
    
    # Check database connection
    print("\n1. Checking database connection...")
    if not check_db_connection():
        print("❌ Database connection failed!")
        print("   Please check your DATABASE_URL environment variable")
        print("   Current default: postgresql://lensing_user:lensing_password@localhost:5432/lensing_db")
        sys.exit(1)
    print("✅ Database connection successful")
    
    # Create tables
    print("\n2. Creating database tables...")
    try:
        init_db()
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        sys.exit(1)
    
    # Create admin user if requested
    if args.create_admin:
        print("\n3. Creating admin user...")
        # Get password from args or env
        admin_password = args.admin_password or os.getenv("ADMIN_PASSWORD")
        if not admin_password:
            print("❌ Error: Admin password required!")
            print("   Please provide --admin-password OR set ADMIN_PASSWORD environment variable.")
            sys.exit(1)

        with get_db_context() as db:
            # Check if admin already exists
            existing_admin = get_user_by_email(db, args.admin_email)
            if existing_admin:
                print(f"⚠️  Admin user already exists: {args.admin_email}")
                print(f"   User ID: {existing_admin.id}")
                print(f"   Username: {existing_admin.username}")
                print(f"   Role: {existing_admin.role}")
            else:
                try:
                    admin_user = create_user(
                        db=db,
                        email=args.admin_email,
                        username=args.admin_username,
                        password=admin_password,
                        full_name=args.admin_name,
                        role=UserRole.ADMIN
                    )
                    print("✅ Admin user created successfully")
                    print(f"   Email: {admin_user.email}")
                    print(f"   Username: {admin_user.username}")
                    print(f"   User ID: {admin_user.id}")
                    print(f"   Role: {admin_user.role}")
                    print("\n⚠️  IMPORTANT: Change the admin password immediately!")
                except Exception as e:
                    print(f"❌ Failed to create admin user: {e}")
                    sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Database initialization complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the API server: uvicorn api.main:app --reload")
    print("2. Visit the API docs: http://localhost:8000/docs")
    print("3. Login with your credentials to get an access token")
    print("4. Use the token to access protected endpoints")
    
    if args.create_admin:
        print(f"\nAdmin credentials:")
        print(f"  Email: {args.admin_email}")
        print(f"  Username: {args.admin_username}")
        # print(f"  Password: {admin_password}") # Don't print password in logs
        print("\n⚠️  Change the password immediately for security!")


if __name__ == "__main__":
    main()
