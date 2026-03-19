import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.database import Database
from api.auth import Auth
from rag.config import Config


def create_admin():
    config = Config()
    database = Database(os.getenv("DATABASE_URL", "sqlite:///./data/app.db").replace("sqlite:///", ""))
    auth = Auth(database, config.jwt_secret)

    username = input("Enter admin username: ").strip()
    if not username:
        print("Username cannot be empty")
        return

    password = input("Enter admin password: ").strip()
    if not password or len(password) < 6:
        print("Password must be at least 6 characters")
        return

    confirm = input("Confirm password: ").strip()
    if password != confirm:
        print("Passwords do not match")
        return

    existing = database.get_user_by_username(username)
    if existing:
        print(f"User '{username}' already exists")
        return

    user = auth.register_user(username=username, password=password, is_admin=True)
    print(f"Admin user '{username}' created successfully!")
    print(f"User ID: {user.id}")


def create_user():
    config = Config()
    database = Database(os.getenv("DATABASE_URL", "sqlite:///./data/app.db").replace("sqlite:///", ""))
    auth = Auth(database, config.jwt_secret)

    username = input("Enter username: ").strip()
    if not username:
        print("Username cannot be empty")
        return

    password = input("Enter password: ").strip()
    if not password or len(password) < 6:
        print("Password must be at least 6 characters")
        return

    confirm = input("Confirm password: ").strip()
    if password != confirm:
        print("Passwords do not match")
        return

    email = input("Enter email (optional): ").strip() or None

    existing = database.get_user_by_username(username)
    if existing:
        print(f"User '{username}' already exists")
        return

    user = auth.register_user(username=username, password=password, email=email)
    print(f"User '{username}' created successfully!")
    print(f"User ID: {user.id}")


def list_users():
    database = Database(os.getenv("DATABASE_URL", "sqlite:///./data/app.db").replace("sqlite:///", ""))
    users = database.list_users()

    print(f"\nTotal users: {len(users)}")
    print("-" * 60)
    print(f"{'ID':<5} {'Username':<20} {'Email':<25} {'Admin':<6} {'Active':<6}")
    print("-" * 60)

    for user in users:
        email = user.email or ""
        print(f"{user.id:<5} {user.username:<20} {email[:23]:<25} {'Yes' if user.is_admin else 'No':<6} {'Yes' if user.is_active else 'No':<6}")


def delete_user():
    database = Database(os.getenv("DATABASE_URL", "sqlite:///./data/app.db").replace("sqlite:///", ""))

    username = input("Enter username to delete: ").strip()
    if not username:
        print("Username cannot be empty")
        return

    user = database.get_user_by_username(username)
    if not user:
        print(f"User '{username}' not found")
        return

    confirm = input(f"Are you sure you want to delete '{username}'? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("Cancelled")
        return

    database.delete_user(user.id)
    print(f"User '{username}' deleted successfully")


def reset_password():
    config = Config()
    database = Database(os.getenv("DATABASE_URL", "sqlite:///./data/app.db").replace("sqlite:///", ""))
    auth = Auth(database, config.jwt_secret)

    username = input("Enter username: ").strip()
    if not username:
        print("Username cannot be empty")
        return

    user = database.get_user_by_username(username)
    if not user:
        print(f"User '{username}' not found")
        return

    new_password = input("Enter new password: ").strip()
    if not new_password or len(new_password) < 6:
        print("Password must be at least 6 characters")
        return

    confirm = input("Confirm password: ").strip()
    if new_password != confirm:
        print("Passwords do not match")
        return

    password_hash = auth.hash_password(new_password)
    database.update_user(user.id, password_hash=password_hash)
    print(f"Password for '{username}' reset successfully")


def init_db():
    database = Database(os.getenv("DATABASE_URL", "sqlite:///./data/app.db").replace("sqlite:///", ""))
    print("Database initialized successfully")


def main():
    parser = argparse.ArgumentParser(description="RAG Internal Tool Admin CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    subparsers.add_parser("create-admin", help="Create admin user")
    subparsers.add_parser("create-user", help="Create regular user")
    subparsers.add_parser("list-users", help="List all users")
    subparsers.add_parser("delete-user", help="Delete a user")
    subparsers.add_parser("reset-password", help="Reset user password")
    subparsers.add_parser("init-db", help="Initialize database")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "create-admin": create_admin,
        "create-user": create_user,
        "list-users": list_users,
        "delete-user": delete_user,
        "reset-password": reset_password,
        "init-db": init_db,
    }

    commands[args.command]()


if __name__ == "__main__":
    main()
