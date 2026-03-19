import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Optional

import jwt

from .database import Database, User


class Auth:
    def __init__(
        self,
        database: Database,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        jwt_expiration_hours: int = 24,
        jwt_refresh_days: int = 7,
        bcrypt_rounds: int = 12,
    ):
        self.database = database
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.jwt_expiration_hours = jwt_expiration_hours
        self.jwt_refresh_days = jwt_refresh_days
        self.bcrypt_rounds = bcrypt_rounds

    def hash_password(self, password: str) -> str:
        import bcrypt
        salt = bcrypt.gensalt(rounds=self.bcrypt_rounds)
        return bcrypt.hashpw(password.encode(), salt).decode()

    def verify_password(self, password: str, password_hash: str) -> bool:
        import bcrypt
        return bcrypt.checkpw(password.encode(), password_hash.encode())

    def create_access_token(self, user: User) -> str:
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "is_admin": user.is_admin,
            "exp": datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours),
            "iat": datetime.utcnow(),
            "type": "access",
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def create_refresh_token(self, user: User) -> str:
        payload = {
            "sub": str(user.id),
            "exp": datetime.utcnow() + timedelta(days=self.jwt_refresh_days),
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": str(uuid.uuid4()),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

    def verify_token(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def get_user_from_token(self, token: str) -> Optional[User]:
        payload = self.verify_token(token)
        if not payload:
            return None

        if payload.get("type") != "access":
            return None

        try:
            user_id = int(payload["sub"])
            return self.database.get_user_by_id(user_id)
        except (ValueError, TypeError):
            return None

    def register_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
    ) -> User:
        password_hash = self.hash_password(password)
        return self.database.create_user(
            username=username,
            password_hash=password_hash,
            email=email,
        )

    def login(self, username: str, password: str) -> Optional[dict]:
        user = self.database.get_user_by_username(username)
        if not user:
            return None

        if not user.is_active:
            return None

        if not self.verify_password(password, user.password_hash):
            return None

        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.jwt_expiration_hours * 3600,
            "user": user.to_dict(),
        }

    def refresh_access_token(self, refresh_token: str) -> Optional[dict]:
        payload = self.verify_token(refresh_token)
        if not payload:
            return None

        if payload.get("type") != "refresh":
            return None

        try:
            user_id = int(payload["sub"])
            user = self.database.get_user_by_id(user_id)
            if not user or not user.is_active:
                return None

            access_token = self.create_access_token(user)
            new_refresh_token = self.create_refresh_token(user)

            return {
                "access_token": access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
                "expires_in": self.jwt_expiration_hours * 3600,
            }
        except (ValueError, TypeError):
            return None
