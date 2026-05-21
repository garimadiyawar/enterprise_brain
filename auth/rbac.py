"""
auth/rbac.py
Lightweight role-based access control.
Users stored in TinyDB. Passwords bcrypt-hashed. JWT sessions.
"""
import jwt
import bcrypt
from datetime import datetime, timedelta
from tinydb import TinyDB, Query
from loguru import logger

from config import BASE_DIR, JWT_SECRET, JWT_EXPIRE_MIN, ROLES

AUTH_DB_PATH = BASE_DIR / "auth.json"


class AuthManager:
    """
    Manages users, roles, and JWT tokens.
    Thread-safe enough for Streamlit (single-process).
    """

    def __init__(self):
        self.db    = TinyDB(AUTH_DB_PATH)
        self.users = self.db.table("users")
        self._seed_admin()

    # ── User Management ───────────────────────────────────────────────────────

    def create_user(
        self,
        username: str,
        password: str,
        role:     str = "viewer",
        full_name: str = "",
    ) -> dict:
        User = Query()
        if self.users.search(User.username == username):
            raise ValueError(f"User '{username}' already exists.")
        if role not in ROLES:
            raise ValueError(f"Invalid role '{role}'. Valid: {list(ROLES.keys())}")

        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        record = {
            "username":   username,
            "pw_hash":    pw_hash,
            "role":       role,
            "full_name":  full_name,
            "created_at": datetime.utcnow().isoformat(),
            "active":     True,
        }
        self.users.insert(record)
        logger.info(f"User created: {username} ({role})")
        return {"username": username, "role": role}

    def delete_user(self, username: str):
        User = Query()
        self.users.remove(User.username == username)
        logger.info(f"User deleted: {username}")

    def update_role(self, username: str, new_role: str):
        if new_role not in ROLES:
            raise ValueError(f"Invalid role: {new_role}")
        User = Query()
        self.users.update({"role": new_role}, User.username == username)
        logger.info(f"Role updated: {username} → {new_role}")

    def list_users(self) -> list[dict]:
        return [
            {k: v for k, v in u.items() if k != "pw_hash"}
            for u in self.users.all()
        ]

    # ── Auth ──────────────────────────────────────────────────────────────────

    def authenticate(self, username: str, password: str) -> str | None:
        """Returns JWT token string on success, None on failure."""
        User   = Query()
        record = self.users.search(User.username == username)
        if not record:
            return None
        user = record[0]
        if not user.get("active", True):
            return None
        if not bcrypt.checkpw(password.encode(), user["pw_hash"].encode()):
            return None

        token = self._generate_token(username, user["role"])
        logger.info(f"Auth success: {username}")
        return token

    def verify_token(self, token: str) -> dict | None:
        """Returns {username, role} if valid, None otherwise."""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return {"username": payload["sub"], "role": payload["role"]}
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired.")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    # ── Permissions ───────────────────────────────────────────────────────────

    def can(self, role: str, action: str) -> bool:
        """Check if role has permission for action."""
        perms = ROLES.get(role, ROLES["viewer"])
        return bool(perms.get(action, False))

    def allowed_collections(self, role: str) -> list[str]:
        perms = ROLES.get(role, ROLES["viewer"])
        return perms.get("collections", ["public"])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_token(self, username: str, role: str) -> str:
        payload = {
            "sub":  username,
            "role": role,
            "exp":  datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN),
            "iat":  datetime.utcnow(),
        }
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    def _seed_admin(self):
        """Create default admin on first run."""
        User = Query()
        if not self.users.search(User.username == "admin"):
            try:
                self.create_user("admin", "admin123", "admin", "System Admin")
                logger.warning("Default admin created. Change password immediately!")
            except Exception:
                pass
