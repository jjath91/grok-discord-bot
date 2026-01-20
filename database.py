"""
Database module for persistent conversation storage and user profiles.
Uses aiosqlite for async SQLite operations.
"""
import aiosqlite
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger('GrokBot.Database')

DB_PATH = "bot_data.db"

# Number of new messages before triggering profile regeneration
PROFILE_REGEN_THRESHOLD = 20


async def init_db() -> None:
    """Initialize the database and create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Messages table - stores all conversation messages
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # User profiles table - stores generated profiles
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                profile_text TEXT,
                message_count INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for faster user lookups
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user_id
            ON messages(user_id)
        """)

        await db.commit()
        logger.info("Database initialized successfully")


async def save_message(
    user_id: int,
    role: str,
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict]] = None,
    tool_call_id: Optional[str] = None
) -> None:
    """Save a message to the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        await db.execute(
            """
            INSERT INTO messages (user_id, role, content, tool_calls, tool_call_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, role, content, tool_calls_json, tool_call_id)
        )
        await db.commit()


async def get_user_messages(
    user_id: int,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Retrieve message history for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        query = """
            SELECT role, content, tool_calls, tool_call_id, timestamp
            FROM messages
            WHERE user_id = ?
            ORDER BY timestamp ASC
        """
        if limit:
            query += f" LIMIT {limit}"

        async with db.execute(query, (user_id,)) as cursor:
            rows = await cursor.fetchall()

        messages = []
        for row in rows:
            msg = {
                "role": row["role"],
                "timestamp": row["timestamp"]
            }
            if row["content"]:
                msg["content"] = row["content"]
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            messages.append(msg)

        return messages


async def get_user_message_count(user_id: int) -> int:
    """Get the total number of messages for a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ?",
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0


async def save_profile(user_id: int, profile_text: str, message_count: int) -> None:
    """Save or update a user's profile."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO user_profiles (user_id, profile_text, message_count, last_updated)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_text = excluded.profile_text,
                message_count = excluded.message_count,
                last_updated = CURRENT_TIMESTAMP
            """,
            (user_id, profile_text, message_count)
        )
        await db.commit()
        logger.info(f"Profile saved for user {user_id}")


async def get_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a user's profile."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT profile_text, message_count, last_updated
            FROM user_profiles
            WHERE user_id = ?
            """,
            (user_id,)
        ) as cursor:
            row = await cursor.fetchone()

        if row:
            return {
                "profile_text": row["profile_text"],
                "message_count": row["message_count"],
                "last_updated": row["last_updated"]
            }
        return None


async def should_regenerate_profile(user_id: int) -> bool:
    """Check if a user's profile should be regenerated based on new messages."""
    profile = await get_profile(user_id)
    if not profile:
        # No profile exists, should generate if user has messages
        msg_count = await get_user_message_count(user_id)
        return msg_count >= PROFILE_REGEN_THRESHOLD

    current_count = await get_user_message_count(user_id)
    messages_since_update = current_count - profile["message_count"]
    return messages_since_update >= PROFILE_REGEN_THRESHOLD


async def delete_user_data(user_id: int) -> bool:
    """Delete all data for a user (GDPR-friendly forget command)."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        await db.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        await db.commit()
        logger.info(f"All data deleted for user {user_id}")
        return True
