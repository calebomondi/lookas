import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("pyrmyd2.db")

DB_PATH = Path(__file__).parent / "orders.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE,
    negotiation_id TEXT,
    status TEXT NOT NULL DEFAULT 'received',
    topic TEXT,
    word_count INTEGER,
    max_analysts INTEGER,
    requester_agent_id TEXT,
    service_id TEXT,
    price TEXT,
    created_at TIMESTAMP NOT NULL,
    paid_at TIMESTAMP,
    completed_at TIMESTAMP,
    failed_at TIMESTAMP,
    error_message TEXT,
    report_length INTEGER
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create the orders table if it doesn't exist."""
    with _connect() as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", DB_PATH)


def record_order(
    order_id: str,
    negotiation_id: str,
    topic: str,
    word_count: int,
    max_analysts: int,
    requester_agent_id: str = "",
    service_id: str = "",
    price: str = "",
):
    """Insert a new order record with status 'received'."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO orders
                (order_id, negotiation_id, status, topic, word_count, max_analysts,
                 requester_agent_id, service_id, price, created_at)
            VALUES (?, ?, 'received', ?, ?, ?, ?, ?, ?, ?)
            """,
            (order_id, negotiation_id, topic, word_count, max_analysts,
             requester_agent_id, service_id, price, now),
        )
    logger.info("Recorded order %s (topic=%s)", order_id, topic)


def update_order(order_id: str, **kwargs):
    """Update order fields. Pass status, error_message, report_length, etc."""
    if not kwargs:
        return
    now = datetime.now(timezone.utc).isoformat()
    if "status" in kwargs:
        status = kwargs.pop("status")
        if status == "researching":
            kwargs["paid_at"] = now
        elif status == "completed":
            kwargs["completed_at"] = now
        elif status == "failed":
            kwargs["failed_at"] = now
        kwargs["status"] = status
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [order_id]
    with _connect() as conn:
        conn.execute(f"UPDATE orders SET {set_clause} WHERE order_id = ?", values)
    logger.info("Updated order %s: %s", order_id, kwargs)
