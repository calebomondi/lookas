import asyncio
import logging
from contextlib import asynccontextmanager

import db
from config import Config
from research_agent import build_agent
from provider_handler import ResearchProviderHandler

from fastapi import FastAPI
from croo import AgentClient, Config as CrooConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("pyrmyd2.web")

handler: ResearchProviderHandler | None = None
agent = None


async def _start_croo():
    """Connect to Croo and start listening for events."""
    global handler
    cfg = Config.from_env()

    if not cfg.croo_sdk_key:
        logger.warning("CROO_SDK_KEY not set — Croo listener not started")
        return

    if not cfg.google_api_key and not cfg.groq_api_key:
        logger.error("No LLM API key set — Croo listener not started")
        return

    agent = build_agent()
    croo_cfg = CrooConfig(base_url=cfg.croo_api_url, ws_url=cfg.croo_ws_url)
    client = AgentClient(croo_cfg, cfg.croo_sdk_key)

    handler = ResearchProviderHandler(client, agent)
    await handler.start()
    logger.info("Croo WebSocket listener started")


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    task = asyncio.create_task(_start_croo())
    yield
    if handler:
        await handler.stop()
    task.cancel()


app = FastAPI(title="PYRMYD2", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "ok", "service": "pyrmyd2"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
