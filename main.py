import asyncio
import logging
import signal
from croo import AgentClient, Config as CrooConfig

from config import Config
from research_agent import build_agent
from provider_handler import ResearchProviderHandler
import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("pyrmyd2")


async def main():
    cfg = Config.from_env()
    db.init_db()

    # IF CROO NOT CONFIGURED 
    if not cfg.croo_sdk_key:
        logger.warning("CROO_SDK_KEY not set — running in standalone mode")
        # Thread
        thread = {"configurable": {"thread_id": "1"}}
        # Agent
        agent = build_agent()
        result = await agent.ainvoke({
            "topic": "Quantum computing breakthroughs in 2025",
            "max_analysts": 2,
            "word_count": 500,
        }, thread)
        # Final Report
        logger.info("Standalone result final_report length: %d", len(result.get("final_report", "")))
        print("\n--- FINAL REPORT ---\n")
        print(result.get("final_report", "No report generated"))
        return

    # IF LLM's NOT CONFIGURED
    if not cfg.google_api_key and not cfg.groq_api_key:
        logger.error("No LLM API key set (GOOGLE_API_KEY or GROQ_API_KEY)")
        return

    # INSTANTIATE AGENT
    agent = build_agent()

    # INSTANTIATE CROO
    croo_cfg = CrooConfig(
        base_url=cfg.croo_api_url, 
        ws_url=cfg.croo_ws_url
    )
    client = AgentClient(croo_cfg, cfg.croo_sdk_key)

    # Accepts Negotiations and Delivers
    handler = ResearchProviderHandler(client, agent)
    await handler.start()

    logger.info("🦉Lookas research agent running. Press Ctrl+C to stop.")

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    await stop.wait()

    await handler.stop()
    await client.close()
    logger.info("🦉Lookas stopped.")


if __name__ == "__main__":
    asyncio.run(main())
