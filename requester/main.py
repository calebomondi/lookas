"""
PYRMYD2 Requester — End-to-end test against the live provider on Croo.

Usage:
    export CROO_SDK_KEY="croo_sk_...requester_key..."
    export CROO_TARGET_SERVICE_ID="b27dac91-f677-4929-84ea-7810169583f0"
    python main.py
"""

import asyncio
import json
import logging
import os
from dotenv import load_dotenv

from croo import (
    AgentClient,
    Config,
    EventType,
    DeliverableType,
    NegotiateOrderRequest,
    Event,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("pyrmyd2.requester")

REQUIREMENTS = {
    "topic": "How quantum computing threatens blockchain cryptography: assessing the risk to ECDSA/Schnorr signatures and SHA-256 hashing, and the state of post-quantum migration in major chains",
    "max_analysts": 3,
    "word_count": 800,
}


async def main() -> None:
    client = AgentClient(
        Config(
            base_url=os.environ.get("CROO_API_URL", "https://api.croo.network"),
            ws_url=os.environ.get("CROO_WS_URL", "wss://api.croo.network/ws"),
        ),
        os.environ["CROO_REQUESTER_SDK_KEY"],
    )

    stream = await client.connect_websocket()
    done = asyncio.Event()

    def on_order_created(e: Event) -> None:
        async def _handle() -> None:
            logger.info("Order %s created, paying...", e.order_id)
            try:
                result = await client.pay_order(e.order_id)
                logger.info("Payment tx: %s", result.tx_hash)
            except Exception as err:
                logger.error("Pay error: %s", err)
        asyncio.create_task(_handle())

    stream.on(EventType.ORDER_CREATED, on_order_created)

    def on_order_completed(e: Event) -> None:
        async def _handle() -> None:
            logger.info("Order %s completed!", e.order_id)
            try:
                delivery = await client.get_delivery(e.order_id)
                if delivery.deliverable_type == DeliverableType.TEXT:
                    text = delivery.deliverable_text or ""
                    logger.info("Report length: %d chars", len(text))
                    logger.info("--- Report Preview (first 500 chars) ---")
                    logger.info("%s", text[:500])
                elif delivery.deliverable_type == DeliverableType.SCHEMA:
                    logger.info("Delivery schema: %s", delivery.deliverable_schema)
            except Exception as err:
                logger.error("Get delivery error: %s", err)
            done.set()
        asyncio.create_task(_handle())

    stream.on(EventType.ORDER_COMPLETED, on_order_completed)

    service_id = os.environ["CROO_TARGET_SERVICE_ID"]
    logger.info("Negotiating order with service %s ...", service_id)
    neg = await client.negotiate_order(
        NegotiateOrderRequest(
            service_id=service_id,
            requirements=json.dumps(REQUIREMENTS),
        )
    )
    logger.info("Negotiation started: %s", neg.negotiation_id)

    logger.info("Waiting for order to complete (timeout 10 min)...")
    try:
        await asyncio.wait_for(done.wait(), timeout=600)
        logger.info("Test completed successfully!")
    except asyncio.TimeoutError:
        logger.error("Timed out waiting for order completion")

    await stream.close()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
