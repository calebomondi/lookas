import asyncio
import json
import logging
from croo import AgentClient, Config as CrooConfig, EventType, DeliverableType, DeliverOrderRequest
import db
from config import Config

logger = logging.getLogger("pyrmyd2.provider")
cfg = Config()


def calculate_min_price(analysts: int, words: int) -> float:
    """Calculate minimum acceptable price based on requirements."""
    return cfg.base_price + (analysts * cfg.per_analyst_price) + (words * cfg.per_word_price)


class ResearchProviderHandler:
    def __init__(self, client: AgentClient, agent):
        self.client = client
        self.agent = agent
        self.stream = None

    # Connect WebSocket
    async def start(self):
        self.stream = await self.client.connect_websocket()
        self.stream.on(EventType.NEGOTIATION_CREATED, self._on_negotiation)
        self.stream.on(EventType.ORDER_PAID, self._on_order_paid)
        self.stream.on(EventType.ORDER_COMPLETED, self._on_completed)
        logger.info("Research provider listening for events...")

    # Accept Incoming Negotiations
    def _on_negotiation(self, event):
        async def _handle():
            try:
                neg = await self.client.get_negotiation(event.negotiation_id)
                reqs = json.loads(neg.requirements) if neg.requirements else {}
                topic = reqs.get("topic", "")
                if not topic:
                    logger.warning("Negotiation %s missing topic, rejecting", event.negotiation_id)
                    await self.client.reject_negotiation(event.negotiation_id, "Missing topic in requirements")
                    return

                max_analysts = reqs.get("max_analysts", 3)
                word_count = reqs.get("word_count", 1000)
                offered_price = float(neg.fund_amount or 0)
                min_price = calculate_min_price(max_analysts, word_count)

                if offered_price < min_price:
                    reason = f"Price ${offered_price:.2f} below minimum ${min_price:.2f} for {max_analysts} analysts / {word_count} words"
                    logger.warning("Rejecting negotiation %s: %s", event.negotiation_id, reason)
                    await self.client.reject_negotiation(event.negotiation_id, reason)
                    return

                result = await self.client.accept_negotiation(event.negotiation_id)
                order_id = result.order.order_id
                logger.info(
                    "Accepted negotiation %s for topic '%s', order %s (price: $%.2f)",
                    event.negotiation_id, topic, order_id, offered_price,
                )
                db.record_order(
                    order_id=order_id,
                    negotiation_id=event.negotiation_id,
                    topic=topic,
                    word_count=word_count,
                    max_analysts=max_analysts,
                    requester_agent_id=event.requester_agent_id,
                    service_id=event.service_id,
                    price=result.order.price or "",
                )
            except Exception as e:
                logger.error("Accept negotiation failed: %s", e)
        asyncio.create_task(_handle())

    # Deliver After Payment
    def _on_order_paid(self, event):
        async def _handle():
            order_id = event.order_id
            logger.info("Order %s paid, starting research...", order_id)
            try:
                order = await self.client.get_order(order_id)
                neg = await self.client.get_negotiation(order.negotiation_id)
                reqs = json.loads(neg.requirements) if neg.requirements else {}

                topic = reqs.get("topic", "")
                word_count = reqs.get("word_count", 1000)
                max_analysts = reqs.get("max_analysts", 3)

                db.update_order(order_id, status="researching")

                thread = {"configurable": {"thread_id": order_id}}
                result = await self.agent.ainvoke({
                    "topic": topic,
                    "max_analysts": max_analysts,
                    "word_count": word_count,
                }, thread)
                report = result.get("final_report", "")
                report_length = len(report.split())

                await self.client.deliver_order(
                    order_id,
                    DeliverOrderRequest(
                        deliverable_type=DeliverableType.TEXT,
                        deliverable_text=report,
                    ),
                )
                db.update_order(order_id, status="completed", report_length=report_length)
                logger.info("Order %s delivered successfully (%d words)", order_id, report_length)
            except Exception as e:
                db.update_order(order_id, status="failed", error_message=str(e))
                logger.error("Deliver order %s failed: %s", order_id, e)
        asyncio.create_task(_handle())

    # Completed Order
    def _on_completed(self, event):
        logger.info("Order %s completed", event.order_id)

    async def stop(self):
        if self.stream:
            await self.stream.close()
