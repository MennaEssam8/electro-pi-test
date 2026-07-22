import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import (
    groq,
    deepgram,
    cartesia,
)

load_dotenv()

logger = logging.getLogger("food-delivery-agent")

# Mocked orders "database" -- stand-in for a real orders microservice/API,
# as permitted by the task ("returning a mocked lookup result").
MOCK_ORDERS = {
    "12345": {"status": "out for delivery", "eta_minutes": 12},
    "67890": {"status": "preparing in the kitchen", "eta_minutes": 35},
    "11111": {"status": "delivered", "eta_minutes": 0},
}


class SupportAgent(Agent):
    """Persona: support assistant for a food delivery app."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly, concise support assistant for a food "
                "delivery app called QuickBite. Help users check their order "
                "status and estimated delivery time. Keep responses short "
                "and conversational, since they may be spoken aloud. "
                "Always ask for the order ID if the user hasn't given one. "
                "If a lookup fails, apologize and ask the user to double-check "
                "the order ID rather than guessing an answer."
            ),
        )

    @function_tool
    async def get_order_status(self, context: RunContext, order_id: str) -> str:
        """Look up the current status of a food delivery order.

        Args:
            order_id: The customer's order ID, e.g. "12345".
        """
        # logger.info(f"[tool call] get_order_status(order_id={order_id!r})")
        logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)
        order = MOCK_ORDERS.get(order_id)
        if order is None:
            return f"No order found with ID {order_id}. Please double-check the order number."
        return f"Order {order_id} is currently: {order['status']}."

    @function_tool
    async def get_delivery_estimate(self, context: RunContext, order_id: str) -> str:
        """Get the estimated delivery time remaining for an order, in minutes.

        Args:
            order_id: The customer's order ID, e.g. "12345".
        """
        logger.info(f"[tool call] get_delivery_estimate(order_id={order_id!r})")
        order = MOCK_ORDERS.get(order_id)
        if order is None:
            return f"No order found with ID {order_id}. Please double-check the order number."
        if order["eta_minutes"] == 0:
            return f"Order {order_id} has already been delivered."
        return f"Order {order_id} should arrive in about {order['eta_minutes']} minutes."


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
             stt=deepgram.STT(),

             llm=groq.LLM(
                model="llama-3.3-70b-versatile"
            ),

              tts=cartesia.TTS(),
        )
    

    await session.start(
        room=ctx.room,
        agent=SupportAgent(),
    )

    await session.generate_reply(
        instructions=(
            "Greet the user warmly as QuickBite support and ask how you can "
            "help with their order."
        )
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))