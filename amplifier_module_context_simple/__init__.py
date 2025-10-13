"""
Simple context manager module.
Basic message list with token counting and compaction.
"""

import logging
from typing import Any
from typing import Optional

from amplifier_core import ModuleCoordinator

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the simple context manager.

    Args:
        coordinator: Module coordinator
        config: Optional configuration

    Returns:
        Optional cleanup function
    """
    config = config or {}
    context = SimpleContextManager(
        max_tokens=config.get("max_tokens", 200_000), compact_threshold=config.get("compact_threshold", 0.92)
    )
    await coordinator.mount("context", context)
    logger.info("Mounted SimpleContextManager")
    return


class SimpleContextManager:
    """
    Basic context manager with message storage and token counting.
    """

    def __init__(self, max_tokens: int = 200_000, compact_threshold: float = 0.92):
        """
        Initialize the context manager.

        Args:
            max_tokens: Maximum context size in tokens
            compact_threshold: Threshold for triggering compaction (0.0-1.0)
        """
        self.messages: list[dict[str, Any]] = []
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self._token_count = 0

    async def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to the context."""
        self.messages.append(message)
        # Simple token estimation (4 chars per token)
        self._token_count += len(str(message)) // 4
        logger.debug(f"Added message: {message.get('role', 'unknown')} - {len(self.messages)} total messages")

    async def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages in the context."""
        return self.messages.copy()

    async def should_compact(self) -> bool:
        """Check if context should be compacted."""
        usage = self._token_count / self.max_tokens
        should = usage >= self.compact_threshold
        if should:
            logger.info(f"Context at {usage:.1%} capacity, compaction recommended")
        return should

    async def compact(self) -> None:
        """Compact the context to reduce size."""
        logger.info(f"Compacting context with {len(self.messages)} messages")

        # Keep system messages and last 10 messages
        system_messages = [m for m in self.messages if m.get("role") == "system"]
        recent_messages = self.messages[-10:]

        # Combine unique messages
        seen = set()
        compacted = []

        for msg in system_messages + recent_messages:
            msg_key = (msg.get("role"), msg.get("content", "")[:100])
            if msg_key not in seen:
                seen.add(msg_key)
                compacted.append(msg)

        self.messages = compacted
        self._recalculate_tokens()
        logger.info(f"Compacted to {len(self.messages)} messages")

    async def set_messages(self, messages: list[dict[str, Any]]) -> None:
        """Set messages from a saved transcript (for session resume)."""
        self.messages = messages.copy()
        self._recalculate_tokens()
        logger.info(f"Restored {len(messages)} messages to context")

    async def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        self._token_count = 0
        logger.info("Context cleared")

    def _recalculate_tokens(self):
        """Recalculate token count after compaction."""
        self._token_count = sum(len(str(msg)) // 4 for msg in self.messages)
