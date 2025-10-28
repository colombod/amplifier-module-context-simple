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
        """Add a message to the context.

        Raises:
            RuntimeError: If adding message would exceed max_tokens after compaction
        """
        # Estimate tokens for this message
        message_tokens = len(str(message)) // 4

        # Check if adding this message would exceed threshold
        projected_total = self._token_count + message_tokens
        usage = projected_total / self.max_tokens

        if usage >= self.compact_threshold:
            logger.info(
                f"Projected usage {usage:.1%} >= threshold {self.compact_threshold:.1%}, "
                f"compacting before adding message"
            )
            await self.compact()

            # Re-check after compaction
            projected_total = self._token_count + message_tokens

            # If still would exceed max_tokens, reject
            if projected_total > self.max_tokens:
                error_msg = (
                    f"Cannot add message: would exceed context limit "
                    f"({projected_total:,} tokens > {self.max_tokens:,} max). "
                    f"Current context: {self._token_count:,} tokens, "
                    f"message: {message_tokens:,} tokens. "
                    f"Suggestions: reduce memory files, use shorter messages, "
                    f"or increase max_tokens in context config."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        # Add message if we have room
        self.messages.append(message)
        self._token_count += message_tokens
        logger.debug(
            f"Added message: {message.get('role', 'unknown')} - "
            f"{len(self.messages)} total messages, {self._token_count:,} tokens "
            f"({self._token_count / self.max_tokens:.1%})"
        )

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
        """Compact the context while preserving tool_use/tool_result pairs as atomic units.

        Anthropic API requires that tool_use blocks in message N have matching tool_result
        blocks in message N+1. These pairs are treated as atomic units during compaction:
        - If keeping a message with tool_calls, keep the next message (tool_result)
        - If keeping a tool message, keep the previous message (tool_use)

        This preserves conversation state integrity per IMPLEMENTATION_PHILOSOPHY.md:
        "Data integrity: Ensure data consistency and reliability"
        """
        logger.info(f"Compacting context with {len(self.messages)} messages")

        # Step 1: Determine base keep set
        keep_indices = set()

        # Keep all system messages
        for i, msg in enumerate(self.messages):
            if msg.get("role") == "system":
                keep_indices.add(i)

        # Keep last 10 messages
        for i in range(max(0, len(self.messages) - 10), len(self.messages)):
            keep_indices.add(i)

        # Step 2: Expand to preserve tool pairs (atomic units)
        expanded = keep_indices.copy()
        for i in keep_indices:
            msg = self.messages[i]

            # If keeping assistant with tool_calls, MUST keep next message
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                if i + 1 < len(self.messages):
                    expanded.add(i + 1)
                    logger.debug(f"Preserving tool pair: message {i} (tool_use) + {i+1} (tool_result)")

            # If keeping tool message, MUST keep previous assistant
            elif msg.get("role") == "tool" and i > 0:
                prev_msg = self.messages[i - 1]
                if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                    expanded.add(i - 1)
                    logger.debug(f"Preserving tool pair: message {i-1} (tool_use) + {i} (tool_result)")

        # Step 3: Build ordered compacted list
        compacted = [self.messages[i] for i in sorted(expanded)]

        # Step 4: Deduplicate (but never deduplicate tool pairs)
        seen = set()
        final = []

        for msg in compacted:
            # Never deduplicate tool-related messages (each is unique by ID)
            if msg.get("role") == "tool":
                final.append(msg)
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                final.append(msg)
            else:
                # Normal deduplication for non-tool messages
                msg_key = (msg.get("role"), msg.get("content", "")[:100])
                if msg_key not in seen:
                    seen.add(msg_key)
                    final.append(msg)

        old_count = len(self.messages)
        self.messages = final
        self._recalculate_tokens()

        # Log tool pair preservation
        tool_use_count = sum(1 for m in final if m.get("tool_calls"))
        tool_result_count = sum(1 for m in final if m.get("role") == "tool")
        logger.info(
            f"Compacted {old_count} → {len(final)} messages " f"({tool_use_count} tool pairs preserved)"
        )

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
