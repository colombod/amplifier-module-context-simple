"""
Simple context manager module.
Basic message list with token counting and internal compaction.
"""

# Amplifier module metadata
__amplifier_module_type__ = "context"

import logging
from typing import Any

from amplifier_core import ModuleCoordinator

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the simple context manager.

    Args:
        coordinator: Module coordinator
        config: Optional configuration
            - max_tokens: Maximum context size (default: 200,000)
            - compact_threshold: Compaction threshold (default: 0.92)

    Returns:
        Optional cleanup function
    """
    config = config or {}
    context = SimpleContextManager(
        max_tokens=config.get("max_tokens", 200_000),
        compact_threshold=config.get("compact_threshold", 0.92),
    )
    await coordinator.mount("context", context)
    logger.info("Mounted SimpleContextManager")
    return


class SimpleContextManager:
    """
    Basic context manager with message storage and token counting.

    Owns memory policy: orchestrators ask for messages via get_messages_for_request(),
    and this context manager decides how to fit them within limits. Compaction is
    handled internally - orchestrators don't know or care about compaction.
    """

    def __init__(
        self,
        max_tokens: int = 200_000,
        compact_threshold: float = 0.92,
    ):
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

        Messages are always accepted. Compaction happens internally when
        get_messages_for_request() is called before LLM requests.

        Tool results MUST be added even if over threshold, otherwise
        tool_use/tool_result pairing breaks.
        """
        # Estimate tokens for this message
        message_tokens = len(str(message)) // 4

        # Add message (no rejection - compaction happens internally)
        self.messages.append(message)
        self._token_count += message_tokens

        usage = self._token_count / self.max_tokens
        logger.debug(
            f"Added message: {message.get('role', 'unknown')} - "
            f"{len(self.messages)} total messages, {self._token_count:,} tokens "
            f"({usage:.1%})"
        )

    async def get_messages_for_request(self, token_budget: int | None = None) -> list[dict[str, Any]]:
        """
        Get messages ready for an LLM request.

        Handles compaction internally if needed. Orchestrators call this before
        every LLM request and trust the context manager to return messages that
        fit within limits.

        Args:
            token_budget: Optional token limit. If None, uses configured max.

        Returns:
            Messages ready for LLM request, compacted if necessary.
        """
        budget = token_budget or self.max_tokens

        # Check if compaction needed
        if self._should_compact(budget):
            await self._compact_internal()

        return self.messages.copy()

    async def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages (raw, uncompacted) for transcripts/debugging."""
        return self.messages.copy()

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

    def _should_compact(self, budget: int | None = None) -> bool:
        """Internal: Check if context should be compacted."""
        effective_budget = budget or self.max_tokens
        usage = self._token_count / effective_budget
        should = usage >= self.compact_threshold
        if should:
            logger.info(f"Context at {usage:.1%} capacity, compaction needed")
        return should

    async def _compact_internal(self) -> None:
        """Internal: Compact the context while preserving tool_use/tool_result pairs.

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
        # IMPORTANT: Must iterate until no new messages added, because:
        # - If tool_result in keep_indices → adds tool_use to expanded
        # - That tool_use → must add ALL its tool_results (not just those in keep_indices)
        expanded = keep_indices.copy()
        processed_indices = set()

        changed = True
        while changed:
            changed = False
            # Process indices we haven't processed yet
            to_process = expanded - processed_indices
            for i in to_process:
                processed_indices.add(i)
                msg = self.messages[i]

                # If keeping assistant with tool_calls, MUST keep ALL matching tool results
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Collect the tool_call IDs we need to find results for
                    expected_tool_ids = set()
                    for tc in msg["tool_calls"]:
                        if tc:
                            tool_id = tc.get("id") or tc.get("tool_call_id")
                            if tool_id:
                                expected_tool_ids.add(tool_id)

                    # Scan forward to find matching tool results
                    tool_results_kept = 0
                    for j in range(i + 1, len(self.messages)):
                        candidate = self.messages[j]
                        if candidate.get("role") == "tool":
                            tool_id = candidate.get("tool_call_id")
                            if tool_id in expected_tool_ids:
                                if j not in expanded:
                                    expanded.add(j)
                                    changed = True
                                expected_tool_ids.discard(tool_id)
                                tool_results_kept += 1
                                if not expected_tool_ids:
                                    break  # Found all tool results

                    if expected_tool_ids:
                        logger.warning(
                            f"Message {i} has {len(msg['tool_calls'])} tool_calls but only "
                            f"{tool_results_kept} matching tool results found (missing IDs: {expected_tool_ids})"
                        )

                    logger.debug(
                        f"Preserving tool group: message {i} (assistant with {len(msg['tool_calls'])} tool_calls) "
                        f"+ {tool_results_kept} tool result messages"
                    )

                # If keeping tool message, MUST keep the assistant with tool_calls
                elif msg.get("role") == "tool":
                    # Walk backwards to find assistant with tool_calls
                    for j in range(i - 1, -1, -1):
                        check_msg = self.messages[j]
                        if check_msg.get("role") == "assistant" and check_msg.get("tool_calls"):
                            if j not in expanded:
                                expanded.add(j)
                                changed = True
                            logger.debug(
                                f"Preserving tool group: message {j} (assistant with tool_calls) "
                                f"includes tool result at {i}"
                            )
                            break
                        if check_msg.get("role") != "tool":
                            logger.warning(
                                f"Tool result at {i} has no matching assistant with tool_calls "
                                f"(found {check_msg.get('role')} at {j} instead)"
                            )
                            break

        # Step 3: Build ordered compacted list
        compacted = [self.messages[i] for i in sorted(expanded)]

        # Step 4: Deduplicate (but never deduplicate tool pairs)
        seen = set()
        final = []

        for msg in compacted:
            # Never deduplicate tool-related messages (each is unique by ID)
            if msg.get("role") == "tool" or msg.get("role") == "assistant" and msg.get("tool_calls"):
                final.append(msg)
            else:
                # Normal deduplication for non-tool messages
                # Content may be a string or list of blocks - convert to string for hashing
                content = msg.get("content", "")
                content_str = str(content) if not isinstance(content, str) else content
                msg_key = (msg.get("role"), content_str[:100])
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
            f"Compacted {old_count} → {len(final)} messages "
            f"({tool_use_count} tool_use, {tool_result_count} tool_result pairs preserved)"
        )

    def _recalculate_tokens(self):
        """Recalculate token count after compaction."""
        self._token_count = sum(len(str(msg)) // 4 for msg in self.messages)
