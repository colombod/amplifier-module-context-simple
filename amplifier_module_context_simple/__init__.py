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
            - compact_threshold: Trigger compaction at this usage (default: 0.92)
            - target_usage: Compact down to this usage (default: 0.50)
            - truncate_boundary: Truncate tool results in first N% of history (default: 0.50)
            - protected_recent: Always protect last N% of messages (default: 0.10)
            - truncate_chars: Characters to keep when truncating tool results (default: 250)

    Returns:
        Optional cleanup function
    """
    config = config or {}
    context = SimpleContextManager(
        max_tokens=config.get("max_tokens", 200_000),
        compact_threshold=config.get("compact_threshold", 0.92),
        target_usage=config.get("target_usage", 0.50),
        truncate_boundary=config.get("truncate_boundary", 0.50),
        protected_recent=config.get("protected_recent", 0.10),
        truncate_chars=config.get("truncate_chars", 250),
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

    Compaction Strategy (Progressive Percentage-Based):
    1. Trigger when usage >= compact_threshold (default 92%)
    2. Phase 1: Truncate old tool results in first truncate_boundary% of history
    3. Phase 2: Remove oldest messages until at target_usage% (default 50%)
    4. Always protect: system messages, last protected_recent% of messages, tool pairs
    """

    def __init__(
        self,
        max_tokens: int = 200_000,
        compact_threshold: float = 0.92,
        target_usage: float = 0.50,
        truncate_boundary: float = 0.50,
        protected_recent: float = 0.10,
        truncate_chars: int = 250,
    ):
        """
        Initialize the context manager.

        Args:
            max_tokens: Maximum context size in tokens
            compact_threshold: Trigger compaction at this usage ratio (0.0-1.0)
            target_usage: Compact down to this usage ratio (0.0-1.0)
            truncate_boundary: Truncate tool results in first N% of history (0.0-1.0)
            protected_recent: Always protect last N% of messages (0.0-1.0)
            truncate_chars: Characters to keep when truncating tool results
        """
        self.messages: list[dict[str, Any]] = []
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.target_usage = target_usage
        self.truncate_boundary = truncate_boundary
        self.protected_recent = protected_recent
        self.truncate_chars = truncate_chars
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

    async def get_messages_for_request(
        self,
        token_budget: int | None = None,
        provider: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get messages ready for an LLM request.

        Handles compaction internally if needed. Orchestrators call this before
        every LLM request and trust the context manager to return messages that
        fit within limits.

        Args:
            token_budget: Optional explicit token limit (deprecated, prefer provider).
            provider: Optional provider instance for dynamic budget calculation.
                If provided, budget = context_window - max_output_tokens - safety_margin.

        Returns:
            Messages ready for LLM request, compacted if necessary.
        """
        budget = self._calculate_budget(token_budget, provider)

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
        """Internal: Compact the context using progressive percentage-based strategy.

        Two-phase approach:
        1. Truncate old tool results (cheap, preserves conversation flow)
        2. Remove oldest messages if still over target (preserves tool pairs)

        Anthropic API requires that tool_use blocks in message N have matching tool_result
        blocks in message N+1. These pairs are treated as atomic units during compaction.

        This preserves conversation state integrity per IMPLEMENTATION_PHILOSOPHY.md:
        "Data integrity: Ensure data consistency and reliability"
        """
        budget = self._calculate_budget(None, None)
        target_tokens = int(budget * self.target_usage)
        old_count = len(self.messages)
        old_tokens = self._token_count

        logger.info(
            f"Compacting context: {len(self.messages)} messages, {self._token_count:,} tokens "
            f"(target: {target_tokens:,} tokens, {self.target_usage:.0%} of {budget:,})"
        )

        # Phase 1: Truncate old tool results
        truncate_boundary_idx = int(len(self.messages) * self.truncate_boundary)
        truncated_count = 0

        for i in range(truncate_boundary_idx):
            msg = self.messages[i]
            if msg.get("role") == "tool" and not msg.get("_truncated"):
                self._truncate_tool_result(msg)
                truncated_count += 1

        self._recalculate_tokens()
        logger.info(
            f"Phase 1: Truncated {truncated_count} tool results in first {self.truncate_boundary:.0%} of history. "
            f"Tokens: {old_tokens:,} → {self._token_count:,}"
        )

        # Phase 2: If still over target, remove oldest messages
        if self._token_count > target_tokens:
            # Determine protected indices
            protected_indices = set()

            # Always protect system messages
            for i, msg in enumerate(self.messages):
                if msg.get("role") == "system":
                    protected_indices.add(i)

            # Always protect last N% of messages
            protected_boundary = int(len(self.messages) * (1 - self.protected_recent))
            for i in range(protected_boundary, len(self.messages)):
                protected_indices.add(i)

            # Build removal candidates (oldest first, excluding protected)
            removal_candidates = []
            for i, msg in enumerate(self.messages):
                if i not in protected_indices:
                    removal_candidates.append(i)

            # Remove messages until under target, preserving tool pairs
            indices_to_remove = set()
            for i in removal_candidates:
                if self._token_count <= target_tokens:
                    break

                msg = self.messages[i]

                # If this is a tool result, also mark its tool_use for removal
                if msg.get("role") == "tool":
                    # Find the assistant with tool_calls
                    for j in range(i - 1, -1, -1):
                        check_msg = self.messages[j]
                        if check_msg.get("role") == "assistant" and check_msg.get("tool_calls"):
                            if j not in protected_indices:
                                indices_to_remove.add(j)
                                # Also remove ALL tool results for this tool_use
                                for tc in check_msg.get("tool_calls", []):
                                    tc_id = tc.get("id") or tc.get("tool_call_id")
                                    if tc_id:
                                        for k, m in enumerate(self.messages):
                                            if m.get("tool_call_id") == tc_id and k not in protected_indices:
                                                indices_to_remove.add(k)
                            break
                        if check_msg.get("role") != "tool":
                            break

                # If this is an assistant with tool_calls, also remove all its tool results
                elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                    indices_to_remove.add(i)
                    for tc in msg.get("tool_calls", []):
                        tc_id = tc.get("id") or tc.get("tool_call_id")
                        if tc_id:
                            for k, m in enumerate(self.messages):
                                if m.get("tool_call_id") == tc_id and k not in protected_indices:
                                    indices_to_remove.add(k)

                # Regular message - just remove it
                else:
                    indices_to_remove.add(i)

                # Estimate token reduction
                removed_tokens = sum(
                    len(str(self.messages[idx])) // 4 for idx in indices_to_remove if idx not in protected_indices
                )
                if self._token_count - removed_tokens <= target_tokens:
                    break

            # Build final message list excluding removed indices
            self.messages = [msg for i, msg in enumerate(self.messages) if i not in indices_to_remove]
            self._recalculate_tokens()

            logger.info(
                f"Phase 2: Removed {len(indices_to_remove)} messages. Tokens: {old_tokens:,} → {self._token_count:,}"
            )

        # Log final state
        tool_use_count = sum(1 for m in self.messages if m.get("tool_calls"))
        tool_result_count = sum(1 for m in self.messages if m.get("role") == "tool")
        logger.info(
            f"Compaction complete: {old_count} → {len(self.messages)} messages, "
            f"{old_tokens:,} → {self._token_count:,} tokens "
            f"({tool_use_count} tool_use, {tool_result_count} tool_result pairs preserved)"
        )

    def _truncate_tool_result(self, msg: dict[str, Any]) -> None:
        """Truncate a tool result message to reduce token count.

        Preserves tool_call_id and structure, replaces content with truncated version.
        """
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > self.truncate_chars:
            original_tokens = len(content) // 4
            truncated_content = content[: self.truncate_chars]

            # Add truncation marker with metadata
            msg["content"] = f"[truncated: ~{original_tokens:,} tokens] {truncated_content}..."
            msg["_truncated"] = True
            msg["_original_tokens"] = original_tokens

            logger.debug(
                f"Truncated tool result {msg.get('tool_call_id', 'unknown')}: "
                f"{original_tokens:,} → ~{len(msg['content']) // 4} tokens"
            )

    def _calculate_budget(self, token_budget: int | None, provider: Any | None) -> int:
        """Calculate effective token budget from provider or fallback to config.

        Priority:
        1. Explicit token_budget parameter (deprecated but supported)
        2. Provider-based calculation: context_window - max_output_tokens - safety_margin
        3. Configured max_tokens fallback
        """
        # Explicit budget takes precedence (for backward compatibility)
        if token_budget is not None:
            return token_budget

        # Try provider-based dynamic budget
        if provider is not None:
            try:
                info = provider.get_info()
                defaults = info.defaults or {}
                context_window = defaults.get("context_window")
                max_output_tokens = defaults.get("max_output_tokens")

                if context_window and max_output_tokens:
                    safety_margin = 1000  # Buffer to avoid hitting hard limits
                    budget = context_window - max_output_tokens - safety_margin
                    logger.debug(
                        f"Calculated budget from provider: {budget} "
                        f"(context={context_window}, output={max_output_tokens}, safety={safety_margin})"
                    )
                    return budget
            except Exception as e:
                logger.debug(f"Could not get budget from provider: {e}")

        # Fall back to configured max_tokens
        return self.max_tokens

    def _recalculate_tokens(self):
        """Recalculate token count after compaction."""
        self._token_count = sum(len(str(msg)) // 4 for msg in self.messages)
