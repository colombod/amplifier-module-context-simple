"""
Simple context manager module.

Implements an in-memory context manager with EPHEMERAL compaction:
  • Messages stored in memory (self.messages is the source of truth)
  • Compaction NEVER modifies self.messages
  • get_messages_for_request() returns compacted VIEW (new list)
  • get_messages() returns FULL history (for transcripts/session persistence)

This design ensures conversation history is never lost, even during compaction.
For persistent storage across sessions, use context-persistent instead.
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
    In-memory context manager with EPHEMERAL compaction.

    Key Principle: self.messages is the source of truth and is NEVER modified
    by compaction. Compaction only returns a compacted VIEW for the current
    LLM request.

    Owns memory policy: orchestrators ask for messages via get_messages_for_request(),
    and this context manager decides how to fit them within limits. Compaction is
    handled internally and ephemerally - the original history is always preserved.

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

    async def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to the context.

        Messages are always accepted. Compaction happens ephemerally when
        get_messages_for_request() is called before LLM requests.

        Tool results MUST be added even if over threshold, otherwise
        tool_use/tool_result pairing breaks.
        """
        # Add message (no rejection - compaction happens ephemerally)
        self.messages.append(message)

        token_count = self._estimate_tokens(self.messages)
        usage = token_count / self.max_tokens
        logger.debug(
            f"Added message: {message.get('role', 'unknown')} - "
            f"{len(self.messages)} total messages, {token_count:,} tokens "
            f"({usage:.1%})"
        )

    async def get_messages_for_request(
        self,
        token_budget: int | None = None,
        provider: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get messages ready for an LLM request.

        Applies EPHEMERAL compaction if needed - returns a NEW list without
        modifying self.messages. The original history is always preserved.

        Args:
            token_budget: Optional explicit token limit (deprecated, prefer provider).
            provider: Optional provider instance for dynamic budget calculation.
                If provided, budget = context_window - max_output_tokens - safety_margin.

        Returns:
            Messages ready for LLM request, compacted if necessary.
        """
        budget = self._calculate_budget(token_budget, provider)
        token_count = self._estimate_tokens(self.messages)

        # Check if compaction needed
        if self._should_compact(token_count, budget):
            # Compact EPHEMERALLY - returns new list, self.messages unchanged
            compacted = self._compact_ephemeral(budget)
            logger.info(
                f"Ephemeral compaction: {len(self.messages)} -> {len(compacted)} messages for this request"
            )
            return compacted

        # Return a copy (never expose internal list directly)
        return list(self.messages)

    async def get_messages(self) -> list[dict[str, Any]]:
        """
        Get ALL messages (full history, never compacted) for transcripts/debugging.

        This returns the complete, unmodified history - suitable for saving
        to transcript files for session persistence.
        """
        return list(self.messages)

    async def set_messages(self, messages: list[dict[str, Any]]) -> None:
        """Set messages from a saved transcript (for session resume)."""
        self.messages = list(messages)
        logger.info(f"Restored {len(messages)} messages to context")

    async def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        logger.info("Context cleared")

    def _should_compact(self, token_count: int, budget: int) -> bool:
        """Check if context should be compacted."""
        usage = token_count / budget if budget > 0 else 0
        should = usage >= self.compact_threshold
        if should:
            logger.info(f"Context at {usage:.1%} capacity, compaction needed")
        return should

    def _compact_ephemeral(self, budget: int) -> list[dict[str, Any]]:
        """
        Compact the context EPHEMERALLY using progressive percentage-based strategy.

        This returns a NEW list - self.messages is NEVER modified.

        Two-phase approach:
        1. Truncate old tool results (cheap, preserves conversation flow)
        2. Remove oldest messages if still over target (preserves tool pairs)

        Anthropic API requires that tool_use blocks in message N have matching tool_result
        blocks in message N+1. These pairs are treated as atomic units during compaction.
        """
        target_tokens = int(budget * self.target_usage)
        old_count = len(self.messages)
        old_tokens = self._estimate_tokens(self.messages)

        logger.info(
            f"Compacting context: {len(self.messages)} messages, {old_tokens:,} tokens "
            f"(target: {target_tokens:,} tokens, {self.target_usage:.0%} of {budget:,})"
        )

        # Work on a copy - never modify self.messages
        working_messages = [dict(msg) for msg in self.messages]

        # Phase 1: Truncate old tool results (in first truncate_boundary% of history)
        truncate_boundary_idx = int(len(working_messages) * self.truncate_boundary)
        truncated_count = 0

        for i in range(truncate_boundary_idx):
            msg = working_messages[i]
            if msg.get("role") == "tool" and not msg.get("_truncated"):
                working_messages[i] = self._truncate_tool_result(msg)
                truncated_count += 1

        current_tokens = self._estimate_tokens(working_messages)
        logger.info(
            f"Phase 1: Truncated {truncated_count} tool results in first {self.truncate_boundary:.0%} of history. "
            f"Tokens: {old_tokens:,} → {current_tokens:,}"
        )

        # Phase 2: If still over target, remove oldest messages
        if current_tokens > target_tokens:
            working_messages = self._remove_oldest_until_target(
                working_messages, target_tokens
            )

        # Log final state
        final_tokens = self._estimate_tokens(working_messages)
        tool_use_count = sum(1 for m in working_messages if m.get("tool_calls"))
        tool_result_count = sum(1 for m in working_messages if m.get("role") == "tool")
        logger.info(
            f"Compaction complete: {old_count} → {len(working_messages)} messages, "
            f"{old_tokens:,} → {final_tokens:,} tokens "
            f"({tool_use_count} tool_use, {tool_result_count} tool_result pairs preserved)"
        )

        return working_messages

    def _truncate_tool_result(self, msg: dict[str, Any]) -> dict[str, Any]:
        """
        Truncate a tool result message to reduce token count.

        Returns a NEW dict - does not modify the original.
        """
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) <= self.truncate_chars:
            return msg

        original_tokens = len(content) // 4
        return {
            **msg,
            "content": f"[truncated: ~{original_tokens:,} tokens] {content[:self.truncate_chars]}...",
            "_truncated": True,
            "_original_tokens": original_tokens,
        }

    def _remove_oldest_until_target(
        self, messages: list[dict[str, Any]], target_tokens: int
    ) -> list[dict[str, Any]]:
        """
        Remove oldest non-protected messages until under target.

        Protected: system messages, recent messages, complete tool pairs.
        Returns a NEW list.
        """
        # Determine protected indices
        protected_indices = set()

        # Always protect system messages
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                protected_indices.add(i)

        # Always protect last N% of messages
        protected_boundary = int(len(messages) * (1 - self.protected_recent))
        for i in range(protected_boundary, len(messages)):
            protected_indices.add(i)

        # Build removal candidates (oldest first, excluding protected)
        removal_candidates = []
        for i, _msg in enumerate(messages):
            if i not in protected_indices:
                removal_candidates.append(i)

        # Remove messages until under target, preserving tool pairs
        indices_to_remove = set()
        current_tokens = self._estimate_tokens(messages)

        for i in removal_candidates:
            if current_tokens <= target_tokens:
                break

            msg = messages[i]

            # If this is a tool result, find its tool_use and check if the ENTIRE pair can be removed
            # CRITICAL: Only remove the pair if ALL tool_results can be removed (same as assistant path)
            if msg.get("role") == "tool":
                # Find the assistant with tool_calls
                for j in range(i - 1, -1, -1):
                    check_msg = messages[j]
                    if check_msg.get("role") == "assistant" and check_msg.get("tool_calls"):
                        # Check if assistant is protected
                        if j in protected_indices:
                            break  # Can't remove protected assistant, skip this candidate
                        
                        # Check if ALL tool_results for this assistant can be removed
                        all_tool_results_removable = True
                        tool_result_indices = []
                        for tc in check_msg.get("tool_calls", []):
                            tc_id = tc.get("id") or tc.get("tool_call_id")
                            if tc_id:
                                for k, m in enumerate(messages):
                                    if m.get("tool_call_id") == tc_id:
                                        if k in protected_indices:
                                            all_tool_results_removable = False
                                        else:
                                            tool_result_indices.append(k)
                        
                        # Only remove if ALL tool_results can be removed (preserves pairs)
                        if all_tool_results_removable:
                            indices_to_remove.add(j)
                            for k in tool_result_indices:
                                indices_to_remove.add(k)
                        # If not all removable, skip this candidate entirely (don't orphan anything)
                        break
                    if check_msg.get("role") != "tool":
                        break

            # If this is an assistant with tool_calls, also remove all its tool results
            # CRITICAL: Only remove the pair if ALL tool_results can be removed
            elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                all_tool_results_removable = True
                tool_result_indices = []
                for tc in msg.get("tool_calls", []):
                    tc_id = tc.get("id") or tc.get("tool_call_id")
                    if tc_id:
                        for k, m in enumerate(messages):
                            if m.get("tool_call_id") == tc_id:
                                if k in protected_indices:
                                    all_tool_results_removable = False
                                else:
                                    tool_result_indices.append(k)

                if all_tool_results_removable:
                    indices_to_remove.add(i)
                    for k in tool_result_indices:
                        indices_to_remove.add(k)

            # Regular message - just remove it
            else:
                indices_to_remove.add(i)

            # Update token estimate
            removed_tokens = sum(
                len(str(messages[idx])) // 4 for idx in indices_to_remove if idx not in protected_indices
            )
            current_tokens = self._estimate_tokens(messages) - removed_tokens

        # Build final message list excluding removed indices
        result = [msg for i, msg in enumerate(messages) if i not in indices_to_remove]

        logger.info(
            f"Phase 2: Removed {len(indices_to_remove)} messages. "
            f"Tokens: {self._estimate_tokens(messages):,} → {self._estimate_tokens(result):,}"
        )

        return result

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

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Rough token estimation (chars / 4)."""
        return sum(len(str(msg)) // 4 for msg in messages)
