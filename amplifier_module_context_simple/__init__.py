"""
Simple context manager module.
Basic message list with token counting and compaction.
"""

import logging
import subprocess
import sys
from pathlib import Path
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
            - max_tokens: Maximum context size (default: 200,000)
            - compact_threshold: Compaction threshold (default: 0.92)
            - inject_git_context: Enable dynamic git context injection (default: False)
            - git_include_status: Include working directory status (default: True)
            - git_include_commits: Number of recent commits (default: 5)
            - git_include_branch: Include current branch (default: True)
            - git_include_main_branch: Detect main branch (default: True)

    Returns:
        Optional cleanup function
    """
    config = config or {}
    context = SimpleContextManager(
        max_tokens=config.get("max_tokens", 200_000),
        compact_threshold=config.get("compact_threshold", 0.92),
        inject_git_context=config.get("inject_git_context", False),
        git_include_status=config.get("git_include_status", True),
        git_include_commits=config.get("git_include_commits", 5),
        git_include_branch=config.get("git_include_branch", True),
        git_include_main_branch=config.get("git_include_main_branch", True),
    )
    await coordinator.mount("context", context)

    if config.get("inject_git_context"):
        logger.info("Mounted SimpleContextManager with git context injection enabled")
    else:
        logger.info("Mounted SimpleContextManager")
    return


class SimpleContextManager:
    """
    Basic context manager with message storage and token counting.
    """

    def __init__(
        self,
        max_tokens: int = 200_000,
        compact_threshold: float = 0.92,
        inject_git_context: bool = False,
        git_include_status: bool = True,
        git_include_commits: int = 5,
        git_include_branch: bool = True,
        git_include_main_branch: bool = True,
    ):
        """
        Initialize the context manager.

        Args:
            max_tokens: Maximum context size in tokens
            compact_threshold: Threshold for triggering compaction (0.0-1.0)
            inject_git_context: Enable dynamic git context injection
            git_include_status: Include working directory status
            git_include_commits: Number of recent commits to show
            git_include_branch: Include current branch name
            git_include_main_branch: Detect and show main branch
        """
        self.messages: list[dict[str, Any]] = []
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self._token_count = 0

        # Git context injection settings
        self.inject_git_context = inject_git_context
        self.git_include_status = git_include_status
        self.git_include_commits = git_include_commits
        self.git_include_branch = git_include_branch
        self.git_include_main_branch = git_include_main_branch

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
        """Get all messages in the context, with optional git context injection."""
        messages = self.messages.copy()

        # Inject git context if enabled
        if self.inject_git_context:
            git_context = self._gather_git_context()
            if git_context:
                git_msg = {"role": "system", "content": git_context}

                # Insert after existing system messages but before conversation
                insert_idx = 0
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        insert_idx = i + 1
                    else:
                        break

                messages.insert(insert_idx, git_msg)
                logger.debug(f"Injected git context at position {insert_idx}")

        return messages

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

            # If keeping assistant with tool_calls, MUST keep ALL following tool result messages
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                num_tool_calls = len(msg["tool_calls"])
                # Keep the next N tool result messages (where N = number of tool_calls)
                # Verify they're actually tool messages to be safe
                tool_results_kept = 0
                offset = 1
                while tool_results_kept < num_tool_calls and i + offset < len(
                    self.messages
                ):
                    next_msg = self.messages[i + offset]
                    if next_msg.get("role") == "tool":
                        expanded.add(i + offset)
                        tool_results_kept += 1
                    else:
                        # Non-tool message found before all results collected - possible corruption
                        logger.warning(
                            f"Message {i} has {num_tool_calls} tool_calls but only {tool_results_kept} "
                            f"tool results found before non-tool message at {i + offset}"
                        )
                        break
                    offset += 1

                logger.debug(
                    f"Preserving tool group: message {i} (assistant with {num_tool_calls} tool_calls) "
                    f"+ next {tool_results_kept} tool result messages"
                )

            # If keeping tool message, MUST keep previous assistant
            elif msg.get("role") == "tool" and i > 0:
                prev_msg = self.messages[i - 1]
                if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                    expanded.add(i - 1)
                    logger.debug(
                        f"Preserving tool pair: message {i - 1} (tool_use) + {i} (tool_result)"
                    )

        # Step 3: Build ordered compacted list
        compacted = [self.messages[i] for i in sorted(expanded)]

        # Step 4: Deduplicate (but never deduplicate tool pairs)
        seen = set()
        final = []

        for msg in compacted:
            # Never deduplicate tool-related messages (each is unique by ID)
            if (
                msg.get("role") == "tool"
                or msg.get("role") == "assistant"
                and msg.get("tool_calls")
            ):
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
            f"Compacted {old_count} → {len(final)} messages "
            f"({tool_use_count} tool_use, {tool_result_count} tool_result pairs preserved)"
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

    def _gather_git_context(self) -> str | None:
        """Gather current git repository context."""
        try:
            # Check if in git repo
            result = self._run_git(["rev-parse", "--git-dir"])
            if result is None:
                return None

            parts = [
                "*Git Status*: This is the git status at the start of the conversation. Note that this status is a snapshot in time, and will not update during the conversation.\n\n"
            ]

            # Current branch
            if self.git_include_branch:
                branch = self._run_git(["branch", "--show-current"])
                if branch:
                    parts.append(f"**Current branch:** `{branch}`\n\n")

            # Working directory status
            if self.git_include_status:
                status = self._run_git(["status", "--short"])
                if status:
                    lines = [line for line in status.split("\n") if line.strip()]
                    modified = [line for line in lines if line.startswith(" M")]
                    added = [
                        line
                        for line in lines
                        if line.startswith("A ") or line.startswith("??")
                    ]
                    deleted = [line for line in lines if line.startswith(" D")]

                    parts.append("**Working directory:**\n")
                    if modified:
                        parts.append(f"  - {len(modified)} file(s) modified\n")
                    if added:
                        parts.append(f"  - {len(added)} file(s) added/untracked\n")
                    if deleted:
                        parts.append(f"  - {len(deleted)} file(s) deleted\n")
                    parts.append("\n")
                else:
                    parts.append("**Working directory:** Clean (no changes)\n\n")

            # Recent commits
            if self.git_include_commits and self.git_include_commits > 0:
                log = self._run_git(
                    ["log", "--oneline", f"-{self.git_include_commits}"]
                )
                if log:
                    parts.append(f"**Recent commits:**\n```\n{log}\n```\n\n")

            # Main branch detection
            if self.git_include_main_branch:
                for main_branch in ["main", "master"]:
                    result = self._run_git(["rev-parse", "--verify", main_branch])
                    if result is not None:
                        parts.append(
                            f"**Main branch:** `{main_branch}` (use for PRs)\n\n"
                        )
                        break

            parts.append("---\n")

            return "".join(parts) if len(parts) > 2 else None

        except Exception as e:
            logger.warning(f"Failed to gather git context: {e}")
            return None

    def _run_git(self, args: list[str], timeout: float = 1.0) -> str | None:
        """Run a git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd(),
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return None
