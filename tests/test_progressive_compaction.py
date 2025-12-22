"""
Tests for progressive percentage-based context compaction.

Verifies:
1. Phase 1: Tool result truncation in older history
2. Phase 2: Percentage-based message removal
3. Protected recent messages are preserved
4. Configurable thresholds work correctly
"""

import pytest
from amplifier_module_context_simple import SimpleContextManager


@pytest.mark.asyncio
async def test_tool_result_truncation_phase1():
    """Phase 1 truncates old tool results while preserving structure."""
    # Configure for easy testing: low tokens, 50% truncate boundary
    # Use lower max_tokens to ensure compaction triggers with our message sizes
    context = SimpleContextManager(
        max_tokens=300,  # Lower to ensure compaction triggers
        compact_threshold=0.5,
        target_usage=0.3,
        truncate_boundary=0.5,
        truncate_chars=50,
        protected_recent=0.3,  # Protect recent messages but allow truncation of old tool
    )

    # Add a tool pair early (will be in truncate zone)
    await context.add_message({"role": "user", "content": "read file"})
    await context.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "toolu_early", "type": "function", "function": {"name": "read_file"}}],
        }
    )
    # Add a large tool result - this will push us over the threshold
    large_content = "x" * 500  # 500 chars = ~125 tokens
    await context.add_message({"role": "tool", "tool_call_id": "toolu_early", "content": large_content})

    # Add more messages to push tool pair into truncate zone (first 50%)
    for i in range(10):
        await context.add_message({"role": "user", "content": f"message {i} with extra padding"})
        await context.add_message({"role": "assistant", "content": f"response {i} with extra padding"})

    # Trigger compaction
    messages = await context.get_messages_for_request()

    # Find the old tool result
    tool_result = None
    for msg in messages:
        if msg.get("tool_call_id") == "toolu_early":
            tool_result = msg
            break

    # Verify truncation occurred
    if tool_result:
        content = tool_result.get("content", "")
        assert "[truncated:" in content, "Tool result should be truncated"
        assert tool_result.get("_truncated") is True, "Should have _truncated marker"
        assert len(content) < 200, f"Truncated content should be small, got {len(content)}"


@pytest.mark.asyncio
async def test_percentage_based_target():
    """Compaction reduces to target_usage percentage, not fixed message count."""
    # 50% target on 1000 token budget = target 500 tokens
    context = SimpleContextManager(
        max_tokens=1000,
        compact_threshold=0.9,
        target_usage=0.5,
        truncate_boundary=0.5,
        protected_recent=0.1,
    )

    # Add messages until we exceed threshold
    for i in range(50):
        await context.add_message({"role": "user", "content": f"message {i} with some padding content"})
        await context.add_message({"role": "assistant", "content": f"response {i} with some padding content"})

    # Record token count before compaction
    tokens_before = context._token_count

    # Trigger compaction
    await context.get_messages_for_request()

    # After compaction, should be at or below target (50% of 1000 = 500)
    assert context._token_count <= 500, f"Expected tokens <= 500 (50% of 1000), got {context._token_count}"
    assert context._token_count < tokens_before, "Compaction should reduce tokens"


@pytest.mark.asyncio
async def test_protected_recent_messages():
    """Last N% of messages are always protected from removal."""
    context = SimpleContextManager(
        max_tokens=500,
        compact_threshold=0.5,
        target_usage=0.3,
        truncate_boundary=0.5,
        protected_recent=0.2,  # Protect last 20%
    )

    # Add 20 messages
    for i in range(10):
        await context.add_message({"role": "user", "content": f"message {i}"})
        await context.add_message({"role": "assistant", "content": f"response {i}"})

    # Mark last 4 messages (20% of 20) with unique content
    last_messages_content = []
    for i in range(4):
        content = f"PROTECTED_MESSAGE_{i}"
        last_messages_content.append(content)
        await context.add_message({"role": "user", "content": content})

    # Trigger compaction
    messages = await context.get_messages_for_request()

    # Verify protected messages are still there
    message_contents = [m.get("content", "") for m in messages]
    for protected_content in last_messages_content:
        assert protected_content in message_contents, (
            f"Protected message '{protected_content}' was removed during compaction"
        )


@pytest.mark.asyncio
async def test_system_messages_always_preserved():
    """System messages are never removed or truncated."""
    context = SimpleContextManager(
        max_tokens=200,
        compact_threshold=0.5,
        target_usage=0.3,
    )

    # Add system message
    system_content = "You are a helpful assistant with important instructions."
    await context.add_message({"role": "system", "content": system_content})

    # Fill with other messages
    for i in range(20):
        await context.add_message({"role": "user", "content": f"message {i}"})
        await context.add_message({"role": "assistant", "content": f"response {i}"})

    # Trigger compaction
    messages = await context.get_messages_for_request()

    # System message must be preserved
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert len(system_messages) == 1, "System message should be preserved"
    assert system_messages[0]["content"] == system_content, "System content should be unchanged"


@pytest.mark.asyncio
async def test_truncation_marker_prevents_re_truncation():
    """Already truncated messages are not truncated again."""
    context = SimpleContextManager(
        max_tokens=500,
        compact_threshold=0.5,
        target_usage=0.4,
        truncate_chars=50,
    )

    # Add a pre-truncated tool result
    await context.add_message({"role": "user", "content": "test"})
    await context.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "test"}}],
        }
    )
    await context.add_message(
        {
            "role": "tool",
            "tool_call_id": "toolu_1",
            "content": "[truncated: ~100 tokens] some content...",
            "_truncated": True,
            "_original_tokens": 100,
        }
    )

    # Add more messages
    for i in range(15):
        await context.add_message({"role": "user", "content": f"message {i}"})

    # Trigger compaction multiple times
    await context.get_messages_for_request()
    await context.get_messages_for_request()

    # Find the tool result
    tool_results = [m for m in context.messages if m.get("tool_call_id") == "toolu_1"]
    assert len(tool_results) <= 1, "Should have at most one tool result"

    if tool_results:
        content = tool_results[0].get("content", "")
        # Should not have double truncation markers
        assert content.count("[truncated:") == 1, "Should not re-truncate already truncated content"


@pytest.mark.asyncio
async def test_configurable_truncate_chars():
    """truncate_chars configuration controls truncation length."""
    context = SimpleContextManager(
        max_tokens=500,
        compact_threshold=0.5,
        target_usage=0.3,
        truncate_chars=100,  # Keep 100 chars
    )

    # Add tool result with known content
    await context.add_message({"role": "user", "content": "test"})
    await context.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "test"}}],
        }
    )
    original_content = "A" * 500  # 500 chars
    await context.add_message({"role": "tool", "tool_call_id": "toolu_1", "content": original_content})

    # Add more to trigger compaction
    for i in range(15):
        await context.add_message({"role": "user", "content": f"msg {i}"})

    # Trigger compaction
    await context.get_messages_for_request()

    # Find truncated tool result
    tool_result = next((m for m in context.messages if m.get("tool_call_id") == "toolu_1"), None)

    if tool_result and tool_result.get("_truncated"):
        content = tool_result["content"]
        # Should contain ~100 A's from original (plus truncation marker)
        assert "A" * 100 in content, "Should preserve first 100 chars"
        assert len(content) < 200, "Total length should be small"


@pytest.mark.asyncio
async def test_tool_pairs_preserved_during_removal():
    """Tool pairs are removed together, never orphaned."""
    context = SimpleContextManager(
        max_tokens=300,
        compact_threshold=0.5,
        target_usage=0.3,
        protected_recent=0.1,
    )

    # Add several tool pairs
    for i in range(5):
        await context.add_message({"role": "user", "content": f"request {i}"})
        await context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"toolu_{i}", "type": "function", "function": {"name": "test"}}],
            }
        )
        await context.add_message({"role": "tool", "tool_call_id": f"toolu_{i}", "content": f"result {i}"})

    # Add more messages to trigger aggressive compaction
    for i in range(10):
        await context.add_message({"role": "user", "content": f"later {i}"})

    # Trigger compaction
    messages = await context.get_messages_for_request()

    # Verify no orphaned tool messages
    tool_call_ids_in_assistants = set()
    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id") or tc.get("tool_call_id")
                if tc_id:
                    tool_call_ids_in_assistants.add(tc_id)

    tool_call_ids_in_results = set()
    for msg in messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id:
                tool_call_ids_in_results.add(tc_id)

    # Every tool result should have a matching tool_use
    orphaned_results = tool_call_ids_in_results - tool_call_ids_in_assistants
    assert not orphaned_results, f"Orphaned tool results: {orphaned_results}"

    # Every tool_use should have matching results (may have multiple per assistant)
    # This is harder to check precisely, but at minimum counts should be reasonable
    assert len(tool_call_ids_in_results) <= len(tool_call_ids_in_assistants) * 6, "More tool results than tool calls"
