"""
Tests for tool pair preservation during context compaction.

Verifies that tool_use and tool_result messages are kept as atomic pairs
during compaction, preventing Anthropic API errors.
"""

import pytest
from amplifier_module_context_simple import SimpleContextManager


@pytest.mark.asyncio
async def test_compact_preserves_tool_pairs_scenario_a():
    """Compaction preserves tool pair when tool_use is in keep window but tool_result is outside.

    Scenario: 11 messages total, compaction keeps last 10
    - Messages 0-8: Regular conversation (9 messages)
    - Message 9: Assistant with tool_calls (IN last 10)
    - Message 10: Tool result (NOT in last 10 without fix)

    Without fix: Keeps message 9, drops message 10 → API error
    With fix: Keeps both 9 and 10 (tool pair preserved)
    """
    context = SimpleContextManager()

    # Add 9 regular messages
    for i in range(9):
        await context.add_message({"role": "user", "content": f"message {i}"})

    # Add tool pair at messages 9-10
    await context.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "toolu_test", "tool": "bash", "arguments": {"cmd": "ls"}}],
        }
    )
    await context.add_message({"role": "tool", "tool_call_id": "toolu_test", "content": "bash output"})

    # Verify we have 11 messages
    assert len(context.messages) == 11

    # Compact (should keep last 10 = indices 1-10, but will expand to include message 9 partner)
    await context.compact()

    messages = await context.get_messages()

    # Verify tool pair preserved
    has_tool_use = any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)
    has_tool_result = any(m.get("role") == "tool" for m in messages)

    assert has_tool_use == has_tool_result, (
        f"Tool pair broken! has_tool_use={has_tool_use}, has_tool_result={has_tool_result}"
    )

    # If tool_use present, verify tool_result immediately follows
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            assert i + 1 < len(messages), f"Tool_use at message {i} but no next message"
            next_msg = messages[i + 1]
            assert next_msg.get("role") == "tool", (
                f"Tool_use at message {i} not followed by tool message (found role={next_msg.get('role')})"
            )


@pytest.mark.asyncio
async def test_compact_preserves_tool_pairs_scenario_b():
    """Compaction preserves tool pair when tool_result is in keep window but tool_use is outside.

    Scenario: 12 messages total, compaction keeps last 10
    - Messages 0-7: Regular conversation (8 messages)
    - Message 8: Assistant with tool_calls (NOT in last 10 without fix)
    - Message 9: Tool result (IN last 10)
    - Messages 10-11: More conversation (2 messages)

    Without fix: Keeps message 9, drops message 8 → API error
    With fix: Keeps both 8 and 9 (tool pair preserved)
    """
    context = SimpleContextManager()

    # Add 8 regular messages
    for i in range(8):
        await context.add_message({"role": "user", "content": f"message {i}"})

    # Add tool pair at messages 8-9
    await context.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "toolu_test2", "tool": "read", "arguments": {"path": "file.txt"}}],
        }
    )
    await context.add_message({"role": "tool", "tool_call_id": "toolu_test2", "content": "file content"})

    # Add 2 more messages
    await context.add_message({"role": "user", "content": "message 10"})
    await context.add_message({"role": "assistant", "content": "response 10"})

    # Verify we have 12 messages
    assert len(context.messages) == 12

    # Compact (should keep last 10 = indices 2-11, but will expand to include message 8)
    await context.compact()

    messages = await context.get_messages()

    # Verify tool pair preserved
    tool_use_count = sum(1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))
    tool_result_count = sum(1 for m in messages if m.get("role") == "tool")

    assert tool_use_count == tool_result_count, (
        f"Tool pair count mismatch! tool_use={tool_use_count}, tool_result={tool_result_count}"
    )

    # Verify adjacency
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            assert i + 1 < len(messages), f"Tool_use at message {i} but no next message"
            next_msg = messages[i + 1]
            assert next_msg.get("role") == "tool", f"Tool_use at message {i} not followed by tool message"


@pytest.mark.asyncio
async def test_compact_never_deduplicates_tool_messages():
    """Tool messages are never deduplicated since each has unique tool_call_id."""
    context = SimpleContextManager()

    # Add tool pair twice with same content but different IDs
    await context.add_message({"role": "user", "content": "test"})

    await context.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "toolu_1", "tool": "bash", "arguments": {"cmd": "ls"}}],
        }
    )
    await context.add_message({"role": "tool", "tool_call_id": "toolu_1", "content": "file1.txt"})

    await context.add_message({"role": "user", "content": "test again"})

    await context.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "toolu_2", "tool": "bash", "arguments": {"cmd": "ls"}}],
        }
    )
    await context.add_message({"role": "tool", "tool_call_id": "toolu_2", "content": "file1.txt"})

    # Compact
    await context.compact()

    messages = await context.get_messages()

    # Both tool pairs should be preserved (not deduplicated despite same content)
    tool_result_count = sum(1 for m in messages if m.get("role") == "tool")
    assert tool_result_count == 2, f"Tool messages were deduplicated! Expected 2, got {tool_result_count}"


@pytest.mark.asyncio
async def test_compact_with_multiple_tool_pairs():
    """Multiple tool pairs are all preserved correctly."""
    context = SimpleContextManager()

    # Add 3 tool pairs
    for i in range(3):
        await context.add_message({"role": "user", "content": f"request {i}"})
        await context.add_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"toolu_{i}", "tool": "bash", "arguments": {}}],
            }
        )
        await context.add_message({"role": "tool", "tool_call_id": f"toolu_{i}", "content": f"result {i}"})

    # Add more messages to push first pair outside window
    for i in range(10):
        await context.add_message({"role": "user", "content": f"later message {i}"})

    # Compact
    await context.compact()

    messages = await context.get_messages()

    # Verify all remaining tool pairs are complete
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Must have following tool message
            assert i + 1 < len(messages), f"Tool_use at {i} without following message"
            assert messages[i + 1].get("role") == "tool", f"Tool_use at {i} not followed by tool message"

    # Verify counts match
    tool_use_count = sum(1 for m in messages if m.get("tool_calls"))
    tool_result_count = sum(1 for m in messages if m.get("role") == "tool")
    assert tool_use_count == tool_result_count, "Tool pair counts don't match"


@pytest.mark.asyncio
async def test_compact_with_multiple_tool_calls_in_one_message():
    """Assistant with MULTIPLE tool_calls in one message preserves ALL tool results.

    This tests the critical bug fix: when an assistant makes 6 tool calls in one message,
    all 6 tool result messages must be preserved during compaction.

    Regression test for: "Message 7 has tool_use IDs without matching tool_result blocks"
    """
    context = SimpleContextManager()

    # Add conversation to fill context
    for i in range(8):
        await context.add_message({"role": "user", "content": f"request {i}"})
        await context.add_message({"role": "assistant", "content": f"response {i}"})

    # Add assistant with 6 tool calls (like web_search + 5 web_fetch)
    await context.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "toolu_1", "tool": "web_search", "arguments": {"query": "test"}},
                {"id": "toolu_2", "tool": "web_fetch", "arguments": {"url": "http://example.com/1"}},
                {"id": "toolu_3", "tool": "web_fetch", "arguments": {"url": "http://example.com/2"}},
                {"id": "toolu_4", "tool": "web_fetch", "arguments": {"url": "http://example.com/3"}},
                {"id": "toolu_5", "tool": "web_fetch", "arguments": {"url": "http://example.com/4"}},
                {"id": "toolu_6", "tool": "web_fetch", "arguments": {"url": "http://example.com/5"}},
            ],
        }
    )

    # Add 6 separate tool result messages (one for each tool call)
    await context.add_message({"role": "tool", "tool_call_id": "toolu_1", "content": "search results"})
    await context.add_message({"role": "tool", "tool_call_id": "toolu_2", "content": "page 1 content"})
    await context.add_message({"role": "tool", "tool_call_id": "toolu_3", "content": "page 2 content"})
    await context.add_message({"role": "tool", "tool_call_id": "toolu_4", "content": "page 3 content"})
    await context.add_message({"role": "tool", "tool_call_id": "toolu_5", "content": "page 4 content"})
    await context.add_message({"role": "tool", "tool_call_id": "toolu_6", "content": "page 5 content"})

    # Add more messages
    await context.add_message({"role": "user", "content": "what did you find?"})

    # Compact (should keep last 10 messages + the assistant with tool_calls + all 6 results)
    await context.compact()

    messages = await context.get_messages()

    # Find the assistant message with 6 tool_calls
    assistant_idx = None
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls") and len(msg["tool_calls"]) == 6:
            assistant_idx = i
            break

    assert assistant_idx is not None, "Assistant message with 6 tool_calls not found after compaction"

    # Verify ALL 6 tool results are preserved
    tool_result_count = 0
    for offset in range(1, 7):
        if assistant_idx + offset < len(messages):
            next_msg = messages[assistant_idx + offset]
            if next_msg.get("role") == "tool":
                tool_result_count += 1

    assert tool_result_count == 6, (
        f"Expected 6 tool results after assistant with 6 tool_calls, "
        f"but found only {tool_result_count}. This is the bug!"
    )
