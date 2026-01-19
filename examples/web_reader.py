#!/usr/bin/env python3
"""Example agent that can download and analyze web content.

Demonstrates using playwright-mcp servers to give an agent the ability to:
- Download web pages (JavaScript-rendered or static)
- Download files (PDFs, JSON, images)
- Run bash commands to analyze downloaded content

Requires:
    - ANTHROPIC_API_KEY in environment
    - playwright-mcp installed at ~/code/playwright-mcp (run `make install` there first)

Usage:
    make web-reader
"""

import asyncio
import os
from pathlib import Path

import immagent

# Path to playwright-mcp repo - adjust if installed elsewhere
PLAYWRIGHT_MCP = Path.home() / "code" / "playwright-mcp"
SANDBOX_DIR = "/tmp/agent-sandbox"


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY")
        return

    if not PLAYWRIGHT_MCP.exists():
        print(f"playwright-mcp not found at {PLAYWRIGHT_MCP}")
        print("Clone it and run 'make install' first")
        return

    # Create sandbox directory
    os.makedirs(SANDBOX_DIR, exist_ok=True)

    async with immagent.MCPManager() as mcp:
        # Connect to URL downloader (Playwright + httpx)
        await mcp.connect(
            "url-downloader",
            "uv",
            ["run", "url-downloader"],
            env={
                "SANDBOX_DIR": SANDBOX_DIR,
                "PLAYWRIGHT_BROWSERS_PATH": str(PLAYWRIGHT_MCP / ".browsers"),
            },
            cwd=str(PLAYWRIGHT_MCP),
        )

        # Connect to bash executor (sandboxed commands)
        await mcp.connect(
            "bash-executor",
            "uv",
            ["run", "bash-executor"],
            env={"SANDBOX_DIR": SANDBOX_DIR},
            cwd=str(PLAYWRIGHT_MCP),
        )

        # Show available tools
        tools = mcp.get_all_tools()
        print(f"Available tools: {[t['function']['name'] for t in tools]}\n")

        # Create a simple agent (no database needed)
        agent = immagent.SimpleAgent(
            name="WebReader",
            system_prompt=(
                "You are a helpful assistant that can download and analyze web content. "
                "You have tools to download web pages and files, and bash tools to analyze them. "
                "When analyzing content, use tools like grep, wc, head, and cat."
            ),
            model="anthropic/claude-3-5-haiku-20241022",
        )

        # Ask the agent to download and analyze a page
        print("Asking agent to download and analyze example.com...\n")
        agent = await agent.advance(
            "Download https://example.com and tell me how many words are on the page. "
            "Also show me the first 5 lines of the HTML.",
            mcp=mcp,
        )

        # Show the conversation
        print("=" * 60)
        print("Conversation:")
        print("=" * 60)
        for msg in agent.messages():
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[Tool call: {tc.name}]")
                    print(f"  Args: {tc.arguments}")
            elif msg.role == "tool":
                content = msg.content
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"\n[Tool result]\n  {content}")
            elif msg.role == "assistant":
                print(f"\nAssistant: {msg.content}")
            elif msg.role == "user":
                print(f"\nUser: {msg.content}")

        print("\n" + "=" * 60)
        print(f"Files in sandbox: {os.listdir(SANDBOX_DIR)}")


if __name__ == "__main__":
    asyncio.run(main())
