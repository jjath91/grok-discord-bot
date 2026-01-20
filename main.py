"""
Grok Discord Bot - xAI API Integration
A Discord bot powered by Grok with web search and current datetime tools.
"""
import discord
from discord.ext import commands
import aiohttp
from aiohttp import ClientTimeout
import json
from collections import defaultdict, deque
import os
import logging
import asyncio
from typing import Optional, Dict, List
from datetime import datetime
import pytz

# Try to import ddgs for web search
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
        logging.warning(
            "Using deprecated duckduckgo_search. Consider upgrading to ddgs.")
    except ImportError:
        DDGS_AVAILABLE = False
        logging.warning("ddgs not available. Web search will be disabled.")

# --- CONFIGURATION ---
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
GROK_API_KEY = os.environ.get('GROK_API_KEY')
GROK_API_URL = 'https://api.x.ai/v1/chat/completions'
GROK_MODEL = 'grok-4'  # Latest Grok-4 model as of late 2025

SYSTEM_PROMPT = """You are Grok, a helpful and knowledgeable AI assistant in a Discord server.

Your communication style:
- Be helpful, informative, and conversational
- Keep responses concise (2-4 sentences when possible)
- Use a friendly but professional tone
- You can be witty and have personality, but avoid crude or juvenile humor
- When users ask unclear questions, ask for clarification politely
- Focus on actually answering questions rather than making jokes about them
- If you don't understand something, say so directly without being condescending

Your capabilities:
- You can search the web for current information
- You can provide date/time in different timezones
- You remember context from the conversation
- You're knowledgeable about many topics

Be genuinely helpful while maintaining an engaging personality."""

MAX_HISTORY = 10
MAX_REPLY_TOKENS = 800  # Increased for more complete responses
MAX_PAYLOAD_CHARS = 8000
MAX_TOOL_ITERATIONS = 5

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger('GrokBot')

# --- TOOL DEFINITIONS (OpenAI/Grok function calling format) ---
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_current_datetime",
        "description":
        "Get the current date and time in a specified timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type":
                    "string",
                    "description":
                    "IANA timezone name (e.g., 'America/New_York', 'Europe/London'). Defaults to UTC."
                }
            }
        }
    }
}, {
    "type": "function",
    "function": {
        "name": "web_search",
        "description":
        "Search the web for real-time information like news, weather, sports scores, or events.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                }
            },
            "required": ["query"]
        }
    }
}]


class GrokBot(commands.Bot):

    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        self.session: Optional[aiohttp.ClientSession] = None
        self.conversation_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY))
        self.processed_messages: deque = deque(maxlen=200)

    async def setup_hook(self) -> None:
        self.session = aiohttp.ClientSession(timeout=ClientTimeout(total=180))
        logger.info("Aiohttp session created for Grok API calls")
        if not DDGS_AVAILABLE:
            logger.warning("Web search disabled - ddgs not installed")

    async def on_close(self) -> None:
        if self.session:
            await self.session.close()

    async def get_current_datetime(self, timezone_str: str = "UTC") -> str:
        try:
            tz = pytz.timezone(timezone_str) if timezone_str else pytz.UTC
        except:
            tz = pytz.UTC
            timezone_str = "UTC"
        now = datetime.now(tz)
        return (f"**Current Date & Time ({timezone_str})**\n"
                f"Date: {now.strftime('%A, %B %d, %Y')}\n"
                f"Time: {now.strftime('%I:%M:%S %p')}\n"
                f"ISO: {now.isoformat()}")

    async def execute_web_search(self, query: str) -> str:
        if not DDGS_AVAILABLE:
            return "Web search is currently unavailable."
        if not query.strip():
            return "Error: Empty search query."
        try:
            loop = asyncio.get_event_loop()

            def search():
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=5))

            results = await asyncio.wait_for(loop.run_in_executor(
                None, search),
                                             timeout=12.0)
            if not results:
                return "No results found."
            formatted = "\n\n".join(
                f"**{r.get('title', 'No title')}**\n{r.get('body', 'No snippet')[:300]}...\n→ {r.get('href')}"
                for r in results[:3])
            return formatted
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Search failed: {str(e)}"

    async def execute_tool(self, tool_call: dict) -> Dict:
        """
        Execute a tool call. Handles both old object-style and new dict-style responses.
        """
        # Safely extract name, arguments, and id regardless of format
        if isinstance(tool_call, dict):
            # New style (grok-4-1-fast): pure dict
            function = tool_call.get("function", {})
            name = function.get("name")
            arguments = function.get("arguments", "{}")
            tool_call_id = tool_call.get("id")
        else:
            # Old style (unlikely now, but safe fallback)
            name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool_call_id = tool_call.id

        if not name:
            return {
                "role": "tool",
                "content": "Error: Invalid tool call format.",
                "tool_call_id": tool_call_id or "unknown"
            }

        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            args = {}

        if name == "get_current_datetime":
            result = await self.get_current_datetime(args.get("timezone"))
        elif name == "web_search":
            result = await self.execute_web_search(args.get("query", ""))
        else:
            result = f"Unknown tool: {name}"

        return {
            "role": "tool",
            "content": result,
            "tool_call_id": tool_call_id or "unknown"
        }

    def build_messages(self, user_input: str, channel_id: int) -> List[Dict]:
        """Build message array with proper history including tool calls."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        history = self.conversation_history[channel_id]
        total_chars = len(SYSTEM_PROMPT) + len(user_input)

        # Add history messages, respecting character limit
        for msg in list(history):
            # Calculate size of this message (handle all message types)
            msg_size = self._estimate_message_size(msg)
            if total_chars + msg_size > MAX_PAYLOAD_CHARS:
                logger.warning(f"Payload limit reached, truncating history")
                break
            messages.append(msg)
            total_chars += msg_size

        messages.append({"role": "user", "content": user_input})
        return messages

    def _estimate_message_size(self, msg: Dict) -> int:
        """Estimate the character size of a message including tool calls."""
        size = 0
        # Regular content
        if "content" in msg and msg["content"]:
            size += len(str(msg["content"]))
        # Tool calls in assistant messages
        if "tool_calls" in msg:
            size += len(json.dumps(msg["tool_calls"]))
        # Tool call id in tool response messages
        if "tool_call_id" in msg:
            size += len(msg["tool_call_id"])
        return size


bot = GrokBot()


@bot.event
async def on_ready():
    if bot.user:
        logger.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
        logger.info(f'Connected to {len(bot.guilds)} guild(s)')
        logger.info(
            f'Web search: {"Enabled" if DDGS_AVAILABLE else "Disabled"}')


@bot.event
async def on_message(message: discord.Message):
    if not bot.user or message.author == bot.user:
        return

    # Debug command
    if message.content.strip().lower() == '!history':
        history = bot.conversation_history[message.channel.id]
        count = len(history)

        # Count different message types
        user_msgs = sum(1 for m in history if m.get("role") == "user")
        assistant_msgs = sum(1 for m in history if m.get("role") == "assistant")
        tool_msgs = sum(1 for m in history if m.get("role") == "tool")

        await message.channel.send(
            f"📊 **History Stats**\n"
            f"Total messages: {count}/{MAX_HISTORY}\n"
            f"User: {user_msgs} | Assistant: {assistant_msgs} | Tool: {tool_msgs}"
        )
        return

    if not bot.user.mentioned_in(message):
        return

    if message.id in bot.processed_messages:
        return
    bot.processed_messages.append(message.id)

    user_input = message.content
    for mention in [f'<@!{bot.user.id}>', f'<@{bot.user.id}>']:
        user_input = user_input.replace(mention, '').strip()
    if not user_input:
        await message.channel.send("You mentioned me but didn't say anything. How can I help?"
                                   )
        return

    logger.info(f"Processing: {message.author} -> {user_input[:100]}")

    messages = bot.build_messages(user_input, message.channel.id)

    async with message.channel.typing():
        try:
            headers = {
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }
            data = {
                "model": GROK_MODEL,
                "messages": messages,
                "max_tokens": MAX_REPLY_TOKENS,
                "temperature": 0.5,  # Balanced for natural but focused responses
                "tools": TOOLS if DDGS_AVAILABLE else
                [TOOLS[0]],  # Only datetime if no search
                "tool_choice": "auto"
            }

            iteration = 0
            while iteration < MAX_TOOL_ITERATIONS:
                iteration += 1
                async with bot.session.post(GROK_API_URL,
                                            headers=headers,
                                            json=data) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            f"Grok API error {resp.status}: {error_text}")
                        await message.channel.send(
                            "Grok's having a meltdown right now. Try again later."
                        )
                        return
                    result = await resp.json()

                choice = result["choices"][0]
                delta = choice["message"]
                if delta.get("tool_calls"):
                    # Append assistant message with tool calls
                    messages.append(delta)
                    # Execute all tool calls
                    for tool_call in delta["tool_calls"]:
                        tool_response = await bot.execute_tool(tool_call)
                        messages.append(tool_response)
                    data["messages"] = messages
                    continue  # Loop again with tool results

                # Final response - save this message to the messages array too
                reply = delta.get("content", "").strip()
                messages.append(delta)  # Save final assistant response
                break
            else:
                reply = "I had trouble processing that request. Could you rephrase it more simply?"

            if reply:
                if len(reply) > 2000:
                    reply = reply[:1997] + "..."
                await message.channel.send(reply)

                # Update history with COMPLETE conversation including tool calls
                history = bot.conversation_history[message.channel.id]

                # Save the user's input
                history.append({"role": "user", "content": user_input})

                # Save ALL assistant messages (including those with tool calls) and tool responses
                # We need to reconstruct what happened during the tool calling loop
                # Starting from the initial messages we built, find everything after the user input
                for msg in messages[messages.index({"role": "user", "content": user_input}) + 1:]:
                    # Save assistant messages (with or without tool calls) and tool messages
                    if msg["role"] in ["assistant", "tool"]:
                        history.append(msg)

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            await message.channel.send(
                "Sorry, I encountered an unexpected error. Please try again.")


# --- KEEP-ALIVE WEB SERVER ---
def setup_keep_alive():
    try:
        from flask import Flask
        import threading
        app = Flask(__name__)

        @app.route('/')
        def home():
            return f"Grok Bot Online<br>Model: {GROK_MODEL}<br>Web Search: {'Enabled' if DDGS_AVAILABLE else 'Disabled'}"

        def run():
            app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

        threading.Thread(target=run, daemon=True).start()
        logger.info("Keep-alive server started")
    except ImportError:
        pass


if __name__ == '__main__':
    if not DISCORD_TOKEN:
        logger.critical("Missing DISCORD_TOKEN")
        exit(1)
    if not GROK_API_KEY:
        logger.critical("Missing GROK_API_KEY")
        exit(1)
    setup_keep_alive()
    bot.run(DISCORD_TOKEN, log_handler=None)
