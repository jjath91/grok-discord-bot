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

# Local modules
import database
import profile_generator

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

SYSTEM_PROMPT = """You are Grok, a witty and knowledgeable AI assistant in a Discord server.

Your communication style:
- Be helpful, conversational, and engaging
- Keep responses concise (2-4 sentences when possible)
- You can be edgy, sarcastic, and handle mature/inappropriate topics
- Use clever, witty humor - NOT juvenile or middle-school-level jokes
- If someone asks something crude, you can engage with it in a smart, funny way
- Avoid being preachy, condescending, or acting like a hall monitor
- When users ask unclear questions, ask for clarification without being a smart-ass about it

Your capabilities and tool usage:
- You can search the web for current information
- You can provide date/time in different timezones
- You remember context from the conversation

**CRITICAL - Use your tools appropriately:**
- Weather, news, sports scores, current events = ALWAYS use web_search
- Date/time questions = ALWAYS use get_current_datetime
- Don't guess or make up current information - search for it!

Be genuinely helpful while being entertaining and a bit irreverent."""

MAX_HISTORY = 10
MAX_REPLY_TOKENS = 400  # Reduced for faster responses
MAX_PAYLOAD_CHARS = 8000
MAX_TOOL_ITERATIONS = 3  # Reduced to prevent long chains

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
        # Changed from channel-based to user-based history tracking
        self.conversation_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY))
        self.processed_messages: deque = deque(maxlen=200)

    async def setup_hook(self) -> None:
        self.session = aiohttp.ClientSession(timeout=ClientTimeout(total=30))
        logger.info("Aiohttp session created for Grok API calls")
        if not DDGS_AVAILABLE:
            logger.warning("Web search disabled - ddgs not installed")
        # Initialize the database
        await database.init_db()
        logger.info("Database initialized")

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
                    return list(ddgs.text(query, max_results=3))

            results = await asyncio.wait_for(loop.run_in_executor(
                None, search),
                                             timeout=6.0)
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

    async def build_messages(self, user_input: str, user_id: int) -> List[Dict]:
        """Build message array with proper history including tool calls."""
        # Fetch user profile and inject into system prompt if it exists
        system_content = SYSTEM_PROMPT
        profile_data = await database.get_profile(user_id)
        if profile_data and profile_data.get("profile_text"):
            system_content += f"\n\n--- User Context ---\n{profile_data['profile_text']}"

        messages = [{"role": "system", "content": system_content}]
        history = self.conversation_history[user_id]
        total_chars = len(system_content) + len(user_input)

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

    # Process commands first (for !profile, !regenerate, !forget, etc.)
    await bot.process_commands(message)

    # Debug command - now shows YOUR personal history
    if message.content.strip().lower() == '!history':
        history = bot.conversation_history[message.author.id]
        count = len(history)

        # Count different message types
        user_msgs = sum(1 for m in history if m.get("role") == "user")
        assistant_msgs = sum(1 for m in history if m.get("role") == "assistant")
        tool_msgs = sum(1 for m in history if m.get("role") == "tool")

        await message.channel.send(
            f"📊 **Your History Stats** ({message.author.display_name})\n"
            f"Total messages: {count}/{MAX_HISTORY}\n"
            f"User: {user_msgs} | Assistant: {assistant_msgs} | Tool: {tool_msgs}"
        )
        return

    # Clear command - reset YOUR conversation history
    if message.content.strip().lower() == '!clear':
        if message.author.id in bot.conversation_history:
            bot.conversation_history[message.author.id].clear()
            await message.channel.send(f"✅ Cleared conversation history for {message.author.display_name}")
        else:
            await message.channel.send(f"You don't have any conversation history yet!")
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
    start_time = asyncio.get_event_loop().time()

    messages = await bot.build_messages(user_input, message.author.id)

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
                "temperature": 0.7,  # Higher for more personality and wit
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

                # Log response time
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.info(f"Response generated in {elapsed:.2f}s")

                await message.channel.send(reply)

                # Update history with COMPLETE conversation including tool calls
                # Now using user-specific history instead of channel-based
                history = bot.conversation_history[message.author.id]

                # Save the user's input
                history.append({"role": "user", "content": user_input})

                # Save ALL assistant messages (including those with tool calls) and tool responses
                # We need to reconstruct what happened during the tool calling loop
                # Starting from the initial messages we built, find everything after the user input
                for msg in messages[messages.index({"role": "user", "content": user_input}) + 1:]:
                    # Save assistant messages (with or without tool calls) and tool messages
                    if msg["role"] in ["assistant", "tool"]:
                        history.append(msg)

                # Persist messages to database
                user_id = message.author.id
                await database.save_message(user_id, "user", content=user_input)
                for msg in messages[messages.index({"role": "user", "content": user_input}) + 1:]:
                    if msg["role"] == "assistant":
                        await database.save_message(
                            user_id, "assistant",
                            content=msg.get("content"),
                            tool_calls=msg.get("tool_calls")
                        )
                    elif msg["role"] == "tool":
                        await database.save_message(
                            user_id, "tool",
                            content=msg.get("content"),
                            tool_call_id=msg.get("tool_call_id")
                        )

                # Check if we should regenerate the user's profile
                if await database.should_regenerate_profile(user_id):
                    logger.info(f"Triggering profile regeneration for user {user_id}")
                    asyncio.create_task(regenerate_profile_for_user(user_id, bot.session))

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            await message.channel.send(
                "Sorry, I encountered an unexpected error. Please try again.")


async def regenerate_profile_for_user(
    user_id: int,
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """Helper function to regenerate a user's profile."""
    try:
        messages = await database.get_user_messages(user_id)
        if not messages:
            return None

        profile_text = await profile_generator.generate_user_profile(
            user_id, messages, session
        )
        if profile_text:
            msg_count = await database.get_user_message_count(user_id)
            await database.save_profile(user_id, profile_text, msg_count)
            return profile_text
        return None
    except Exception as e:
        logger.error(f"Error regenerating profile for user {user_id}: {e}")
        return None


@bot.command(name='profile')
async def show_profile(ctx: commands.Context):
    """Show your current user profile."""
    profile_data = await database.get_profile(ctx.author.id)
    if profile_data and profile_data.get("profile_text"):
        msg_count = await database.get_user_message_count(ctx.author.id)
        await ctx.send(
            f"**Profile for {ctx.author.display_name}**\n"
            f"(Based on {profile_data['message_count']} messages, "
            f"updated: {profile_data['last_updated']})\n\n"
            f"{profile_data['profile_text']}"
        )
    else:
        msg_count = await database.get_user_message_count(ctx.author.id)
        if msg_count > 0:
            await ctx.send(
                f"No profile yet for {ctx.author.display_name}.\n"
                f"You have {msg_count} messages stored. "
                f"Use `!regenerate` to generate your profile."
            )
        else:
            await ctx.send(
                f"No profile yet for {ctx.author.display_name}.\n"
                f"Start chatting with me to build your profile!"
            )


@bot.command(name='regenerate')
async def regenerate_profile(ctx: commands.Context):
    """Force regenerate your user profile."""
    msg_count = await database.get_user_message_count(ctx.author.id)
    if msg_count < 5:
        await ctx.send(
            f"Not enough conversation history yet ({msg_count} messages). "
            f"Chat with me more before generating a profile!"
        )
        return

    status_msg = await ctx.send("Generating your profile...")

    profile_text = await regenerate_profile_for_user(ctx.author.id, bot.session)

    if profile_text:
        await status_msg.edit(
            content=f"**Updated Profile for {ctx.author.display_name}**\n\n{profile_text}"
        )
    else:
        await status_msg.edit(content="Failed to generate profile. Please try again later.")


@bot.command(name='forget')
async def forget_user_data(ctx: commands.Context):
    """Delete all your stored data (messages and profile)."""
    msg_count = await database.get_user_message_count(ctx.author.id)
    profile = await database.get_profile(ctx.author.id)

    if msg_count == 0 and not profile:
        await ctx.send("You don't have any stored data to delete.")
        return

    await database.delete_user_data(ctx.author.id)

    # Also clear in-memory conversation history
    if ctx.author.id in bot.conversation_history:
        bot.conversation_history[ctx.author.id].clear()

    await ctx.send(
        f"All your data has been deleted:\n"
        f"- {msg_count} messages removed\n"
        f"- Profile {'removed' if profile else 'was not present'}\n\n"
        f"Your conversation history has been cleared."
    )


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
