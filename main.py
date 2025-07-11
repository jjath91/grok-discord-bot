# This is my grok bot. There are many like it, but this is mine. Jage69420
import discord
from discord.ext import commands, tasks
import aiohttp
import json
from collections import defaultdict, deque
import os
import logging
import asyncio
from typing import Optional

# --- CONFIGURATION ---
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN')
GROK_API_KEY = os.environ.get('GROK_API_KEY')
GROK_API_URL = 'https://api.x.ai/v1/chat/completions'
GROK_MODEL = 'grok-4-latest'

# REVISED SYSTEM PROMPT: More direct and forceful about being concise.
SYSTEM_PROMPT = "You are Grok, a sarcastic and unfiltered AI in a Discord server. Your replies must be hilarious, non-PC, and brutally concise. NEVER write more than three sentences. Keep it short, punchy, and under 100 words. You are based, not woke."

MAX_HISTORY = 10
MAX_REPLY_TOKENS = 150 # Your desired value for reply length.
# This is a safe character limit for the *entire* API payload (system prompt + history + user input)
# to avoid exceeding the API's actual token limit. This is more reliable than token counting.
MAX_PAYLOAD_CHARS = 8000 # Your desired context size.

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- BOT SETUP (SUBCLASSING FOR BETTER STATE MANAGEMENT) ---
# This class structure is the modern, stable way to build discord.py bots.
# It properly manages state and background tasks without blocking.
class GrokBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)

        # State managed within the bot object
        self.session: Optional[aiohttp.ClientSession] = None
        self.conversation_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY))
        # A deque with a maxlen is more efficient for preventing duplicates than a set that is manually cleared.
        self.processed_messages = deque(maxlen=200)

    async def setup_hook(self) -> None:
        """This function is called once before the bot logs in."""
        # Create a single, reusable session for all API calls for efficiency.
        self.session = aiohttp.ClientSession()
        logging.info("Aiohttp session started.")

    async def on_close(self) -> None:
        """Called when the bot is shutting down to clean up the session."""
        if self.session:
            await self.session.close()
            logging.info("Aiohttp session closed.")

    async def on_ready(self):
        """Called when the bot is connected and ready."""
        if self.user is None:
            logging.error("Bot user is None, cannot start!")
            return
        logging.info(f'Logged in as {self.user} - ready to shit post my way to the top!')

bot = GrokBot()

# --- BOT EVENTS ---
@bot.event
async def on_message(message: discord.Message):
    """This event is triggered for every message in every channel the bot can see."""
    # Ensure the bot object is valid and the message is not from the bot itself
    if not bot.user or message.author == bot.user:
        return

    # Process only if the bot is mentioned
    if bot.user.mentioned_in(message):
        # --- DUPLICATE MESSAGE PREVENTION ---
        if message.id in bot.processed_messages:
            logging.warning(f"Skipping duplicate message event {message.id}")
            return
        bot.processed_messages.append(message.id)

        # Clean user input
        user_input = message.content.replace(f'<@!{bot.user.id}>', '').replace(f'<@{bot.user.id}>', '').strip()
        if not user_input:
            return

        logging.info(f"Processing mention from {message.author} in {message.channel.id}: {message.id}")

        # --- DYNAMIC PAYLOAD/HISTORY MANAGEMENT ---
        # This logic robustly builds the API payload without breaking on long messages.
        messages = []
        current_payload_chars = 0

        # Start with the system prompt and the new user message
        system_message = {'role': 'system', 'content': SYSTEM_PROMPT}
        user_message = {'role': 'user', 'content': user_input}
        messages.extend([system_message, user_message])
        current_payload_chars += len(SYSTEM_PROMPT) + len(user_input)

        # Add recent history, newest first, until the character limit is reached
        channel_id = message.channel.id
        history = bot.conversation_history[channel_id]
        for msg in reversed(history):
            msg_len = len(msg.get('content', ''))
            if current_payload_chars + msg_len > MAX_PAYLOAD_CHARS:
                logging.warning(f"Payload limit reached. Truncating history for message {message.id}.")
                break
            messages.insert(1, msg) # Insert after system prompt
            current_payload_chars += msg_len

        # Show "typing..." indicator
        async with message.channel.typing():
            headers = {
                'Authorization': f'Bearer {GROK_API_KEY}',
                'Content-Type': 'application/json'
            }
            data = {
                'model': GROK_MODEL,
                'messages': messages,
                'temperature': 0.5,
                'max_tokens': MAX_REPLY_TOKENS
            }

            try:
                # Ensure the bot's session is active and reuse it
                if not bot.session:
                    logging.error("Aiohttp session not available!")
                    await message.channel.send("Bot is not properly initialized. Session is missing.")
                    return

                # Make the API call
                async with bot.session.post(GROK_API_URL, headers=headers, json=data, timeout=180) as response:
                    response.raise_for_status()
                    response_json = await response.json()
                    grok_reply = response_json['choices'][0]['message']['content'].strip()

                    # Send reply and update history
                    await message.channel.send(grok_reply)
                    history.append({'role': 'user', 'content': user_input})
                    history.append({'role': 'assistant', 'content': grok_reply})
                    logging.info(f"Sent reply for message {message.id}")

            except aiohttp.ClientResponseError as e:
                logging.error(f"API Client Error for message {message.id}: {e.status} {e.message}")
                await message.channel.send(f"You egg! The API choked with a {e.status} error. Probably too much text.")
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logging.error(f"API Parse Error for message {message.id}: {e}")
                await message.channel.send("Grok spat out gibberish. Try again, ya landlubber.")
            except asyncio.TimeoutError:
                logging.error(f"API Timeout for message {message.id}")
                await message.channel.send("Grok took too long to think. He's probably day-dreaming. Try again.")
            except Exception as e:
                logging.error(f"Unexpected error for message {message.id}: {e}")
                await message.channel.send("Something broke, mate. Check the logs.")

# --- REPLIT KEEP-ALIVE (Optional but recommended) ---
# A simple web server to keep the bot alive on Replit.
# This is separate from the bot logic for better organization.
from flask import Flask
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return "Bot is alive!"

def run_web():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

threading.Thread(target=run_web, daemon=True).start()


# --- RUN THE BOT ---
if not DISCORD_TOKEN or not GROK_API_KEY:
    logging.critical("FATAL ERROR: DISCORD_TOKEN or GROK_API_KEY is not set in environment variables.")
else:
    # This will run the bot's async event loop
    bot.run(DISCORD_TOKEN)
