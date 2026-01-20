"""
Profile generator module that uses Grok API to create user profiles
by summarizing their conversation history.
"""
import aiohttp
import json
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger('GrokBot.ProfileGenerator')

GROK_API_URL = 'https://api.x.ai/v1/chat/completions'
GROK_API_KEY = os.environ.get('GROK_API_KEY')

# Use a faster/cheaper model for profile generation
PROFILE_MODEL = 'grok-3-mini-fast'

PROFILE_GENERATION_PROMPT = """Analyze this user's conversation history and create a concise profile summary.

Extract and summarize:
1. **Communication Style**: How they write (casual/formal, humor type, verbosity)
2. **Interests**: Topics they frequently discuss or ask about
3. **Context**: Any personal details mentioned (job, location, projects, hobbies)
4. **Interaction Patterns**: How they typically engage (questions, banter, requests)
5. **Notable Points**: Anything unique or memorable about this user

Format your response EXACTLY like this (keep it under 300 tokens):

## User Profile
- **Style**: [their communication style]
- **Interests**: [topics they care about]
- **Context**: [relevant personal details]
- **Notes**: [anything notable about interactions]

Be specific but concise. If there's not enough data for a section, write "Not enough data yet."

Here is the conversation history to analyze:
"""


def format_messages_for_profile(messages: List[Dict[str, Any]]) -> str:
    """Format message history into a readable format for the LLM."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Skip tool messages and empty content
        if role == "tool" or not content:
            continue

        # Truncate very long messages
        if len(content) > 500:
            content = content[:500] + "..."

        if role == "user":
            formatted.append(f"USER: {content}")
        elif role == "assistant":
            formatted.append(f"GROK: {content}")

    return "\n".join(formatted)


async def generate_user_profile(
    user_id: int,
    messages: List[Dict[str, Any]],
    session: Optional[aiohttp.ClientSession] = None
) -> Optional[str]:
    """
    Generate a user profile by sending their message history to Grok for summarization.

    Args:
        user_id: The Discord user ID
        messages: List of message dictionaries from the database
        session: Optional aiohttp session to reuse

    Returns:
        The generated profile text, or None if generation failed
    """
    if not GROK_API_KEY:
        logger.error("GROK_API_KEY not set, cannot generate profile")
        return None

    if not messages:
        logger.info(f"No messages to generate profile for user {user_id}")
        return None

    # Format messages for the prompt
    formatted_history = format_messages_for_profile(messages)

    if not formatted_history.strip():
        logger.info(f"No valid messages to generate profile for user {user_id}")
        return None

    # Build the prompt
    full_prompt = PROFILE_GENERATION_PROMPT + "\n\n" + formatted_history

    # Prepare API request
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": PROFILE_MODEL,
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "max_tokens": 400,
        "temperature": 0.3  # Lower temperature for more consistent summaries
    }

    should_close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        should_close_session = True

    try:
        async with session.post(GROK_API_URL, headers=headers, json=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                logger.error(f"Profile generation API error {resp.status}: {error_text}")
                return None

            result = await resp.json()
            profile_text = result["choices"][0]["message"]["content"].strip()
            logger.info(f"Generated profile for user {user_id}")
            return profile_text

    except Exception as e:
        logger.error(f"Error generating profile for user {user_id}: {e}")
        return None

    finally:
        if should_close_session:
            await session.close()
