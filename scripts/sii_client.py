"""
Async OpenAI-compatible client for SII API.

Usage:
    from scripts.sii_client import SIIClient
    client = SIIClient()
    result = await client.chat([{"role": "user", "content": "Hello"}])
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Any


class SIIClient:
    """Async client for SII API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen3.5-397b-w8a8",
        max_concurrent: int = 20,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ["INF_API_KEY"]
        self.base_url = base_url or os.environ.get(
            "INF_API_BASE",
            "https://gdopegmaqmedckqgm58mj9j5kdmaoj9b.openapi-sj.sii.edu.cn/v1",
        )
        self.model = model
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client = None

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=120.0,
            )
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: int = 2048,
    ) -> str:
        """Single chat completion with retries."""
        async with self._semaphore:
            for attempt in range(self.max_retries):
                try:
                    client = await self._get_client()
                    kwargs = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if response_format:
                        kwargs["response_format"] = response_format

                    response = await client.chat.completions.create(**kwargs)
                    msg = response.choices[0].message
                    # Some models return content in 'reasoning' field instead of 'content'
                    return msg.content or getattr(msg, 'reasoning', None) or ""
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait = 2 ** attempt
                    print(f"  API error (attempt {attempt+1}/{self.max_retries}): {e}, "
                          f"retrying in {wait}s...")
                    await asyncio.sleep(wait)

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Any:
        """Chat completion with JSON response parsing."""
        text = await self.chat(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        # Try to extract JSON from response (handle markdown code blocks)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:]) if lines[0].startswith("```") else text
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from: {text[:500]}...")
            return None

    async def batch_chat_json(
        self,
        prompts: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> List[Any]:
        """Process multiple prompts concurrently."""
        tasks = [
            self.chat_json(msgs, temperature=temperature, max_tokens=max_tokens)
            for msgs in prompts
        ]
        return await asyncio.gather(*tasks)
