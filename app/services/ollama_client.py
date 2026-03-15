import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional

import aiohttp

from app.core.config import settings
from app.core.logger import logger
from app.models.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChoiceMessage,
    ChunkChoice,
    DeltaMessage,
    UsageInfo,
)
from app.services.backend_manager import backend_manager, BackendInfo
from app.services.proxy_manager import proxy_manager


class OllamaClient:
    _session: aiohttp.ClientSession | None = None
    _session_lock = asyncio.Lock()

    @classmethod
    async def init(cls):
        await cls._get_session()

    @classmethod
    async def shutdown(cls):
        async with cls._session_lock:
            if cls._session:
                await cls._session.close()
                cls._session = None

    @classmethod
    async def _get_session(cls) -> aiohttp.ClientSession:
        if cls._session and not cls._session.closed:
            return cls._session
        async with cls._session_lock:
            if cls._session and not cls._session.closed:
                return cls._session
            connector = aiohttp.TCPConnector(
                limit=200,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            cls._session = aiohttp.ClientSession(connector=connector)
            return cls._session

    @staticmethod
    async def chat(request: ChatCompletionRequest):
        max_retries = settings.max_retries
        exclude = set()
        last_error = None

        for attempt in range(max_retries):
            backend = await backend_manager.get_backend(
                model=request.model, exclude=exclude
            )
            if not backend:
                break

            key = f"{backend.ip}:{backend.port}"
            try:
                if request.stream:
                    return OllamaClient._stream_chat(backend, request)
                else:
                    return await OllamaClient._normal_chat(backend, request)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Backend {key} failed (attempt {attempt+1}): {e}"
                )
                await backend_manager.record_failure(backend)
                exclude.add(key)

        raise Exception(
            f"All backends failed after {max_retries} retries: {last_error}"
        )

    @staticmethod
    def _build_payload(request: ChatCompletionRequest) -> dict:
        messages = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg.content, list):
                parts = []
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                messages.append({"role": msg.role, "content": " ".join(parts)})
            else:
                messages.append({"role": msg.role, "content": str(msg.content)})

        payload = {"model": request.model, "messages": messages, "stream": request.stream}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop is not None:
            payload["stop"] = request.stop
        if request.frequency_penalty is not None:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            payload["presence_penalty"] = request.presence_penalty
        return payload

    @staticmethod
    async def _get_proxy_url():
        return await proxy_manager.get_proxy_url()

    @staticmethod
    def _build_headers(backend: BackendInfo) -> dict:
        headers = {}
        if backend.api_key:
            headers["Authorization"] = f"Bearer {backend.api_key}"
        return headers

    @staticmethod
    async def _normal_chat(
        backend: BackendInfo, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        url = f"{backend.base_url}/v1/chat/completions"
        payload = OllamaClient._build_payload(request)
        payload["stream"] = False
        payload["model"] = backend.resolve_model(request.model)
        proxy = await OllamaClient._get_proxy_url()
        headers = OllamaClient._build_headers(backend)

        start = time.time()
        timeout = aiohttp.ClientTimeout(
            total=settings.request_timeout, connect=settings.connect_timeout
        )
        session = await OllamaClient._get_session()
        async with session.post(url, json=payload, proxy=proxy, timeout=timeout, headers=headers) as resp:
            latency = (time.time() - start) * 1000
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"HTTP {resp.status}: {text[:200]}")
            data = await resp.json()

        await backend_manager.record_success(backend, latency)

        if "choices" in data:
            return ChatCompletionResponse(
                id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}"),
                model=request.model,
                choices=[
                    Choice(
                        index=c.get("index", 0),
                        message=ChoiceMessage(
                            role=c.get("message", {}).get("role", "assistant"),
                            content=c.get("message", {}).get("content", ""),
                        ),
                        finish_reason=c.get("finish_reason", "stop"),
                    )
                    for c in data["choices"]
                ],
                usage=UsageInfo(**(data.get("usage") or {})),
            )

        content = data.get("message", {}).get("content", "")
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=request.model,
            choices=[Choice(message=ChoiceMessage(content=content))],
        )

    @staticmethod
    async def _stream_chat(
        backend: BackendInfo, request: ChatCompletionRequest
    ) -> AsyncGenerator[str, None]:
        """流式聊天 - 直接作为 async generator 使用"""
        url = f"{backend.base_url}/v1/chat/completions"
        payload = OllamaClient._build_payload(request)
        payload["stream"] = True
        payload["model"] = backend.resolve_model(request.model)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        proxy = await OllamaClient._get_proxy_url()

        start = time.time()
        timeout = aiohttp.ClientTimeout(
            total=settings.request_timeout, connect=settings.connect_timeout
        )
        headers = OllamaClient._build_headers(backend)
        try:
            session = await OllamaClient._get_session()
            async with session.post(url, json=payload, proxy=proxy, timeout=timeout, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"HTTP {resp.status}: {text[:200]}")

                buf = b""
                async for raw in resp.content.iter_any():
                    if not raw:
                        continue
                    buf += raw
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            line = line[6:]
                        if line == "[DONE]":
                            buf = b""
                            break
                        try:
                            chunk_data = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "choices" in chunk_data:
                            for c in chunk_data["choices"]:
                                delta = c.get("delta", {})
                                chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    model=request.model,
                                    choices=[
                                        ChunkChoice(
                                            index=c.get("index", 0),
                                            delta=DeltaMessage(
                                                role=delta.get("role"),
                                                content=delta.get("content"),
                                            ),
                                            finish_reason=c.get("finish_reason"),
                                        )
                                    ],
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                        else:
                            content = chunk_data.get("message", {}).get("content", "")
                            if content:
                                chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    model=request.model,
                                    choices=[ChunkChoice(delta=DeltaMessage(content=content))],
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"

                yield "data: [DONE]\n\n"

            latency = (time.time() - start) * 1000
            await backend_manager.record_success(backend, latency)

        except Exception as e:
            await backend_manager.record_failure(backend)
            error_chunk = ChatCompletionChunk(
                id=completion_id,
                model=request.model,
                choices=[
                    ChunkChoice(
                        delta=DeltaMessage(content=f"\n\n[Error: {e}]"),
                        finish_reason="stop",
                    )
                ],
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
