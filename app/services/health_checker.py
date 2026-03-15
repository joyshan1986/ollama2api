import asyncio
import time

import json

import aiohttp

from app.core.config import settings
from app.core.constants import TARGET_MODELS
from app.core.logger import logger
from app.services.backend_manager import backend_manager


class HealthChecker:
    def __init__(self):
        self._task: asyncio.Task = None
        self._running = False
        self._progress = {"total": 0, "checked": 0, "running": False}
        self._session: aiohttp.ClientSession | None = None

    async def init(self):
        connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300, enable_cleanup_closed=True)
        self._session = aiohttp.ClientSession(connector=connector)
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Health checker started")

    async def _loop(self):
        await asyncio.sleep(10)
        while self._running:
            try:
                await self.check_all()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            await asyncio.sleep(settings.health_check_interval)

    async def check_all(self):
        backends = backend_manager.get_all()
        self._progress = {"total": len(backends), "checked": 0, "running": True}

        sem = asyncio.Semaphore(10)

        async def check_one(info):
            key = info["key"]
            b = backend_manager.get_backend_by_key(key)
            if not b or not b.enabled:
                self._progress["checked"] += 1
                return
            async with sem:
                await self._check_backend(b)
            self._progress["checked"] += 1

        tasks = [check_one(info) for info in backends]
        await asyncio.gather(*tasks, return_exceptions=True)
        self._progress["running"] = False
        await backend_manager.flush()
        logger.info(
            f"Health check complete: {self._progress['checked']}/{self._progress['total']}"
        )

    async def _check_backend(self, b):
        headers = {}
        if b.api_key:
            headers["Authorization"] = f"Bearer {b.api_key}"

        if b.backend_type == "cloud":
            await self._check_cloud_backend(b, headers)
        else:
            await self._check_local_backend(b, headers)

    async def _check_local_backend(self, b, headers: dict):
        url = f"{b.base_url}/api/tags"
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        start = time.time()
        try:
            session = self._session
            if not session:
                return
            async with session.get(url, timeout=timeout, headers=headers) as resp:
                latency = (time.time() - start) * 1000
                if resp.status == 200:
                    data = await resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    valid_models = [m for m in models if m]
                    failed = await self._test_models(b, valid_models, headers)
                    await backend_manager.update_health(
                        b, models=valid_models, failed_models=failed,
                        status="online", latency_ms=latency,
                    )
                else:
                    await backend_manager.update_health(b, status="offline")
        except Exception:
            await backend_manager.update_health(b, status="offline")

    # Representative models to probe cloud connectivity
    CLOUD_PROBE_MODELS = ["glm-5", "kimi-k2.5", "minimax-m2.5", "qwen3.5"]

    async def _check_cloud_backend(self, b, headers: dict):
        """Cloud 后端：先 /api/tags 获取完整模型列表，再抽样测试连通性"""
        session = self._session
        if not session:
            return
        start = time.time()

        # Step 1: GET /api/tags to discover all available models
        valid_models = []
        tags_url = f"{b.base_url}/api/tags"
        try:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            async with session.get(tags_url, timeout=timeout, headers=headers) as resp:
                if resp.status != 200:
                    await backend_manager.update_health(b, status="offline")
                    return
                data = await resp.json()
                valid_models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            await backend_manager.update_health(b, status="offline")
            return

        if not valid_models:
            await backend_manager.update_health(b, status="offline")
            return

        # Step 2: Probe a few representative models via /api/chat
        failed_models = []
        chat_url = f"{b.base_url}/api/chat"
        probe_timeout = aiohttp.ClientTimeout(total=30, connect=5)
        to_probe = [m for m in valid_models if m.split(":")[0] in self.CLOUD_PROBE_MODELS]
        for model in to_probe:
            try:
                payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50, "stream": False}
                async with session.post(chat_url, json=payload, timeout=probe_timeout, headers=headers) as resp:
                    if resp.status != 200:
                        failed_models.append(model)
                        logger.info(f"Cloud probe failed: {b.ip} / {model} -> HTTP {resp.status}")
            except Exception as e:
                failed_models.append(model)
                logger.info(f"Cloud probe failed: {b.ip} / {model} -> {e}")

        latency = (time.time() - start) * 1000
        status = "offline" if to_probe and len(failed_models) == len(to_probe) else "online"
        await backend_manager.update_health(
            b, models=valid_models, failed_models=failed_models,
            status=status, latency_ms=latency,
        )

    async def _test_models(self, b, models: list, headers: dict = None) -> list:
        to_test = [m for m in models if m.split(":")[0] in TARGET_MODELS]
        if not to_test:
            return []
        failed = []
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        url = f"{b.base_url}/v1/chat/completions"
        payload_base = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 50, "stream": False}
        session = self._session
        if not session:
            return to_test
        for model in to_test:
            try:
                payload = {**payload_base, "model": model}
                async with session.post(url, json=payload, timeout=timeout, headers=headers) as resp:
                    if resp.status != 200:
                        failed.append(model)
                        logger.info(f"Model test failed: {b.ip} / {model} -> HTTP {resp.status}")
            except Exception as e:
                failed.append(model)
                logger.info(f"Model test failed: {b.ip} / {model} -> {e}")
        if failed:
            logger.info(f"Backend {b.ip}: failed models {failed}")
        return failed

    def get_progress(self) -> dict:
        return dict(self._progress)

    async def shutdown(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()
            self._session = None


health_checker = HealthChecker()
