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

    async def _check_cloud_backend(self, b, headers: dict):
        """Cloud 后端：直接用 /api/chat 逐个测试 TARGET_MODELS"""
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        url = f"{b.base_url}/api/chat"
        session = self._session
        if not session:
            return
        valid_models = []
        failed_models = []
        start = time.time()
        for model in TARGET_MODELS:
            try:
                payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50, "stream": False}
                async with session.post(url, json=payload, timeout=timeout, headers=headers) as resp:
                    if resp.status == 200:
                        valid_models.append(model)
                    else:
                        failed_models.append(model)
                        logger.info(f"Cloud model test failed: {b.ip} / {model} -> HTTP {resp.status}")
            except Exception as e:
                failed_models.append(model)
                logger.info(f"Cloud model test failed: {b.ip} / {model} -> {e}")
        latency = (time.time() - start) * 1000
        status = "online" if valid_models else "offline"
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
