import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.logger import logger
from app.core.storage import storage_manager


@dataclass
class BackendInfo:
    ip: str
    port: int = 11434
    models: List[str] = field(default_factory=list)
    status: str = "unknown"  # online / offline / cooldown / unknown
    enabled: bool = True
    latency_ms: float = 0
    fail_count: int = 0
    consecutive_failures: int = 0
    cooldown_until: float = 0
    last_check: float = 0
    last_used: float = 0
    request_count: int = 0
    success_count: int = 0
    failed_models: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    api_key: Optional[str] = None
    scheme: str = "http"
    backend_type: str = "local"  # "local" (OpenAI compat) or "cloud" (Ollama native)

    @property
    def node_key(self) -> str:
        base = f"{self.ip}:{self.port}"
        if self.api_key:
            suffix = hashlib.sha256(self.api_key.encode()).hexdigest()[:8]
            return f"{base}@{suffix}"
        return base

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.ip}:{self.port}"

    @property
    def is_available(self) -> bool:
        return (
            self.enabled
            and self.status != "offline"
            and self.status != "cooldown"
            and time.time() > self.cooldown_until
        )

    def resolve_model(self, short_name: str) -> str:
        for m in self.models:
            if m == short_name or m.split(":")[0] == short_name:
                return m
        return short_name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["base_url"] = self.base_url
        d["is_available"] = self.is_available
        return d


class BackendManager:
    def __init__(self):
        self._backends: Dict[str, BackendInfo] = {}
        self._lock = asyncio.Lock()
        self._dirty = False
        self._save_task: asyncio.Task = None
        # Hot cache: model -> (timestamp, [(key, score)])
        self._hot_cache: Dict[str, tuple] = {}
        self._cache_ttl = 60  # seconds

    async def init(self):
        data = await storage_manager.load_json("backends.json", default={})
        if data and "backends" in data:
            for key, info in data["backends"].items():
                try:
                    self._backends[key] = BackendInfo(**info)
                except Exception:
                    pass
        if not self._backends:
            await self._import_from_file()
        self._save_task = asyncio.create_task(self._periodic_save())
        logger.info(f"Backend manager initialized: {len(self._backends)} backends")

    async def _periodic_save(self):
        while True:
            await asyncio.sleep(30)
            try:
                if self._dirty:
                    await self._save()
                    self._dirty = False
            except Exception:
                pass

    async def _import_from_file(self):
        import os
        paths = [
            os.path.join(settings.storage_path, "hit_ips.txt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..", "..", "scan-server", "hit_ips.txt"),
        ]
        for path in paths:
            if os.path.exists(path):
                logger.info(f"Importing IPs from {path}")
                with open(path, "r") as f:
                    ips = [line.strip() for line in f if line.strip()]
                await self.add_backends_batch(ips)
                return
        logger.warning("No hit_ips.txt found, starting with empty backend pool")

    async def add_backends_batch(self, ips: List[str], port: int = None,
                                api_key: Optional[str] = None,
                                scheme: str = "http",
                                backend_type: str = "local") -> dict:
        port = port or settings.default_ollama_port
        added, skipped = 0, 0
        async with self._lock:
            for ip in ips:
                ip = ip.strip()
                if not ip:
                    continue
                b = BackendInfo(ip=ip, port=port, api_key=api_key, scheme=scheme, backend_type=backend_type)
                key = b.node_key
                if key in self._backends:
                    skipped += 1
                    continue
                self._backends[key] = b
                added += 1
        await self._save()
        logger.info(f"Batch add: {added} added, {skipped} skipped")
        return {"added": added, "skipped": skipped}

    async def remove_backend(self, key: str) -> bool:
        async with self._lock:
            if key in self._backends:
                del self._backends[key]
                await self._save()
                return True
        return False

    def _score_backend(self, b: BackendInfo, now: float) -> float:
        """Score a backend. Lower = better. Factors: latency, success rate, recency, failures."""
        lat = b.latency_ms if b.latency_ms > 0 else 5000
        total = max(b.request_count, 1)
        success_rate = b.success_count / total
        recency = max(0, 300 - (now - b.last_used)) / 300 if b.last_used > 0 else 0
        return lat * (1 + b.consecutive_failures * 3) / (0.3 + success_rate + recency * 0.5)

    async def get_backend(self, model: str = None, exclude: set = None) -> Optional[BackendInfo]:
        exclude = exclude or set()
        now = time.time()
        cache_key = model or "__all__"

        async with self._lock:
            # Try hot cache first
            if cache_key in self._hot_cache:
                ts, cached = self._hot_cache[cache_key]
                if now - ts < self._cache_ttl:
                    valid = [(k, self._backends[k]) for k, _ in cached
                             if k in self._backends and k not in exclude and self._backends[k].is_available]
                    if valid:
                        weights = [1.0 / (i + 1) ** 1.5 for i in range(len(valid))]
                        key, backend = random.choices(valid, weights=weights, k=1)[0]
                        backend.last_used = now
                        backend.request_count += 1
                        return backend

            # Build candidate list
            candidates = []
            for key, b in self._backends.items():
                if key in exclude:
                    continue
                if not b.is_available:
                    if b.cooldown_until and now > b.cooldown_until:
                        b.status = "unknown"
                        b.consecutive_failures = 0
                    else:
                        continue
                if model and b.models and not any(m.split(":")[0] == model or m == model for m in b.models):
                    continue
                if model and any(m.split(":")[0] == model or m == model for m in b.failed_models):
                    continue
                candidates.append((key, b))

            if not candidates:
                return None

            # Score and sort
            scored = [(k, b, self._score_backend(b, now)) for k, b in candidates]
            scored.sort(key=lambda x: x[2])

            # Cache top backends
            top_n = max(3, len(scored) // 3)
            self._hot_cache[cache_key] = (now, [(k, s) for k, _, s in scored[:top_n]])

            # Weighted random from top pool (favor best)
            pool = scored[:top_n]
            weights = [1.0 / (i + 1) ** 1.5 for i in range(len(pool))]
            key, backend, _ = random.choices(pool, weights=weights, k=1)[0]
            backend.last_used = now
            backend.request_count += 1
            return backend

    async def record_success(self, backend: BackendInfo, latency_ms: float = 0):
        async with self._lock:
            backend.consecutive_failures = 0
            backend.success_count += 1
            backend.status = "online"
            if latency_ms > 0:
                backend.latency_ms = latency_ms
            self._dirty = True

    async def record_failure(self, backend: BackendInfo):
        async with self._lock:
            backend.fail_count += 1
            backend.consecutive_failures += 1
            threshold = settings.cooldown_threshold
            if backend.consecutive_failures >= threshold:
                backend.cooldown_until = time.time() + settings.cooldown_duration
                backend.status = "cooldown"
                logger.warning(
                    f"Backend {backend.ip}:{backend.port} entered cooldown "
                    f"({backend.consecutive_failures} consecutive failures)"
                )
            # Invalidate hot cache so failed backend gets re-scored
            self._hot_cache.clear()
            self._dirty = True

    async def update_health(self, backend: 'BackendInfo', models: list = None,
                            failed_models: list = None, status: str = None,
                            latency_ms: float = None):
        """健康检查专用：批量更新后端状态（公共接口，替代直接访问 _lock）"""
        async with self._lock:
            if models is not None:
                backend.models = models
            if failed_models is not None:
                backend.failed_models = failed_models
            if status is not None:
                backend.status = status
            if latency_ms is not None:
                backend.latency_ms = latency_ms
            backend.last_check = time.time()
            if status == "online":
                backend.consecutive_failures = 0
                backend.cooldown_until = 0
            self._dirty = True

    async def flush(self):
        """立即持久化当前状态到磁盘"""
        await self._save()

    async def update_backend(self, key: str, **kwargs) -> bool:
        async with self._lock:
            b = self._backends.get(key)
            if not b:
                return False
            for k, v in kwargs.items():
                if hasattr(b, k):
                    setattr(b, k, v)
            await self._save()
            return True

    async def clear_cooldown(self, key: str) -> bool:
        async with self._lock:
            b = self._backends.get(key)
            if not b:
                return False
            b.cooldown_until = 0
            b.consecutive_failures = 0
            b.status = "unknown"
            await self._save()
            return True

    def get_all(self) -> List[dict]:
        return [
            {"key": k, **b.to_dict()}
            for k, b in sorted(self._backends.items(), key=lambda x: x[1].created_at)
        ]

    def get_stats(self) -> dict:
        total = len(self._backends)
        online = sum(1 for b in self._backends.values() if b.status == "online")
        offline = sum(1 for b in self._backends.values() if b.status == "offline")
        cooldown = sum(1 for b in self._backends.values() if b.status == "cooldown")
        enabled = sum(1 for b in self._backends.values() if b.enabled)
        models = set()
        for b in self._backends.values():
            models.update(b.models)
        return {
            "total": total,
            "online": online,
            "offline": offline,
            "cooldown": cooldown,
            "enabled": enabled,
            "models": sorted(models),
        }

    def get_backend_by_key(self, key: str) -> Optional[BackendInfo]:
        return self._backends.get(key)

    async def _save(self):
        data = {"backends": {k: asdict(v) for k, v in self._backends.items()}}
        await storage_manager.save_json("backends.json", data)

    async def shutdown(self):
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        await self._save()


backend_manager = BackendManager()
