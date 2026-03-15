import asyncio
import json
import os
import time
from typing import Optional

import aiohttp
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.core.auth import require_admin, cleanup_sessions, create_session
from app.core.config import settings, runtime_config
from app.core.constants import TARGET_MODELS
from app.services.backend_manager import backend_manager
from app.services.api_keys import api_key_manager
from app.services.request_stats import request_stats
from app.services.request_logger import request_logger
from app.services.health_checker import health_checker
from app.services.scanner import scanner_service

router = APIRouter()


# --- Auth ---

class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/admin/api/login")
async def admin_login(body: LoginRequest):
    cleanup_sessions()
    if body.username == settings.admin_username and body.password == settings.admin_password:
        token = create_session(body.username)
        return {"success": True, "token": token}
    return JSONResponse(status_code=401, content={"success": False, "message": "账户或密码错误"})


@router.get("/admin/login", response_class=HTMLResponse)
async def admin_login_page():
    tpl = os.path.join(os.path.dirname(os.path.dirname(__file__)), "template", "login.html")
    if os.path.exists(tpl):
        with open(tpl, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Login page not found</h1>")


@router.get("/admin", response_class=HTMLResponse)
async def admin_page():
    tpl = os.path.join(os.path.dirname(os.path.dirname(__file__)), "template", "admin.html")
    if os.path.exists(tpl):
        with open(tpl, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Admin page not found</h1>")


# --- Backends ---

class AddBackendsRequest(BaseModel):
    ips: str
    port: Optional[int] = None
    api_key: Optional[str] = None
    scheme: str = "http"
    backend_type: str = "local"


class UpdateBackendRequest(BaseModel):
    enabled: Optional[bool] = None


@router.get("/admin/api/backends")
async def list_backends(session=Depends(require_admin)):
    return {
        "backends": backend_manager.get_all(),
        "stats": backend_manager.get_stats(),
    }


@router.post("/admin/api/backends")
async def add_backends(body: AddBackendsRequest, session=Depends(require_admin)):
    ips = [ip.strip() for ip in body.ips.replace(",", "\n").split("\n") if ip.strip()]
    result = await backend_manager.add_backends_batch(
        ips, body.port, api_key=body.api_key, scheme=body.scheme,
        backend_type=body.backend_type,
    )
    return {"success": True, **result}


@router.put("/admin/api/backends/{key:path}")
async def update_backend(key: str, body: UpdateBackendRequest, session=Depends(require_admin)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    ok = await backend_manager.update_backend(key, **updates)
    return {"success": ok}


@router.delete("/admin/api/backends/{key:path}")
async def delete_backend(key: str, session=Depends(require_admin)):
    ok = await backend_manager.remove_backend(key)
    return {"success": ok}


@router.post("/admin/api/backends/{key:path}/clear-cooldown")
async def clear_cooldown(key: str, session=Depends(require_admin)):
    ok = await backend_manager.clear_cooldown(key)
    return {"success": ok}


@router.post("/admin/api/backends/health-check")
async def trigger_health_check(session=Depends(require_admin)):
    asyncio.create_task(health_checker.check_all())
    return {"success": True, "message": "Health check started"}


@router.get("/admin/api/backends/health-progress")
async def health_progress(session=Depends(require_admin)):
    return health_checker.get_progress()


# --- Stats ---

@router.get("/admin/api/stats/summary")
async def stats_summary(session=Depends(require_admin)):
    return request_stats.get_summary()


@router.get("/admin/api/stats/hourly")
async def stats_hourly(hours: int = 24, session=Depends(require_admin)):
    return {"data": request_stats.get_hourly(hours)}


@router.get("/admin/api/stats/daily")
async def stats_daily(days: int = 7, session=Depends(require_admin)):
    return {"data": request_stats.get_daily(days)}


# --- Logs ---

@router.get("/admin/api/logs")
async def get_logs(limit: int = 50, offset: int = 0, session=Depends(require_admin)):
    return request_logger.get_logs(limit, offset)


@router.delete("/admin/api/logs")
async def clear_logs(session=Depends(require_admin)):
    await request_logger.clear()
    return {"success": True}


# --- API Keys ---

class CreateKeyRequest(BaseModel):
    name: str = ""


class CreateKeysBatchRequest(BaseModel):
    names: str = ""


class UpdateKeyRequest(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None


class DeleteKeysBatchRequest(BaseModel):
    keys: list


@router.get("/admin/api/keys")
async def list_keys(session=Depends(require_admin)):
    return {"keys": api_key_manager.get_all(), "stats": api_key_manager.get_stats()}


@router.post("/admin/api/keys")
async def create_key(body: CreateKeyRequest, session=Depends(require_admin)):
    info = await api_key_manager.create_key(body.name)
    return {"success": True, "key": info.key}


@router.post("/admin/api/keys/batch")
async def create_keys_batch(body: CreateKeysBatchRequest, session=Depends(require_admin)):
    names = [n.strip() for n in body.names.replace(",", "\n").split("\n") if n.strip()]
    if not names:
        names = [""]
    keys = await api_key_manager.create_keys_batch(names)
    return {"success": True, "keys": [k.key for k in keys]}


@router.put("/admin/api/keys/{key}")
async def update_key(key: str, body: UpdateKeyRequest, session=Depends(require_admin)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    ok = await api_key_manager.update_key(key, **updates)
    return {"success": ok}


@router.delete("/admin/api/keys/batch")
async def delete_keys_batch(body: DeleteKeysBatchRequest, session=Depends(require_admin)):
    count = await api_key_manager.delete_keys_batch(body.keys)
    return {"success": True, "deleted": count}


# --- Config ---

@router.get("/admin/api/config")
async def get_config(session=Depends(require_admin)):
    return {"schema": runtime_config.get_schema(), "values": runtime_config.get_all()}


class UpdateConfigRequest(BaseModel):
    updates: dict


@router.put("/admin/api/config")
async def update_config(body: UpdateConfigRequest, session=Depends(require_admin)):
    changed = runtime_config.set_batch(body.updates)
    return {"success": True, "changed": changed}


@router.post("/admin/api/config/reset")
async def reset_config(session=Depends(require_admin)):
    runtime_config.reset()
    return {"success": True}


# --- AI Chat ---

class AIChatRequest(BaseModel):
    messages: list
    model: str = "glm-4.7"


def _build_system_prompt() -> str:
    report = scanner_service.get_system_report()
    stats = report["backend_stats"]
    md = report["model_distribution"]
    fd = report["failed_model_distribution"]
    sc = report["scan_stats"]
    recommended = scanner_service.get_recommended_ranges()
    unscanned = [r for r in recommended if not r["scanned"]]

    model_lines = []
    for m in TARGET_MODELS:
        ok = md.get(m, 0)
        fail = fd.get(m, 0)
        model_lines.append(f"  - {m}: {ok} 可用, {fail} 不可用")

    unscanned_lines = []
    for r in unscanned[:6]:
        mins = r["estimated_seconds"] // 60
        unscanned_lines.append(
            f"  - {r['name']} ({r['country']}): {r['start']}-{r['end']}, "
            f"约 {r['ip_count']} IP, 预计 {mins} 分钟 — {r['description']}"
        )

    # Build config summary
    cfg = runtime_config.get_all()
    cfg_lines = [f"  - {k}: {v}" for k, v in cfg.items() if k not in ("admin_password",)]

    # Build backend list summary
    backends = backend_manager.get_all()
    online_backends = [b for b in backends if b["status"] == "online"]
    offline_backends = [b for b in backends if b["status"] == "offline"]
    cooldown_backends = [b for b in backends if b["status"] == "cooldown"]

    # Build key summary
    keys_all = api_key_manager.get_all()
    keys_stats = api_key_manager.get_stats()

    return f"""你是 Ollama2API 智能管理助手，拥有系统的完整管理权限。

## 你的身份
你是这个系统的 AI 运维助手，可以直接执行操作：管理后端节点、扫描发现新节点、管理 API Key、修改系统配置、诊断问题、优化集群。

## 当前系统状态
- 后端总数: {stats['total']}, 在线: {stats['online']}, 离线: {stats['offline']}, 冷却中: {stats['cooldown']}
- 已启用: {stats['enabled']}, 可用模型种类: {len(stats['models'])}
- 模型分布:
{chr(10).join(model_lines)}
- 扫描统计: 已扫描 {sc['total_ranges']} 个段, 总发现 {sc['total_found']}, 总入库 {sc['total_usable']}
- 扫描器状态: {'扫描中' if report['scanning'] else '空闲'}
- 扫描引擎: {'masscan (高速)' if report.get('masscan_available') else '纯 Python (较慢)'}
- API Key: 共 {keys_stats.get('total', 0)} 个, 启用 {keys_stats.get('enabled', 0)} 个

## 当前配置
{chr(10).join(cfg_lines)}

## 在线后端 ({len(online_backends)} 个)
{chr(10).join(f"  - {b['key']}" for b in online_backends[:20]) if online_backends else '  无'}
{'  ... 等' + str(len(online_backends)) + '个' if len(online_backends) > 20 else ''}

## 离线后端 ({len(offline_backends)} 个)
{chr(10).join(f"  - {b['key']}" for b in offline_backends[:10]) if offline_backends else '  无'}

## 冷却中后端 ({len(cooldown_backends)} 个)
{chr(10).join(f"  - {b['key']}" for b in cooldown_backends[:10]) if cooldown_backends else '  无'}

## 未扫描的推荐地区
{chr(10).join(unscanned_lines) if unscanned_lines else '  所有推荐地区已扫描完毕！'}

## 你的操作能力（通过 ACTION 命令执行）
你可以在回复中插入操作命令，格式为 `[ACTION:操作名:参数]`，前端会解析并执行。

可用操作：
1. `[ACTION:scan:起始IP:结束IP]` — 启动 IP 段扫描
2. `[ACTION:auto_scan]` — 一键扫描所有未扫描的推荐地区
3. `[ACTION:health_check]` — 触发全局健康检查
4. `[ACTION:cleanup]` — 清理离线超24小时的后端
5. `[ACTION:add_backend:IP地址:端口]` — 添加后端节点
6. `[ACTION:remove_backend:IP:端口]` — 删除后端节点
7. `[ACTION:toggle_backend:IP:端口:true/false]` — 启用/禁用后端
8. `[ACTION:clear_cooldown:IP:端口]` — 清除后端冷却状态
9. `[ACTION:create_key:备注名]` — 创建 API Key
10. `[ACTION:delete_key:key值]` — 删除 API Key
11. `[ACTION:toggle_key:key值:true/false]` — 启用/禁用 Key
12. `[ACTION:set_config:配置项:值]` — 修改系统配置
13. `[ACTION:clean_useless]` — 清理无可用目标模型的后端

## 操作规则
- 执行危险操作（删除、清理）前，先告知用户影响范围，等用户确认后再输出 ACTION 命令
- 扫描操作可以直接执行，不需要确认
- 每次回复最多包含 5 个 ACTION 命令
- ACTION 命令单独一行，前后不要有其他文字
- 推荐扫描时列出表格：地区名 | 国家 | IP段 | IP数 | 预计时间
- 用中文回答，给出明确的操作建议
- 如果系统有问题（大量离线、模型不可用），主动提醒用户"""


@router.post("/admin/api/ai/chat")
async def ai_chat(body: AIChatRequest, session=Depends(require_admin)):
    system_prompt = _build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}] + body.messages

    backend = await backend_manager.get_backend(model=body.model)
    if not backend:
        return JSONResponse(status_code=503, content={"error": "无可用后端"})

    url = f"{backend.base_url}/v1/chat/completions"
    payload = {"model": backend.resolve_model(body.model), "messages": messages, "stream": True}
    req_headers = {}
    if backend.api_key:
        req_headers["Authorization"] = f"Bearer {backend.api_key}"

    async def generate():
        timeout = aiohttp.ClientTimeout(total=settings.request_timeout, connect=settings.connect_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(url, json=payload, headers=req_headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        yield f"data: {json.dumps({'error': text[:200]})}\n\n"
                        return
                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if decoded:
                            yield decoded + "\n"
                            if decoded.strip() == "data: [DONE]":
                                return
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


class AIActionRequest(BaseModel):
    action: str
    params: list = []


@router.post("/admin/api/ai/execute")
async def ai_execute(body: AIActionRequest, session=Depends(require_admin)):
    a, p = body.action, body.params
    try:
        if a == "scan" and len(p) >= 2:
            asyncio.create_task(scanner_service.scan_range(p[0], p[1]))
            return {"success": True, "message": f"扫描 {p[0]}-{p[1]} 已启动"}
        elif a == "auto_scan":
            asyncio.create_task(scanner_service.auto_scan_recommended())
            return {"success": True, "message": "自动扫描已启动"}
        elif a == "health_check":
            asyncio.create_task(health_checker.check_all())
            return {"success": True, "message": "健康检查已启动"}
        elif a == "cleanup":
            removed = await scanner_service.cleanup_offline()
            return {"success": True, "message": f"已清理 {removed} 个离线后端"}
        elif a == "add_backend" and len(p) >= 1:
            port = int(p[1]) if len(p) > 1 else settings.default_ollama_port
            result = await backend_manager.add_backends_batch([p[0]], port)
            return {"success": True, "message": f"添加 {result.get('added', 0)} 个, 跳过 {result.get('skipped', 0)} 个"}
        elif a == "remove_backend" and len(p) >= 1:
            key = f"{p[0]}:{p[1]}" if len(p) > 1 else p[0]
            ok = await backend_manager.remove_backend(key)
            return {"success": ok, "message": f"{'已删除' if ok else '未找到'} {key}"}
        elif a == "toggle_backend" and len(p) >= 2:
            key = f"{p[0]}:{p[1]}" if len(p) > 2 else p[0]
            enabled = p[-1].lower() == "true"
            ok = await backend_manager.update_backend(key, enabled=enabled)
            return {"success": ok, "message": f"{key} 已{'启用' if enabled else '禁用'}"}
        elif a == "clear_cooldown" and len(p) >= 1:
            key = f"{p[0]}:{p[1]}" if len(p) > 1 else p[0]
            ok = await backend_manager.clear_cooldown(key)
            return {"success": ok, "message": f"已清除 {key} 冷却"}
        elif a == "create_key":
            name = p[0] if p else ""
            info = await api_key_manager.create_key(name)
            return {"success": True, "message": f"Key 已创建: {info.key}"}
        elif a == "delete_key" and len(p) >= 1:
            count = await api_key_manager.delete_keys_batch(p)
            return {"success": True, "message": f"已删除 {count} 个 Key"}
        elif a == "toggle_key" and len(p) >= 2:
            enabled = p[1].lower() == "true"
            ok = await api_key_manager.update_key(p[0], enabled=enabled)
            return {"success": ok, "message": f"Key 已{'启用' if enabled else '禁用'}"}
        elif a == "set_config" and len(p) >= 2:
            ok = runtime_config.set(p[0], p[1])
            return {"success": ok, "message": f"配置 {p[0]} 已更新" if ok else f"无效配置项: {p[0]}"}
        elif a == "clean_useless":
            target = TARGET_MODELS
            backends = backend_manager.get_all()
            useless = []
            for b in backends:
                ok_models = set(m.split(":")[0] for m in (b.get("models") or [])) - set(m.split(":")[0] for m in (b.get("failed_models") or []))
                if not any(m in target for m in ok_models):
                    useless.append(b["key"])
            for key in useless:
                await backend_manager.remove_backend(key)
            return {"success": True, "message": f"已清理 {len(useless)} 个无用后端"}
        else:
            return {"success": False, "message": f"未知操作: {a}"}
    except Exception as e:
        return {"success": False, "message": f"执行失败: {str(e)}"}


# --- Scanner: test-ip-model, add-ip, smart-recommend ---

class TestIpModelRequest(BaseModel):
    ip: str
    model: str
    port: int = 11434


@router.post("/admin/api/scanner/test-ip-model")
async def test_ip_model(body: TestIpModelRequest, session=Depends(require_admin)):
    url = f"http://{body.ip}:{body.port}/v1/chat/completions"
    payload = {"model": body.model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": False}
    timeout = aiohttp.ClientTimeout(total=10, connect=5)
    try:
        t0 = time.time()
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, json=payload) as resp:
                latency = round((time.time() - t0) * 1000)
                if resp.status == 200:
                    return {"success": True, "latency_ms": latency}
                text = await resp.text()
                return {"success": False, "error": f"HTTP {resp.status}: {text[:100]}", "latency_ms": latency}
    except Exception as e:
        return {"success": False, "error": str(e)}


class AddIpRequest(BaseModel):
    ip: str
    port: int = 11434


@router.post("/admin/api/scanner/add-ip")
async def add_discovered_ip(body: AddIpRequest, session=Depends(require_admin)):
    result = await backend_manager.add_backends_batch([body.ip], body.port)
    return {"success": True, **result}


class ScanRangeRequest(BaseModel):
    start: str
    end: str
    force: bool = False


@router.post("/admin/api/scanner/scan-range")
async def trigger_scan_range(body: ScanRangeRequest, session=Depends(require_admin)):
    """触发 IP 段扫描（非阻塞，后台执行）"""
    if scanner_service._scanning:
        return {"success": False, "error": "扫描正在进行中"}
    asyncio.create_task(scanner_service.scan_range(body.start, body.end, force=body.force))
    return {"success": True, "message": f"扫描已启动: {body.start} - {body.end}"}


@router.get("/admin/api/scanner/progress")
async def scan_progress(session=Depends(require_admin)):
    """获取当前扫描进度"""
    return {
        "scanning": scanner_service._scanning,
        "progress": scanner_service._progress,
        "auto_queue": getattr(scanner_service, "_auto_queue", None),
    }


@router.get("/admin/api/scanner/smart-recommend")
async def smart_recommend(session=Depends(require_admin)):
    return scanner_service.get_smart_recommendations()


class AIRecommendRequest(BaseModel):
    model: str = "glm-5"


@router.post("/admin/api/scanner/ai-recommend")
async def ai_recommend(body: AIRecommendRequest, session=Depends(require_admin)):
    """Stream AI analysis of scan history to recommend ranges."""
    # Build context from scan data
    history = scanner_service.get_history()
    stats = scanner_service.get_stats()
    recommended = scanner_service.get_recommended_ranges()
    unscanned = [r for r in recommended if not r["scanned"]]

    # Summarize scan history by provider/country
    provider_summary = {}
    for h in history:
        for r in scanner_service.KNOWN_RANGES:
            if r["start"] == h["start"] and r["end"] == h["end"]:
                provider = r["name"].split()[0]
                country = r["country"]
                key = f"{provider}({country})"
                if key not in provider_summary:
                    provider_summary[key] = {"scanned": 0, "found": 0, "usable": 0}
                provider_summary[key]["scanned"] += 1
                provider_summary[key]["found"] += h.get("found", 0)
                provider_summary[key]["usable"] += h.get("usable", 0)
                break

    summary_lines = [f"  - {k}: 扫描{v['scanned']}段, 发现{v['found']}, 入库{v['usable']}, 命中率{v['found']/max(v['scanned'],1):.1f}/段"
                     for k, v in sorted(provider_summary.items(), key=lambda x: x[1]['found'], reverse=True)]

    unscanned_lines = [f"  - {r['name']} ({r['country']}): {r['start']}-{r['end']}, {r['ip_count']}IP — {r['description']}"
                       for r in unscanned]

    prompt = f"""请分析以下扫描数据，推荐最值得扫描的IP段。

## 扫描统计
- 已扫描: {stats.get('total_ranges', 0)}段, 总发现: {stats.get('total_found', 0)}, 总入库: {stats.get('total_usable', 0)}

## 各供应商/地区命中情况
{chr(10).join(summary_lines) if summary_lines else '  暂无数据'}

## 未扫描的候选段 ({len(unscanned)}个)
{chr(10).join(unscanned_lines[:30]) if unscanned_lines else '  全部已扫描'}

## 要求
1. 根据历史命中率、供应商质量、地区分布，分析哪些段最值得扫描
2. 给出你的分析推理过程
3. 最后用以下**严格JSON格式**输出推荐列表（必须是合法JSON数组）:
```json
[{{"name":"段名称","start":"起始IP","end":"结束IP","reason":"推荐理由"}}]
```
4. 推荐5-10个最优先的段，按优先级排序"""

    messages = [
        {"role": "system", "content": "你是IP段扫描策略分析专家。根据历史扫描数据分析命中规律，推荐最有价值的扫描目标。"},
        {"role": "user", "content": prompt},
    ]

    backend = await backend_manager.get_backend(model=body.model)
    if not backend:
        return JSONResponse(status_code=503, content={"error": "无可用后端"})

    url = f"{backend.base_url}/v1/chat/completions"
    payload = {"model": backend.resolve_model(body.model), "messages": messages, "stream": True}
    req_headers = {}
    if backend.api_key:
        req_headers["Authorization"] = f"Bearer {backend.api_key}"

    async def generate():
        timeout = aiohttp.ClientTimeout(total=settings.request_timeout, connect=settings.connect_timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(url, json=payload, headers=req_headers) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        yield f"data: {json.dumps({'error': text[:200]})}\n\n"
                        return
                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if decoded:
                            yield decoded + "\n"
                            if decoded.strip() == "data: [DONE]":
                                return
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- Scanner ---

class ScanRequest(BaseModel):
    start_ip: str
    end_ip: str
    force: bool = False


@router.post("/admin/api/scanner/start")
async def scanner_start(body: ScanRequest, session=Depends(require_admin)):
    asyncio.create_task(scanner_service.scan_range(body.start_ip, body.end_ip, body.force))
    return {"success": True, "message": "扫描已启动"}


@router.get("/admin/api/scanner/progress")
async def scanner_progress(session=Depends(require_admin)):
    return scanner_service.get_progress()


@router.get("/admin/api/scanner/history")
async def scanner_history(session=Depends(require_admin)):
    return {"history": scanner_service.get_history(), "stats": scanner_service.get_stats()}


@router.post("/admin/api/scanner/cleanup")
async def scanner_cleanup(session=Depends(require_admin)):
    removed = await scanner_service.cleanup_offline()
    return {"success": True, "removed": removed}


@router.get("/admin/api/scanner/recommend")
async def scanner_recommend(session=Depends(require_admin)):
    return {"ranges": scanner_service.get_recommended_ranges()}


class EstimateRequest(BaseModel):
    start_ip: str
    end_ip: str


@router.post("/admin/api/scanner/estimate")
async def scanner_estimate(body: EstimateRequest, session=Depends(require_admin)):
    return scanner_service.estimate_scan(body.start_ip, body.end_ip)


@router.post("/admin/api/scanner/auto-scan")
async def scanner_auto_scan(session=Depends(require_admin)):
    asyncio.create_task(scanner_service.auto_scan_recommended())
    return {"success": True, "message": "自动扫描已启动"}


@router.get("/admin/api/scanner/auto-progress")
async def scanner_auto_progress(session=Depends(require_admin)):
    return scanner_service.get_auto_progress()


@router.post("/admin/api/scanner/stop")
async def scanner_stop(session=Depends(require_admin)):
    ok = scanner_service.stop_scan()
    return {"success": ok, "message": "扫描已停止" if ok else "当前没有扫描任务"}


@router.post("/admin/api/backends/{key:path}/test")
async def test_backend(key: str, session=Depends(require_admin)):
    b = backend_manager.get_backend_by_key(key)
    if not b:
        return {"success": False, "error": "后端不存在"}
    url = f"{b.base_url}/api/tags"
    req_headers = {}
    if b.api_key:
        req_headers["Authorization"] = f"Bearer {b.api_key}"
    timeout = aiohttp.ClientTimeout(total=5, connect=3)
    try:
        t0 = time.time()
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url, headers=req_headers) as resp:
                latency = round((time.time() - t0) * 1000)
                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}", "latency_ms": latency}
                data = await resp.json()
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                target_ok = [m for m in TARGET_MODELS if m in models]
                return {"success": True, "latency_ms": latency, "models": models, "target_models": target_ok}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/admin/api/scanner/report")
async def scanner_report(session=Depends(require_admin)):
    return scanner_service.get_system_report()
