"""供应商统一管理（provider_hub）。

问题背景
--------
AstrBot 的模型供应商引用是分散的：
  · 核心 ``cmd_config.json``：主对话、回退对话模型、上下文压缩、图像理解、语音、嵌入、重排序
  · 会话配置档 ``config/abconf_*.json``：同上，且覆盖全局
  · 各插件自己的 ``config/astrbot_plugin_*_config.json``：创作、记忆、情感、检索……

用 Agent（或手改文件）改完主配置后，插件里仍指着旧供应商，表现为日志里
「没配置过的供应商在跑」。本插件把全部引用点扫出来摊到一页上，改完一键写回。

分类口径（与本体「模型提供商」面板完全一致）
------------------------------------------
五类能力取自 ``astrbot.core.provider.entities.ProviderType``：

    chat_completion  对话
    speech_to_text   语音转文字
    text_to_speech   文字转语音
    embedding        嵌入
    rerank           重排序

判类规则直接照抄本体 ``dashboard/services/config_service.py::list_providers``：

    1. 条目自己写了 ``provider_type`` → 用条目声明的；
    2. 否则挂在某个 ``provider_sources`` 下 → 继承该来源的 ``provider_type``；
    3. 都没有 → 视为「对话」。

**不做任何字段名/适配器名猜测**：手写启发式会与本体面板出现分类分歧，这里
宁可少分类也不自作主张。``provider_type == "agent_runner"`` 的条目本体面板
不列出，本插件同样跳过。每个引用点的下拉框只列出**同类型**供应商，避免把
embedding 填进对话位。

``api_state`` 还会顺带用本体接口 ``/api/v1/providers`` 做一次对账：若磁盘配置里
的供应商与核心内存里的列表有出入（例如手改过文件却没重载），面板会提示数量
差异，避免"配置里改了、实际跑的还是旧的"。

实现要点
--------
* 核心配置 **不直接写文件**：走 dashboard 的 ``/api/config/astrbot/update``，
  服务端保存后会调用 ``reload_pipeline_scheduler()``，流水线当场重建。
* 插件配置精确改 JSON 字段后，调用 ``/api/plugin/reload`` 让该插件重新加载。
* 每次同步前把涉及的文件备份到 ``data/config_backups/provider_hub-<时间>/``，
  同步历史追加到 ``data/plugin_data/provider_hub/history.jsonl``。
* 所有持久化数据都在 ``data`` 目录下，插件自身目录只放代码与静态页面，
  因此更新/重装插件不会丢数据。
* 网络请求统一用 httpx（异步），不使用 requests。
* 无常驻任务、无定时器：只有点击「一键同步」时才执行。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import httpx
from quart import request

from astrbot.api import logger
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "provider_hub"
PLUGIN_VERSION = "1.4.0"

# ---------------------------------------------------------------- 目录约定
# main.py 位于 <data>/plugins/astrbot_plugin_provider_hub/，parents[2] 即 data 目录。
DATA_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = DATA_DIR / "config"
MAIN_CONFIG = DATA_DIR / "cmd_config.json"
# 备份与历史都落在 data 下，重装插件不会影响。
BACKUP_DIR = DATA_DIR / "config_backups"
PLUGIN_DATA_DIR = DATA_DIR / "plugin_data" / PLUGIN_NAME
HISTORY_FILE = PLUGIN_DATA_DIR / "history.jsonl"
HISTORY_LIMIT = 50
# 记住上次连通的管理端口（桌面端会随机化端口，环境变量缺失时靠它兜底）。
PORT_HINT_FILE = PLUGIN_DATA_DIR / "dashboard_port.txt"

# ---------------------------------------------------------------- 分类口径
# 键与 astrbot.core.provider.entities.ProviderType 的取值完全一致。
PROVIDER_TYPE_LABELS: dict[str, str] = {
    "chat_completion": "对话",
    "speech_to_text": "语音转文字",
    "text_to_speech": "文字转语音",
    "embedding": "嵌入",
    "rerank": "重排序",
}
PROVIDER_TYPE_ORDER = [
    "chat_completion",
    "speech_to_text",
    "text_to_speech",
    "embedding",
    "rerank",
]
DEFAULT_PROVIDER_TYPE = "chat_completion"
# 本体面板不展示这类条目（如 Dify / Coze 等 Agent 执行器），本插件同样跳过。
_SKIPPED_PROVIDER_TYPES = ("agent_runner",)

# 引用点「用途」标签：面板上按这个分组显示，与「供应商类型」是两个维度。
CATEGORY_LABELS: dict[str, str] = {
    "main": "主对话",
    "fallback": "回退对话模型",
    "compression": "上下文压缩",
    "vision": "图像理解",
    "plugin": "插件功能",
    "embedding": "向量嵌入",
    "rerank": "重排序",
    "stt": "语音转文字",
    "tts": "语音合成",
    "tts_voice": "音色表",
    "other": "其他",
}

# 只用于展示的字段：其值恰好等于某个供应商 id 时不算引用（如音色表里的 name）。
_DISPLAY_FIELD_SUFFIXES = (
    ".name",
    ".label",
    ".title",
    ".desc",
    ".description",
    ".remark",
    ".comment",
    ".note",
    ".id",
)

# 引用点字段路径 → 需要的供应商类型。命中即锁定下拉候选池。
_PATH_TYPE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("provider_stt_settings", "speech_to_text", "whisper", "stt_provider"),
        "speech_to_text",
    ),
    (
        ("provider_tts_settings", "text_to_speech", "tts_provider", "tts_conversion"),
        "text_to_speech",
    ),
    (("embedding_provider", "embedding_model"), "embedding"),
    (("rerank_provider", "rerank_model"), "rerank"),
    # 回退对话模型是一个有序数组，元素一定是对话类供应商。
    (("fallback_provider",), "chat_completion"),
)

_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


# ------------------------------------------------------------------ 通用工具


def _read_json(path: Path) -> dict:
    """读取 JSON，兼容带 BOM 的文件（核心配置就是 UTF-8 BOM）。"""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, data: Any) -> None:
    """原子写 JSON：先写临时文件再替换，避免写一半断电导致配置损坏。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def _split_path(path: str) -> list[Any]:
    """把 ``a.b[0].c`` 拆成 ``["a", "b", 0, "c"]``。"""
    parts: list[Any] = []
    for match in _TOKEN_RE.finditer(path):
        if match.group(1) is not None:
            parts.append(match.group(1))
        else:
            parts.append(int(match.group(2)))
    return parts


def _set_path(obj: Any, path: str, value: Any) -> None:
    """按路径写入嵌套结构，路径不存在时抛异常（由调用方捕获并计入失败项）。"""
    parts = _split_path(path)
    if not parts:
        raise ValueError(f"非法字段路径: {path}")
    cursor = obj
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


# ------------------------------------------------------------------ 鉴权与接口


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _jwt(secret: str, username: str = "astrbot", ttl: int = 900) -> str:
    """构造 dashboard 用的 HS256 JWT（payload 只需 username 与 exp）。"""
    header = _b64(
        json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode()
    )
    payload = _b64(
        json.dumps(
            {"username": username, "exp": int(time.time()) + ttl}, separators=(",", ":")
        ).encode()
    )
    signing_input = f"{header}.{payload}".encode()
    signature = _b64(hmac.new(secret.encode(), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


class DashboardError(RuntimeError):
    """dashboard 接口调用失败。"""


def _remember_port(port: int) -> None:
    """记下这次连通的端口，下次环境变量缺失时还能找到核心（失败不影响主流程）。"""
    try:
        PLUGIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PORT_HINT_FILE.write_text(str(int(port)), encoding="utf-8")
    except OSError:
        pass


def _last_port() -> str:
    try:
        text = PORT_HINT_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text if text.isdigit() else ""


class _Dashboard:
    """极简 dashboard 客户端：只做本机调用，跳过系统代理。"""

    def __init__(self) -> None:
        cfg = _read_json(MAIN_CONFIG)
        dashboard_cfg = cfg.get("dashboard") or {}
        self.secret = str(dashboard_cfg.get("jwt_secret") or "")
        if not self.secret:
            raise DashboardError(
                "核心配置里没有 dashboard.jwt_secret，无法调用管理接口"
            )
        # 桌面端启动器会注入随机的 DASHBOARD_PORT：环境变量优先，其次配置文件里
        # 写的端口，最后是上次成功连通的端口。逐个试，谁通用谁。
        candidates = [
            os.environ.get("DASHBOARD_PORT"),
            dashboard_cfg.get("port"),
            _last_port(),
        ]
        self.ports: list[int] = []
        for value in candidates:
            try:
                port = int(str(value).strip())
            except (TypeError, ValueError):
                continue
            if port not in self.ports:
                self.ports.append(port)

    async def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + _jwt(self.secret),
        }
        last_error: Exception | None = None
        for port in self.ports:
            base = f"http://127.0.0.1:{port}"
            try:
                async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
                    resp = await client.request(
                        method, base + path, headers=headers, json=body
                    )
            except httpx.HTTPError as exc:
                # 端口不对时连接会被拒绝：换下一个候选端口继续试。
                last_error = exc
                continue

            _remember_port(port)
            if resp.status_code >= 400:
                raise DashboardError(
                    f"{path} 返回 HTTP {resp.status_code}: {resp.text[:200]}"
                )
            try:
                return resp.json() if resp.content else {}
            except ValueError as exc:
                raise DashboardError(f"{path} 返回的不是 JSON") from exc

        tried = "、".join(str(port) for port in self.ports) or "无可用端口"
        raise DashboardError(f"管理接口连不上（试过端口 {tried}）: {last_error}")

    async def get_conf(self, conf_id: str) -> dict:
        res = await self._call("GET", f"/api/config/abconf?id={conf_id}")
        return (res.get("data") or {}).get("config") or {}

    async def save_conf(self, conf_id: str, config: dict) -> str:
        """保存核心配置；服务端保存后会自动重建流水线，无需重启。"""
        res = await self._call(
            "POST", "/api/config/astrbot/update", {"config": config, "conf_id": conf_id}
        )
        if res.get("status") == "error":
            raise DashboardError(str(res.get("message") or "保存失败")[:300])
        return str(res.get("message") or "saved")

    async def reload_plugin(self, plugin_id: str) -> None:
        """让单个插件重新加载，使其读到刚写入的配置。"""
        # legacy 路由是单数 /api/plugin/reload；v1 为 /api/v1/plugins/reload。
        await self._call("POST", "/api/plugin/reload", {"name": plugin_id})

    async def list_profiles(self) -> list[dict]:
        """取配置档列表（含用户在 WebUI 里起的名字）。

        只有 ``/api/v1/config-profiles`` 会带名字，legacy 的 ``/api/config/abconfs``
        只给 id；拿不到名字时前端会回退显示 id 前缀，不影响同步。
        """
        res = await self._call("GET", "/api/v1/config-profiles")
        payload = res.get("data")
        items = payload.get("info_list") if isinstance(payload, dict) else None
        profiles: list[dict] = []
        for item in items or []:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            profiles.append(
                {"id": str(item["id"]), "name": str(item.get("name") or "")}
            )
        return profiles

    async def provider_ids(self) -> set[str]:
        """取核心内存里的供应商 id 集合，用于核对磁盘配置是否已生效。"""
        res = await self._call("GET", "/api/v1/providers")
        payload = res.get("data")
        items = (payload or {}).get("providers") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise DashboardError("/api/v1/providers 返回结构异常")
        return {
            str(item.get("id"))
            for item in items
            if isinstance(item, dict) and item.get("id")
        }


# ------------------------------------------------------------------ 供应商分类


def _source_types(cfg: dict) -> dict[str, str]:
    """``provider_sources`` 里每个来源声明的能力类型（本体就是这么取继承值的）。"""
    result: dict[str, str] = {}
    for source in cfg.get("provider_sources") or []:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id") or "").strip()
        if not source_id:
            continue
        declared = str(source.get("provider_type") or "").strip()
        result[source_id] = (
            declared if declared in PROVIDER_TYPE_LABELS else DEFAULT_PROVIDER_TYPE
        )
    return result


def _provider_type(entry: dict, source_types: dict[str, str]) -> tuple[str, str]:
    """判定供应商能力类型，返回 ``(类型, 判定来源)``。

    规则与本体 dashboard 的供应商面板完全一致（``config_service.list_providers``）：
    条目声明 → 所属来源声明 → 兜底「对话」。判定来源只用于面板展示
    （entry = 条目自己写的，source = 继承自 provider_sources，default = 兜底）。
    """
    declared = str(entry.get("provider_type") or "").strip()
    if declared in PROVIDER_TYPE_LABELS:
        return declared, "entry"

    source_id = str(entry.get("provider_source_id") or "").strip()
    if source_id and source_id in source_types:
        return source_types[source_id], "source"

    return DEFAULT_PROVIDER_TYPE, "default"


def _provider_model(entry: dict, provider_type: str) -> str:
    """按类型挑出模型名，用于面板展示。"""
    candidates = {
        "embedding": ("embedding_model", "model"),
        "rerank": ("rerank_model", "model"),
        "speech_to_text": ("stt_model", "model"),
        "text_to_speech": ("tts_model", "model"),
    }.get(provider_type, ("model",))
    for field in candidates:
        value = entry.get(field)
        if value:
            return str(value)
    return ""


def _load_providers() -> tuple[list[dict], dict]:
    """读出全部供应商条目并按本体口径分类；坏条目跳过而不是整体失败。

    返回 ``(供应商列表, 附加信息)``；附加信息里带上来源类型表与跳过的条目数，
    面板据此说明"为什么数量比配置文件里少"。
    """
    cfg = _read_json(MAIN_CONFIG)
    source_types = _source_types(cfg)
    providers: list[dict] = []
    skipped = 0
    for entry in cfg.get("provider") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("provider_type") or "") in _SKIPPED_PROVIDER_TYPES:
            skipped += 1
            continue
        provider_id = str(entry.get("id") or "").strip()
        if not provider_id:
            continue
        provider_type, type_from = _provider_type(entry, source_types)
        source_id = str(entry.get("provider_source_id") or "")
        providers.append(
            {
                "id": provider_id,
                "provider_type": provider_type,
                "provider_type_label": PROVIDER_TYPE_LABELS[provider_type],
                # 类型是条目自己声明的，还是从来源继承的（面板上给出提示）。
                "type_from": type_from,
                "model": _provider_model(entry, provider_type),
                "source": source_id or str(entry.get("provider") or ""),
                "enable": bool(entry.get("enable", True)),
            }
        )
    return providers, {
        "source_types": source_types,
        "skipped_agent_runner": skipped,
        "raw_total": len(cfg.get("provider") or []),
    }


def _group_providers(providers: list[dict]) -> list[dict]:
    """按五类能力分组，空类也保留（面板需要显示「这一类还没有供应商」）。"""
    buckets: dict[str, list[dict]] = {key: [] for key in PROVIDER_TYPE_ORDER}
    for provider in providers:
        buckets.setdefault(provider["provider_type"], []).append(provider)
    return [
        {
            "provider_type": key,
            "label": PROVIDER_TYPE_LABELS[key],
            "count": len(buckets.get(key, [])),
            "items": buckets.get(key, []),
        }
        for key in PROVIDER_TYPE_ORDER
    ]


# ------------------------------------------------------------------ 引用扫描


def _expected_provider_type(field_path: str, fallback: str) -> str:
    """由字段路径推断该位置需要哪类供应商，推断不出时沿用当前值的类型。"""
    lowered = field_path.lower()
    for keywords, provider_type in _PATH_TYPE_HINTS:
        if any(keyword in lowered for keyword in keywords):
            return provider_type
    return fallback


def _categorize(field_path: str) -> str:
    """判断引用点的用途，仅用于面板分组展示。

    音色表（``voices[N].provider_id``）单列一类：它天然每个角色不同，
    面板默认隐藏，避免淹没真正需要统一的少量引用。
    """
    lowered = field_path.lower()
    if "voices[" in lowered or "voice_map" in lowered or "voice_mapping" in lowered:
        return "tts_voice"
    if "embedding" in lowered:
        return "embedding"
    if "rerank" in lowered:
        return "rerank"
    if "stt" in lowered or "speech_to_text" in lowered or "whisper" in lowered:
        return "stt"
    if "tts" in lowered or "voice" in lowered:
        return "tts"
    if "compression" in lowered:
        return "compression"
    if "caption" in lowered or "vision" in lowered:
        return "vision"
    if "fallback" in lowered:
        return "fallback"
    if "agent_runner" in lowered and "model" in lowered:
        return "main"
    if "provider_id" in lowered or "provider_ids" in lowered:
        return "plugin"
    return "other"


def _collect_config_files() -> list[dict]:
    """列出要扫描的配置文件：全局、每个会话配置档、每个插件配置。"""
    items: list[dict] = []
    if MAIN_CONFIG.exists():
        items.append(
            {
                "file": "cmd_config.json",
                "label": "全局配置",
                "conf_id": "default",
                "plugin_id": "",
            }
        )
    if not CONFIG_DIR.is_dir():
        return items

    for path in sorted(CONFIG_DIR.glob("abconf_*.json")):
        conf_id = path.stem.removeprefix("abconf_")
        items.append(
            {
                "file": f"config/{path.name}",
                "label": f"配置档 {conf_id[:8]}",
                "conf_id": conf_id,
                "plugin_id": "",
            }
        )
    for path in sorted(CONFIG_DIR.glob("astrbot_plugin_*_config.json")):
        plugin_id = path.stem.removeprefix("astrbot_plugin_").removesuffix("_config")
        items.append(
            {
                "file": f"config/{path.name}",
                "label": f"插件 {plugin_id}",
                "conf_id": "",
                "plugin_id": plugin_id,
            }
        )
    return items


def _build_profiles(refs: list[dict], live: list[dict]) -> list[dict]:
    """按配置文件汇总：默认配置 + 每个会话配置档，附带名称与引用数。

    ``live`` 来自 ``/api/v1/config-profiles``（id + 用户起的名字）。名字拿不到时
    回退成 id 前 8 位，至少和 WebUI 的配置文件下拉能对上号。
    """
    names = {
        str(item.get("id")): str(item.get("name") or "")
        for item in live
        if isinstance(item, dict) and item.get("id")
    }
    counts: dict[str, int] = {}
    files: dict[str, str] = {}
    for ref in refs:
        conf_id = ref["conf_id"]
        if not conf_id:
            continue
        counts[conf_id] = counts.get(conf_id, 0) + 1
        files.setdefault(conf_id, ref["file"])

    profiles: list[dict] = []
    for conf_id, file in files.items():
        raw_name = names.get(conf_id, "")
        if conf_id == "default":
            label = "默认配置文件"
            if raw_name and raw_name != "default":
                label = f"默认配置文件（{raw_name}）"
        else:
            label = raw_name or f"配置档 {conf_id[:8]}"
        profiles.append(
            {
                "id": conf_id,
                "name": raw_name,
                "label": label,
                "file": file,
                "is_default": conf_id == "default",
                "ref_count": counts[conf_id],
            }
        )
    profiles.sort(key=lambda item: (not item["is_default"], item["label"]))
    return profiles


def _collect_id_fields(
    obj: Any,
    path: str,
    known: dict[str, dict],
    scalars: list[tuple[str, str]],
    arrays: list[tuple[str, list[str]]],
) -> None:
    """递归收集引用：标量字段进 ``scalars``，全是已知供应商 id 的数组进 ``arrays``。

    数组整体处理而不是拆成 ``[0]``/``[1]``：``fallback_provider_ids`` 这类字段
    语义上就是一个可多选的列表（本体面板也是多选），拆开会让人以为是两个独立配置。
    只有当数组里**全是**已知供应商 id 时才整体收集；含未知元素时退回逐项处理，
    避免整体回写时把认不出的元素冲掉。
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            _collect_id_fields(value, child, known, scalars, arrays)
    elif isinstance(obj, list):
        strings = [item for item in obj if isinstance(item, str)]
        is_id_list = (
            bool(strings)
            and len(strings) == len(obj)
            and any(item in known for item in strings)
        )
        if is_id_list and all(item in known for item in strings):
            arrays.append((path, list(obj)))
            return
        for index, value in enumerate(obj):
            _collect_id_fields(value, f"{path}[{index}]", known, scalars, arrays)
    elif isinstance(obj, str) and obj in known:
        scalars.append((path, obj))


def _is_skipped_field(field_path: str) -> bool:
    """判断字段是否应当忽略：供应商自身定义、以及只用于展示的字段。"""
    if field_path.startswith(("provider[", "provider_sources[")):
        return True
    return field_path.lower().endswith(_DISPLAY_FIELD_SUFFIXES)


def _build_ref(
    meta: dict,
    field_path: str,
    values: list[str],
    multi: bool,
    known: dict[str, dict],
) -> dict | None:
    """把一处引用点整理成前端要的结构；该字段属于忽略范围时返回 None。"""
    if _is_skipped_field(field_path):
        return None
    category = _categorize(field_path)
    current_type = known[values[0]]["provider_type"]
    expected_type = _expected_provider_type(field_path, current_type)
    # 多值字段里每个元素都可能是不同类，任一不符即标红。
    mismatch = any(known[value]["provider_type"] != expected_type for value in values)
    return {
        "key": f"{meta['file']}::{field_path}",
        "file": meta["file"],
        "file_label": meta["label"],
        "conf_id": meta["conf_id"],
        "plugin_id": meta["plugin_id"],
        "path": field_path,
        # multi=True 表示这个字段是一个 id 数组，界面上可以多选。
        "multi": multi,
        "values": values,
        "value": ", ".join(values),
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "provider_type": expected_type,
        "provider_type_label": PROVIDER_TYPE_LABELS.get(expected_type, expected_type),
        "current_provider_type": current_type,
        "type_mismatch": mismatch,
    }


def _scan_refs(providers: list[dict]) -> list[dict]:
    """扫描所有配置文件里指向已知供应商 id 的字段（含数组型多值字段）。"""
    known = {item["id"]: item for item in providers}
    refs: list[dict] = []
    for meta in _collect_config_files():
        path = DATA_DIR / meta["file"]
        if not path.exists():
            continue
        try:
            data = _read_json(path)
        except (OSError, ValueError) as exc:
            logger.warning("[provider_hub] 跳过无法解析的配置 %s: %s", path.name, exc)
            continue

        scalars: list[tuple[str, str]] = []
        arrays: list[tuple[str, list[str]]] = []
        _collect_id_fields(data, "", known, scalars, arrays)

        for field_path, value in scalars:
            ref = _build_ref(meta, field_path, [value], False, known)
            if ref is not None:
                refs.append(ref)
        for field_path, values in arrays:
            ref = _build_ref(meta, field_path, values, True, known)
            if ref is not None:
                refs.append(ref)
    refs.sort(
        key=lambda item: (
            PROVIDER_TYPE_ORDER.index(item["provider_type"])
            if item["provider_type"] in PROVIDER_TYPE_ORDER
            else 99,
            item["file"],
            item["path"],
        )
    )
    return refs


# ------------------------------------------------------------------ 备份与历史


def _backup(files: list[str]) -> str:
    """把待改文件整份复制到 data/config_backups/provider_hub-<时间>/。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = BACKUP_DIR / f"provider_hub-{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    for rel in files:
        src = DATA_DIR / rel
        if src.exists():
            try:
                shutil.copy2(src, target / Path(rel).name)
            except OSError as exc:
                logger.warning("[provider_hub] 备份 %s 失败: %s", rel, exc)
    return str(target)


def _append_history(record: dict) -> None:
    """追加一条同步历史，只保留最近 HISTORY_LIMIT 条；失败不影响主流程。"""
    try:
        PLUGIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        if HISTORY_FILE.exists():
            lines = [
                line
                for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        lines.append(json.dumps(record, ensure_ascii=False))
        HISTORY_FILE.write_text(
            "\n".join(lines[-HISTORY_LIMIT:]) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("[provider_hub] 写入同步历史失败: %s", exc)


# ------------------------------------------------------------------ 与本体对账


async def _check_core(providers: list[dict]) -> dict:
    """把磁盘上的供应商列表与核心内存里的列表核对一遍。

    核心读的是保存那一刻的内存配置；若有人手改过 ``cmd_config.json`` 却没让
    核心重载，两边就会不一致——此时面板提示差异，而不是让你对着旧列表改。
    对账失败（例如接口不可用）不算错误，只是标记为「未核对」。
    """
    disk_ids = {item["id"] for item in providers}
    try:
        core_ids = await _Dashboard().provider_ids()
    except DashboardError as exc:
        logger.warning("[provider_hub] 与核心对账失败: %s", exc)
        return {"checked": False, "error": str(exc)[:200], "disk_total": len(disk_ids)}

    return {
        "checked": True,
        "error": "",
        "disk_total": len(disk_ids),
        "core_total": len(core_ids),
        # 只在核心里有 = 磁盘没写进去；只在磁盘上有 = 核心还没重载。
        "only_in_core": sorted(core_ids - disk_ids)[:20],
        "only_on_disk": sorted(disk_ids - core_ids)[:20],
    }


# ------------------------------------------------------------------ 插件主体


@register(PLUGIN_NAME, "xuanbaon", "供应商引用统一管理与同步", PLUGIN_VERSION)
class ProviderHubPlugin(Star):
    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context, config)
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/state", self.api_state, ["GET"], "扫描供应商与引用点"
        )
        self.context.register_web_api(
            f"/{PLUGIN_NAME}/apply", self.api_apply, ["POST"], "批量写回供应商引用"
        )

    async def api_state(self, **_kwargs):
        """返回供应商池（按五类分组）、全部引用点，以及与核心内存的对账结果。"""
        try:
            providers, meta = _load_providers()
            refs = _scan_refs(providers)
        except (OSError, ValueError) as exc:
            logger.error("[provider_hub] 扫描失败: %s", exc, exc_info=True)
            return {"status": "error", "message": f"扫描失败: {exc}", "data": {}}

        by_category: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for ref in refs:
            by_category[ref["category"]] = by_category.get(ref["category"], 0) + 1
            by_type[ref["provider_type"]] = by_type.get(ref["provider_type"], 0) + 1

        # 配置档名字只有新接口带；拿不到就回退成 id 前缀，不影响其它功能。
        live_profiles: list[dict] = []
        try:
            live_profiles = await _Dashboard().list_profiles()
        except DashboardError as exc:
            logger.warning("[provider_hub] 读取配置档列表失败: %s", exc)

        profiles = _build_profiles(refs, live_profiles)
        # 分组标题也用配置档的名字（原来的 file_label 只有 id 前缀，看不出是哪个配置）
        label_by_conf = {item["id"]: item["label"] for item in profiles}
        for ref in refs:
            label = label_by_conf.get(ref["conf_id"])
            if label:
                ref["file_label"] = label
        plugin_refs = sum(1 for ref in refs if not ref["conf_id"])

        return {
            "status": "ok",
            "data": {
                "providers": providers,
                "provider_groups": _group_providers(providers),
                "provider_type_order": PROVIDER_TYPE_ORDER,
                "provider_type_labels": PROVIDER_TYPE_LABELS,
                "profiles": profiles,
                "refs": refs,
                "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "core_check": await _check_core(providers),
                "summary": {
                    "total": len(refs),
                    "multi_total": sum(1 for ref in refs if ref["multi"]),
                    "provider_total": len(providers),
                    "profile_total": len(profiles),
                    "plugin_refs": plugin_refs,
                    "mismatch": sum(1 for ref in refs if ref["type_mismatch"]),
                    "by_category": by_category,
                    "by_provider_type": by_type,
                    "skipped_agent_runner": meta["skipped_agent_runner"],
                    "raw_total": meta["raw_total"],
                },
            },
        }

    async def api_apply(self, **_kwargs):
        """把面板上的改动写回配置：核心走保存接口，插件走精确改 JSON + 重载。"""
        payload = await request.get_json(silent=True) or {}
        changes = payload.get("changes") or []
        if not isinstance(changes, list) or not changes:
            return {"status": "error", "message": "没有需要同步的改动", "data": {}}

        try:
            providers, _meta = _load_providers()
        except (OSError, ValueError) as exc:
            logger.error("[provider_hub] 读取供应商失败: %s", exc, exc_info=True)
            return {"status": "error", "message": f"读取供应商失败: {exc}", "data": {}}
        valid_ids = {item["id"] for item in providers}
        type_of = {item["id"]: item["provider_type"] for item in providers}

        grouped: dict[tuple[str, str], list[dict]] = {}
        rejected: list[dict] = []
        for item in changes:
            if not isinstance(item, dict):
                rejected.append({"file": "", "path": "", "reason": "改动项格式非法"})
                continue
            file = str(item.get("file") or "")
            field_path = str(item.get("path") or "")
            conf_id = str(item.get("conf_id") or "")
            plugin_id = str(item.get("plugin_id") or "")
            expected_type = str(item.get("provider_type") or "")

            # 多值字段（如回退对话模型）用 values 数组提交，单值字段用 value。
            raw_values = item.get("values")
            multi = isinstance(raw_values, list)
            if multi:
                values = []
                for raw in raw_values:
                    text = str(raw or "").strip()
                    if text and text not in values:
                        values.append(text)
            else:
                text = str(item.get("value") or "").strip()
                values = [text] if text else []

            if not file or not field_path:
                rejected.append(
                    {"file": file, "path": field_path, "reason": "缺少文件名或字段路径"}
                )
                continue
            if not values and not multi:
                rejected.append(
                    {"file": file, "path": field_path, "reason": "没有选择供应商"}
                )
                continue
            unknown = [value for value in values if value not in valid_ids]
            if unknown:
                rejected.append(
                    {
                        "file": file,
                        "path": field_path,
                        "reason": "未知供应商: " + "、".join(unknown),
                    }
                )
                continue
            # 类型守卫：禁止把嵌入模型填进对话位这类跨类误配。
            mismatch = [
                value
                for value in values
                if expected_type
                and expected_type in PROVIDER_TYPE_LABELS
                and type_of[value] != expected_type
            ]
            if mismatch:
                value = mismatch[0]
                rejected.append(
                    {
                        "file": file,
                        "path": field_path,
                        "reason": (
                            f"类型不匹配：该位置需要{PROVIDER_TYPE_LABELS[expected_type]}类，"
                            f"而 {value} 是{PROVIDER_TYPE_LABELS[type_of[value]]}类"
                        ),
                    }
                )
                continue
            if not (DATA_DIR / file).exists():
                rejected.append(
                    {"file": file, "path": field_path, "reason": "配置文件不存在"}
                )
                continue
            grouped.setdefault((conf_id, plugin_id), []).append(
                {
                    "file": file,
                    "path": field_path,
                    # 多值字段整体回写数组，单值字段写字符串。
                    "value": values if multi else values[0],
                    "multi": multi,
                }
            )

        if not grouped:
            return {
                "status": "error",
                "message": "全部改动被拒绝",
                "data": {"rejected": rejected},
            }

        backup_dir = _backup(
            sorted({entry["file"] for items in grouped.values() for entry in items})
        )
        try:
            dashboard = _Dashboard()
        except DashboardError as exc:
            return {
                "status": "error",
                "message": str(exc),
                "data": {"rejected": rejected},
            }

        applied: list[dict] = []
        failed: list[dict] = []

        for (conf_id, plugin_id), items in grouped.items():
            # ---- 核心配置（全局 / 会话配置档）：走官方保存接口，保存后自动热重载
            if conf_id:
                try:
                    config = await dashboard.get_conf(conf_id)
                    for entry in items:
                        _set_path(config, entry["path"], entry["value"])
                    message = await dashboard.save_conf(conf_id, config)
                    for entry in items:
                        applied.append({**entry, "scope": conf_id, "result": message})
                except (DashboardError, KeyError, IndexError, TypeError) as exc:
                    logger.error(
                        "[provider_hub] 保存 %s 失败: %s", conf_id, exc, exc_info=True
                    )
                    for entry in items:
                        failed.append(
                            {**entry, "scope": conf_id, "error": str(exc)[:300]}
                        )
                continue

            # ---- 插件配置：精确改 JSON 字段（不走表单，避免 schema 默认值回填）
            touched: set[str] = set()
            for entry in items:
                target = DATA_DIR / entry["file"]
                try:
                    config = _read_json(target)
                    _set_path(config, entry["path"], entry["value"])
                    _write_json(target, config)
                    touched.add(entry["file"])
                    applied.append({**entry, "scope": plugin_id, "result": "written"})
                except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                    logger.error(
                        "[provider_hub] 写入 %s 失败: %s",
                        entry["file"],
                        exc,
                        exc_info=True,
                    )
                    failed.append(
                        {**entry, "scope": plugin_id, "error": str(exc)[:300]}
                    )
            if touched and plugin_id:
                try:
                    await dashboard.reload_plugin(plugin_id)
                except DashboardError as exc:
                    # 写盘已成功，只是重载失败：提示但不计为失败项。
                    logger.warning(
                        "[provider_hub] 重载插件 %s 失败: %s", plugin_id, exc
                    )

        _append_history(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "applied": len(applied),
                "failed": len(failed),
                "rejected": len(rejected),
                "backup": backup_dir,
                "changes": applied,
            }
        )

        return {
            "status": "ok" if not failed else "partial",
            "message": f"已同步 {len(applied)} 处，失败 {len(failed)} 处，拒绝 {len(rejected)} 处",
            "data": {
                "applied": applied,
                "failed": failed,
                "rejected": rejected,
                "backup": backup_dir,
            },
        }
