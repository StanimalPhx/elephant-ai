"""JSON API endpoints for trace inspection."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from elephant.database import DatabaseInstance
    from elephant.router import ChatRouter

router_key: web.AppKey[ChatRouter] = web.AppKey("chat_router", object)  # type: ignore[arg-type]


def _get_router(request: web.Request) -> ChatRouter:
    return request.app[router_key]


def _find_db(request: web.Request) -> DatabaseInstance | None:
    db_name = request.match_info["db_name"]
    for d in _get_router(request).get_all_databases():
        if d.name == db_name:
            return d
    return None


# ---------------------------------------------------------------------------
# GET /api/traces/databases
# ---------------------------------------------------------------------------

async def databases_handler(request: web.Request) -> web.Response:
    """Return list of database names."""
    router = _get_router(request)
    names = [db.name for db in router.get_all_databases()]
    return web.json_response({"databases": [{"name": n} for n in names]})


# ---------------------------------------------------------------------------
# GET /api/traces/{db_name}?page=0&per_page=30
# ---------------------------------------------------------------------------

def _trace_summary(trace_data: dict[str, Any]) -> dict[str, Any]:
    """Project a trace dict into a lightweight summary."""
    steps = trace_data.get("steps", [])
    counts: dict[str, int] = {}
    for s in steps:
        st = s.get("step_type", "unknown")
        counts[st] = counts.get(st, 0) + 1

    text = trace_data.get("message_text", "")
    return {
        "trace_id": trace_data.get("trace_id", ""),
        "started_at": trace_data.get("started_at", ""),
        "intent": trace_data.get("intent", ""),
        "message_text": text[:120] + ("..." if len(text) > 120 else ""),
        "sender": trace_data.get("sender", ""),
        "step_counts": counts,
        "has_error": trace_data.get("error") is not None,
    }


async def traces_list_handler(request: web.Request) -> web.Response:
    """Return paginated trace summaries for a database."""
    db = _find_db(request)
    if db is None:
        db_name = request.match_info["db_name"]
        return web.json_response({"error": f"unknown database: {db_name}"}, status=404)

    page = int(request.query.get("page", "0"))
    per_page = int(request.query.get("per_page", "30"))
    offset = page * per_page

    traces, total = db.store.read_traces(limit=per_page, offset=offset)
    summaries = [_trace_summary(t.model_dump(mode="json")) for t in traces]
    return web.json_response({"traces": summaries, "total": total})


# ---------------------------------------------------------------------------
# GET /api/traces/{db_name}/{trace_id}
# ---------------------------------------------------------------------------

async def trace_detail_handler(request: web.Request) -> web.Response:
    """Return full trace JSON."""
    trace_id = request.match_info["trace_id"]
    db = _find_db(request)
    if db is None:
        db_name = request.match_info["db_name"]
        return web.json_response({"error": f"unknown database: {db_name}"}, status=404)

    trace = db.store.read_trace_by_id(trace_id)
    if trace is None:
        return web.json_response({"error": "trace not found"}, status=404)

    return web.json_response(trace.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# GET /api/git/{db_name}/{sha}?mode=stat|diff
# ---------------------------------------------------------------------------

async def git_show_handler(request: web.Request) -> web.Response:
    """Return git show output for a commit SHA from the database repo."""
    db = _find_db(request)
    if db is None:
        db_name = request.match_info["db_name"]
        return web.json_response(
            {"error": f"unknown database: {db_name}"}, status=404,
        )

    sha = request.match_info["sha"]
    mode = request.query.get("mode", "stat")

    repo_dir = db.git.repo_dir
    if mode == "diff":
        cmd = ["git", "-C", repo_dir, "show", sha]
    else:
        cmd = ["git", "-C", repo_dir, "show", "--stat", sha]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
        return web.json_response({"sha": sha, "output": result.stdout})
    except subprocess.CalledProcessError as exc:
        return web.json_response(
            {"error": f"git show failed: {exc.stderr.strip()}"}, status=404,
        )


# ---------------------------------------------------------------------------
# GET /api/people/{db_name}
# ---------------------------------------------------------------------------

async def people_handler(request: web.Request) -> web.Response:
    """Return all people for a database."""
    from elephant.brain.people_completeness import score_person

    db = _find_db(request)
    if db is None:
        db_name = request.match_info["db_name"]
        return web.json_response({"error": f"unknown database: {db_name}"}, status=404)

    people = db.store.read_all_people()
    result = []
    for p in people:
        data = p.model_dump(mode="json")
        data["completeness_score"] = score_person(p)
        result.append(data)
    return web.json_response({"people": result})


# ---------------------------------------------------------------------------
# GET /api/groups/{db_name}
# ---------------------------------------------------------------------------

async def groups_handler(request: web.Request) -> web.Response:
    """Return all groups for a database."""
    db = _find_db(request)
    if db is None:
        db_name = request.match_info["db_name"]
        return web.json_response({"error": f"unknown database: {db_name}"}, status=404)

    groups = db.store.read_all_groups()
    return web.json_response({"groups": [g.model_dump(mode="json") for g in groups]})


# ---------------------------------------------------------------------------
# GET /api/memories/{db_name}?page=0&per_page=50&person=&type=&year=
# ---------------------------------------------------------------------------


def _memory_summary(memory: object) -> dict[str, Any]:
    """Project a Memory into an API response dict."""
    from elephant.data.models import Memory

    assert isinstance(memory, Memory)
    return {
        "id": memory.id,
        "date": memory.date.isoformat(),
        "title": memory.title,
        "type": memory.type,
        "people": memory.people,
        "description": memory.description,
        "nostalgia_score": memory.nostalgia_score,
        "location": memory.location,
        "source_user": memory.source_user,
    }


async def memories_list_handler(request: web.Request) -> web.Response:
    """Return paginated, filtered memories for a database."""
    from datetime import date

    db = _find_db(request)
    if db is None:
        db_name = request.match_info["db_name"]
        return web.json_response({"error": f"unknown database: {db_name}"}, status=404)

    page = int(request.query.get("page", "0"))
    per_page = int(request.query.get("per_page", "50"))
    person = request.query.get("person", "").strip()
    memory_type = request.query.get("type", "").strip()
    year = request.query.get("year", "").strip()

    date_from = None
    date_to = None
    if year:
        try:
            y = int(year)
            date_from = date(y, 1, 1)
            date_to = date(y, 12, 31)
        except ValueError:
            pass

    people_filter = [person] if person else None
    type_filter = memory_type if memory_type else None

    all_memories = db.store.list_memories(
        date_from=date_from,
        date_to=date_to,
        people=people_filter,
        memory_type=type_filter,
        limit=None,
    )
    total = len(all_memories)
    offset = page * per_page
    page_memories = all_memories[offset : offset + per_page]

    return web.json_response({
        "memories": [_memory_summary(m) for m in page_memories],
        "total": total,
    })


# ---------------------------------------------------------------------------
# SPA catch-all: serve index.html for client-side routing
# ---------------------------------------------------------------------------

async def spa_handler(request: web.Request) -> web.StreamResponse:
    """Serve frontend/dist/index.html for SPA client-side routes."""
    dist_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "dist")
    index_path = os.path.normpath(os.path.join(dist_dir, "index.html"))
    if not os.path.isfile(index_path):
        msg = "Frontend not built. Run: cd frontend && npm run build"
        return web.Response(text=msg, status=404)
    return web.FileResponse(index_path)


def register_routes(app: web.Application, router: ChatRouter) -> None:
    """Register all trace API routes on the app."""
    app[router_key] = router
    app.router.add_get("/api/traces/databases", databases_handler)
    app.router.add_get("/api/traces/{db_name}", traces_list_handler)
    app.router.add_get("/api/traces/{db_name}/{trace_id}", trace_detail_handler)
    app.router.add_get("/api/git/{db_name}/{sha}", git_show_handler)
    app.router.add_get("/api/people/{db_name}", people_handler)
    app.router.add_get("/api/groups/{db_name}", groups_handler)
    app.router.add_get("/api/memories/{db_name}", memories_list_handler)

    # Static assets from the frontend build
    dist_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "dist")
    assets_dir = os.path.normpath(os.path.join(dist_dir, "assets"))
    if os.path.isdir(assets_dir):
        app.router.add_static("/traces/assets/", assets_dir, name="trace_assets")

    # SPA catch-all (must come after static)
    app.router.add_get("/traces/{path:.*}", spa_handler)
    app.router.add_get("/traces", spa_handler)
