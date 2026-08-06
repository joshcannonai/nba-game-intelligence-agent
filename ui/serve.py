"""Local bridge between the hosted UI and the agent running on this machine.

    python -m ui.serve            # http://localhost:8000

WHY THIS EXISTS. The demo site is served from Vercel, but the model is Gemma 4
running in Ollama on this laptop and the datasets are CSVs on this disk. Vercel's
functions run in a cloud container and can reach neither. The browser, however, IS
on this laptop during a screen share, so the page calls localhost directly and this
process does the work. Chrome treats localhost as a trustworthy origin, so an HTTPS
page is allowed to call it without mixed-content blocking.

WHY NOT PORT THE AGENT TO JAVASCRIPT. The date gate lives in `agent/sources.py` and
is the claim the entire project rests on. A second implementation in browser JS
would be a second thing to keep correct, and the first time the two disagreed the
gate would be worthless. So the loop stays in Python and this is a thin wrapper --
it adds no logic of its own, it only forwards.

Nothing here bypasses the gate: every request goes through the same `build_tools`
and the same `CsvSource` the CLI and the eval harness use.
"""

from __future__ import annotations

import json
import pathlib
import queue
import threading
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.run import build_agent
from agent.sources import get_source
from agent.tools import build_tools

app = FastAPI(title="NBA Agent local bridge")

# The deployed origin is not known until `vercel deploy` prints it, and a demo is
# not the place to discover a CORS failure. Any origin may call this: it serves
# read-only basketball data from a laptop, and it is not reachable from outside
# this machine in the first place.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def allow_private_network(request, call_next):
    """Let an HTTPS page on the public internet call this loopback server.

    CORS alone is not enough. Chrome's Private Network Access check treats a
    request from a public HTTPS origin to 127.0.0.1 as a private-network request
    and blocks it unless the preflight comes back with this header. Without it the
    hosted page fails silently -- the fetch never resolves and the status pill sits
    on "bridge down" while the server logs nothing, which is exactly the kind of
    thing you do not want to debug during a screen share.
    """
    if request.method == "OPTIONS":
        from starlette.responses import Response

        response = Response(status_code=204)
    else:
        response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Headers", "*")
    response.headers.setdefault("Access-Control-Allow-Methods", "*")
    return response


class RunRequest(BaseModel):
    matchup_id: str
    as_of_date: str
    include_model: bool = True
    model_backend: str = "ollama"


@app.get("/api/health")
def health() -> dict:
    """What the UI shows in its status pill. Cheap enough to poll."""
    import urllib.error
    import urllib.request

    ollama, models = False, []
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
            ollama = True
    except (urllib.error.URLError, OSError, ValueError):
        pass

    source = get_source("real")
    return {
        "bridge": True,
        "ollama": ollama,
        "models": models,
        "tools": [t.name for t in build_tools(source, include_model=True)],
        "source": source.name,
    }



class PredictRequest(BaseModel):
    matchup_id: str
    as_of_date: str


@app.post("/api/predict")
def predict_only(req: PredictRequest) -> dict:
    """Arm A: the fitted model on its own, no language model in the loop.

    Instant, because scoring a logistic regression is a dot product. Having this
    next to the agent is the point of the comparison: the same game, the same
    as-of date, one answer from a model and one from a model plus an agent.
    """
    from models.notebook_model_output import predict_model_only

    out = predict_model_only(req.matchup_id, req.as_of_date)
    out["arm"] = "A"
    return out


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/run")
def run(req: RunRequest) -> StreamingResponse:
    """Stream the agent's tool calls as they happen, then the final report.

    Streaming is the whole point of the demo: a spinner followed by a JSON blob
    proves nothing, while watching seven named tools get called in order shows the
    thing we actually built. Gemma takes ~40s a game, which is a long time to look
    at a blank panel.
    """

    def generate():
        events: queue.Queue = queue.Queue()

        def work():
            try:
                source = get_source("real")
                agent = build_agent(source, req.model_backend, req.include_model)
                events.put(
                    (
                        "start",
                        {
                            "matchup_id": req.matchup_id,
                            "as_of_date": req.as_of_date,
                            "model": req.model_backend,
                            "arm": "C" if req.include_model else "B",
                        },
                    )
                )
                user = (
                    f"Produce a pregame report for matchup_id={req.matchup_id} "
                    f"as_of_date={req.as_of_date}."
                )
                final = None
                for chunk in agent.stream(
                    {"messages": [{"role": "user", "content": user}]},
                    stream_mode="updates",
                ):
                    for node, update in chunk.items():
                        for message in update.get("messages", []) or []:
                            for call in getattr(message, "tool_calls", []) or []:
                                events.put(
                                    (
                                        "tool_call",
                                        {"name": call["name"], "args": call["args"]},
                                    )
                                )
                            name = getattr(message, "name", None)
                            content = getattr(message, "content", None)
                            if name and content:
                                events.put(
                                    (
                                        "tool_result",
                                        {"name": name, "content": str(content)[:4000]},
                                    )
                                )
                            elif content and not getattr(message, "tool_calls", None):
                                final = content
                if isinstance(final, list):
                    final = "\n".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in final
                    )
                events.put(("final", {"content": str(final or "")}))
            except Exception as exc:  # surfaced in the UI, not swallowed
                events.put(("error", {"message": f"{type(exc).__name__}: {exc}"}))
            finally:
                events.put(("done", {}))

        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        started = time.time()
        while True:
            try:
                event, data = events.get(timeout=1.0)
            except queue.Empty:
                yield _sse("heartbeat", {"elapsed": round(time.time() - started, 1)})
                continue
            data["elapsed"] = round(time.time() - started, 1)
            yield _sse(event, data)
            if event == "done":
                return

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the built site from this process too, so the demo runs same-origin.
#
# The hosted copy on Vercel is the review link: Tools and System are baked in at
# build time and need no backend. But the Agent tab needs THIS machine, and a public
# HTTPS page calling loopback goes through Chrome's private-network check, which we
# watched hang rather than fail. Mounting dist/ here removes that hop from the demo
# path entirely -- localhost:8000 is the same app, same origin, nothing to block.
_DIST = pathlib.Path(__file__).resolve().parent / "web" / "dist"
if _DIST.is_dir():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="site")


def main() -> None:
    import uvicorn

    if not _DIST.is_dir():
        print(
            "note: ui/web/dist missing, API only. Build it with: cd ui/web && bun run build"
        )
    print("demo UI:  http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
