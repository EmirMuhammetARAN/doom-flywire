"""
DOOM-FlyWire Hugging Face Space Entry Point
============================================
ZeroGPU REQUIRES demo.launch() — not uvicorn.run() — to register @spaces.GPU
functions with HF's backend proxy. We satisfy this by calling demo.launch()
while injecting our game routes (WebSocket, assets) into Gradio's internal
FastAPI via a staticmethod patch on App.create_app, which demo.launch() calls
internally. Single server, no double-bind.
"""

# spaces MUST be imported before everything else
import spaces

import os
import asyncio
import json
import time
import threading
import gradio as gr
from gradio.routes import App as GradioApp
from starlette.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

# ── ZeroGPU stub ──────────────────────────────────────────────────────────────
@spaces.GPU
def _gpu_ready():
    import torch
    return f"GPU available: {torch.cuda.is_available()}"


# ── Web assets directory ───────────────────────────────────────────────────────
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
os.makedirs(WEB_DIR, exist_ok=True)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="DOOM-FlyWire Connectome") as demo:
    gr.HTML(
        '<iframe src="/doom" '
        'style="width:100%;height:78vh;border:none;background:#000;" '
        'id="doom-frame"></iframe>'
    )
    with gr.Row():
        gpu_btn = gr.Button("🚀 Test GPU", variant="primary", scale=1)
        gpu_out = gr.Textbox(label="GPU Status", interactive=False, scale=3)
    gpu_btn.click(fn=_gpu_ready, inputs=[], outputs=[gpu_out])
    gr.Markdown(
        "*Full DOOM + brain visualizer is in the iframe above. "
        "If it does not load, [open directly](/doom).*"
    )


# ── Game agent (lazy init — NOT at startup) ───────────────────────────────────
_agent = None
_agent_lock = threading.Lock()
_agent_error = None


def _get_agent():
    global _agent, _agent_error
    with _agent_lock:
        if _agent is not None:
            return _agent
        try:
            import torch
            from src.doom_agent import DoomConnectomeAgent
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Agent] Lazy-init on {device}...")
            _agent = DoomConnectomeAgent(
                scenario_name="defend_the_center.cfg",
                window_visible=False,
                device=device,
            )
            print("[Agent] Ready.")
            return _agent
        except Exception as exc:
            _agent_error = str(exc)
            print(f"[Agent] Init failed: {exc}")
            return None


# ── Patch GradioApp.create_app to inject our routes ───────────────────────────
# demo.launch() calls App.create_app(self) internally to build the FastAPI app.
# By patching this staticmethod BEFORE calling launch(), our game routes are
# baked into the single Gradio/FastAPI server — no second uvicorn needed.

_orig_create_app = GradioApp.__dict__["create_app"]  # staticmethod descriptor


@staticmethod
def _patched_create_app(*args, **kwargs):
    # ── Call the real Gradio create_app ───────────────────────────────────────
    app = _orig_create_app.__func__(*args, **kwargs)

    # ── Custom HTTP routes ────────────────────────────────────────────────────
    def _file(rel, media=""):
        f = os.path.join(WEB_DIR, rel)
        if os.path.exists(f):
            return FileResponse(f, media_type=media) if media else FileResponse(f)
        return JSONResponse({"error": f"{rel} not found"}, status_code=404)

    @app.get("/doom")
    async def doom(): return _file("index.html")

    @app.get("/health")
    async def health(): return JSONResponse({"status": "ok"})

    @app.get("/brain_anatomy.json")
    async def ba(): return _file("brain_anatomy.json", "application/json")

    @app.get("/brain_anatomy_3d.json")
    async def ba3(): return _file("brain_anatomy_3d.json", "application/json")

    @app.get("/brain_139k_pos.bin")
    async def bpos(): return _file("brain_139k_pos.bin", "application/octet-stream")

    @app.get("/brain_139k_col.bin")
    async def bcol(): return _file("brain_139k_col.bin", "application/octet-stream")

    @app.get("/synapses_top5k.bin")
    async def syn(): return _file("synapses_top5k.bin", "application/octet-stream")

    @app.get("/js/three.min.js")
    async def three(): return _file(os.path.join("js", "three.min.js"), "application/javascript")

    @app.get("/js/OrbitControls.js")
    async def orbit(): return _file(os.path.join("js", "OrbitControls.js"), "application/javascript")

    # ── WebSocket game stream ─────────────────────────────────────────────────
    @app.websocket("/ws/game")
    async def ws_game(websocket: WebSocket):
        await websocket.accept()
        print("[WS] Client connected.")

        agent = _get_agent()
        if agent is None:
            await websocket.send_json({"type": "error", "message": _agent_error or "init failed"})
            await websocket.close()
            return

        agent.is_running = True
        fps_base = 20.0
        speed = 1.0
        frame_interval = (1.0 / fps_base) / speed

        try:
            while True:
                t0 = time.perf_counter()
                # Non-blocking receive
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    msg = json.loads(raw)
                    kind = msg.get("type", "")
                    if kind == "update_params":
                        agent.update_sandbox(msg.get("params", {}))
                    elif kind in ("reset", "start"):
                        agent.reset_episode()
                    elif kind == "toggle_pause":
                        agent.toggle_pause()
                    elif kind == "set_speed":
                        speed = max(0.25, min(3.0, float(msg.get("speed", 1.0))))
                        frame_interval = (1.0 / fps_base) / speed
                except asyncio.TimeoutError:
                    pass
                except json.JSONDecodeError:
                    pass

                data = agent.step()
                if not data:
                    await asyncio.sleep(0.01)
                    continue

                dt = time.perf_counter() - t0
                data["instant_fps"] = round(1.0 / max(dt, 1e-4), 1)
                await websocket.send_json(data)

                elapsed = time.perf_counter() - t0
                if frame_interval > elapsed:
                    await asyncio.sleep(frame_interval - elapsed)

        except WebSocketDisconnect:
            print("[WS] Client disconnected.")
        except Exception as exc:
            print(f"[WS Error] {exc}")

    return app


# Apply patch
GradioApp.create_app = _patched_create_app


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[HF Space] Launching DOOM-FlyWire on 0.0.0.0:{port}...")
    # demo.launch() → calls our patched App.create_app → registers ZeroGPU
    demo.launch(server_name="0.0.0.0", server_port=port, quiet=False)
