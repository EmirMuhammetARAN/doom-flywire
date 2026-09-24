"""
DOOM-FlyWire Hugging Face Space Entry Point
============================================
Edge-to-edge DOOM-FlyWire Connectome Autonomous Agent & 3D Visualizer.
Game routes (/doom, /ws/game, assets) are injected directly at the head of
Gradio's FastAPI router before demo.launch() is called.
"""

import os
# Disable Gradio 6 Node.js SSR server so FastAPI handles all requests directly
os.environ["GRADIO_SSR_MODE"] = "false"
os.environ["GRADIO_NODE_PATH"] = ""

try:
    import gradio.node_server
    gradio.node_server.start_node_server = lambda *a, **k: (None, None, None)
except Exception:
    pass

try:
    import gradio.routes
    gradio.routes.start_node_server = lambda *a, **k: (None, None, None)
except Exception:
    pass

# spaces MUST be imported first: HF detects @spaces.GPU at import time
import spaces

import asyncio
import json
import threading
import time
import gradio as gr
from gradio.routes import App as GradioApp
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

# ── ZeroGPU stub ──────────────────────────────────────────────────────────────
# HF detects @spaces.GPU at scan/import time. No UI elements required.
@spaces.GPU
def _gpu_ready():
    import torch
    return torch.cuda.is_available()


WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
os.makedirs(WEB_DIR, exist_ok=True)


# ── Fullscreen Edge-to-Edge Visualizer ────────────────────────────────────────
with gr.Blocks(title="DOOM-FlyWire Connectome", fill_width=True, fill_height=True) as demo:
    gr.HTML("""
    <style>
        footer, .gradio-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; overflow: hidden !important; }
        #doom-full-frame {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            border: none;
            margin: 0;
            padding: 0;
            background: #050508;
            z-index: 99999;
        }
    </style>
    <iframe id="doom-full-frame" src="/doom" allow="fullscreen; autoplay"></iframe>
    """)


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


# ── Patch App.create_app to inject our routes ───────────────────────────
_orig_create_app = GradioApp.__dict__["create_app"]  # staticmethod descriptor


@staticmethod
def _patched_create_app(*args, **kwargs):
    app = _orig_create_app.__func__(*args, **kwargs)
    print(f"[FastAPI] Injected game routes at head of router. WEB_DIR={WEB_DIR}")

    def _serve(rel, media=""):
        f = os.path.join(WEB_DIR, rel) if rel else None
        if f and os.path.exists(f):
            return FileResponse(f, media_type=media) if media else FileResponse(f)
        return JSONResponse({"error": f"{rel} not found"}, status_code=404)

    async def doom_page(request): return _serve("index.html")
    async def health_page(request): return JSONResponse({"status": "ok"})
    async def ba_page(request): return _serve("brain_anatomy.json", "application/json")
    async def ba3_page(request): return _serve("brain_anatomy_3d.json", "application/json")
    async def bpos_page(request): return _serve("brain_139k_pos.bin", "application/octet-stream")
    async def bcol_page(request): return _serve("brain_139k_col.bin", "application/octet-stream")
    async def syn_page(request): return _serve("synapses_top5k.bin", "application/octet-stream")
    async def three_page(request): return _serve(os.path.join("js", "three.min.js"), "application/javascript")
    async def orbit_page(request): return _serve(os.path.join("js", "OrbitControls.js"), "application/javascript")

    # ── WebSocket game stream ─────────────────────────────────────────────────
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
        fi = (1.0 / fps_base) / speed

        try:
            while True:
                t0 = time.perf_counter()
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
                        fi = (1.0 / fps_base) / speed
                except (asyncio.TimeoutError, json.JSONDecodeError):
                    pass

                data = agent.step()
                if not data:
                    await asyncio.sleep(0.01)
                    continue

                dt = time.perf_counter() - t0
                data["instant_fps"] = round(1.0 / max(dt, 1e-4), 1)
                await websocket.send_json(data)

                elapsed = time.perf_counter() - t0
                slack = fi - elapsed
                if slack > 0:
                    await asyncio.sleep(slack)

        except WebSocketDisconnect:
            print("[WS] Client disconnected.")
        except Exception as exc:
            print(f"[WS Error] {exc}")

    # Inject routes at the head of app.router.routes so they take precedence over everything
    custom_routes = [
        Route("/doom", endpoint=doom_page, methods=["GET", "HEAD"]),
        Route("/doom/", endpoint=doom_page, methods=["GET", "HEAD"]),
        Route("/health", endpoint=health_page, methods=["GET"]),
        Route("/brain_anatomy.json", endpoint=ba_page, methods=["GET"]),
        Route("/brain_anatomy_3d.json", endpoint=ba3_page, methods=["GET"]),
        Route("/brain_139k_pos.bin", endpoint=bpos_page, methods=["GET"]),
        Route("/brain_139k_col.bin", endpoint=bcol_page, methods=["GET"]),
        Route("/synapses_top5k.bin", endpoint=syn_page, methods=["GET"]),
        Route("/js/three.min.js", endpoint=three_page, methods=["GET"]),
        Route("/js/OrbitControls.js", endpoint=orbit_page, methods=["GET"]),
        WebSocketRoute("/ws/game", endpoint=ws_game),
    ]
    for r in reversed(custom_routes):
        app.router.routes.insert(0, r)

    return app


GradioApp.create_app = _patched_create_app


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[HF Space] Launching DOOM-FlyWire on 0.0.0.0:{port}...")
    demo.launch(server_name="0.0.0.0", server_port=port, quiet=False)
