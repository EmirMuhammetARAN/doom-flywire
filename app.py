"""
DOOM-FlyWire Hugging Face Space Entry Point
============================================
demo.launch() is required for ZeroGPU registration with HF's backend proxy.
Game routes (/doom, /ws/game, assets) are injected into Gradio's internal
FastAPI via a staticmethod patch on App.create_app before demo.launch() is
called - single server, no double-bind.

SSR_MODE IS EXPLICITLY DISABLED (ssr_mode=False) so that Node.js does not
hijack port 7860 and intercept custom routes like /doom and /ws/game.
"""

import os
# Must disable Gradio 6 Node.js SSR server so FastAPI handles all routes directly
os.environ["GRADIO_SSR_MODE"] = "False"

# spaces MUST be imported first: HF detects @spaces.GPU at import time
import spaces

import asyncio
import json
import threading
import time
import gradio as gr
from gradio.routes import App as GradioApp
from starlette.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

# ── ZeroGPU stub ──────────────────────────────────────────────────────────────
@spaces.GPU
def _gpu_ready():
    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback'
    return f"ZeroGPU Active: CUDA available = {torch.cuda.is_available()} ({dev})"


WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
os.makedirs(WEB_DIR, exist_ok=True)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
custom_css = """
.launch-card {
    background: linear-gradient(135deg, rgba(20, 20, 35, 0.95), rgba(10, 10, 20, 0.95));
    border: 1px solid rgba(0, 245, 212, 0.3);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    text-align: center;
}
.launch-btn {
    display: inline-block;
    padding: 14px 32px;
    background: linear-gradient(135deg, #00f5d4, #00bb9e);
    color: #050508 !important;
    font-weight: 800;
    font-size: 1.15rem;
    border-radius: 8px;
    text-decoration: none !important;
    box-shadow: 0 4px 20px rgba(0, 245, 212, 0.4);
    transition: all 0.2s ease;
}
.launch-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 28px rgba(0, 245, 212, 0.6);
}
"""

with gr.Blocks(title="DOOM-FlyWire Connectome", css=custom_css) as demo:
    gr.HTML("""
    <div class="launch-card">
        <h1 style="color: #00f5d4; font-size: 2.2rem; margin-bottom: 8px; font-weight: 800;">
            🧠 DOOM-FlyWire: Whole-Brain Connectome
        </h1>
        <p style="color: #bbb; font-size: 1.1rem; max-width: 800px; margin: auto; line-height: 1.6;">
            A real-time biological digital twin running the full adult <em>Drosophila melanogaster</em> connectome
            (<strong>139,248 neurons</strong>, <strong>15,090,883 synapses</strong>) as an autonomous agent playing <strong>DOOM</strong>.
        </p>
        <div style="margin-top: 20px;">
            <a href="/doom" target="_blank" class="launch-btn">
                🎮 Launch Fullscreen Visualizer (New Tab)
            </a>
            <a href="/doom" style="display: inline-block; margin-left: 12px; padding: 14px 24px; background: rgba(255,255,255,0.08); color: #fff; border: 1px solid rgba(255,255,255,0.2); border-radius: 8px; text-decoration: none; font-weight: 600;">
                🕹️ Open Directly
            </a>
        </div>
    </div>
    """)

    with gr.Row():
        gpu_btn = gr.Button("🚀 Test ZeroGPU Allocation", variant="primary", scale=1)
        gpu_out = gr.Textbox(label="GPU Status", interactive=False, scale=3, placeholder="Click button to test ZeroGPU worker allocation...")
    gpu_btn.click(fn=_gpu_ready, inputs=[], outputs=[gpu_out])

    gr.HTML("""
    <div style="margin-top: 20px; border: 1px solid #333; border-radius: 12px; overflow: hidden; background: #000; box-shadow: 0 8px 32px rgba(0,0,0,0.6);">
        <div style="background: #111; padding: 8px 16px; font-size: 0.85rem; color: #888; border-bottom: 1px solid #222; display: flex; justify-content: space-between; align-items: center;">
            <span>Live WebGL & Stream Preview</span>
            <a href="/doom" target="_blank" style="color: #00f5d4; text-decoration: none;">⛶ Fullscreen</a>
        </div>
        <iframe src="/doom" style="width: 100%; height: 75vh; border: none; background: #050508;" allow="fullscreen; autoplay"></iframe>
    </div>
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
    kwargs["ssr_mode"] = False
    app = _orig_create_app.__func__(*args, **kwargs)
    print(f"[FastAPI] Injected game routes into FastAPI app. WEB_DIR={WEB_DIR} (exists={os.path.exists(WEB_DIR)})")

    def _serve(rel, media=""):
        f = os.path.join(WEB_DIR, rel) if rel else None
        if f and os.path.exists(f):
            return FileResponse(f, media_type=media) if media else FileResponse(f)
        return JSONResponse({"error": f"{rel} not found"}, status_code=404)

    @app.get("/doom")
    @app.get("/doom/")
    async def doom_page():
        return _serve("index.html")

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.get("/brain_anatomy.json")
    async def ba():
        return _serve("brain_anatomy.json", "application/json")

    @app.get("/brain_anatomy_3d.json")
    async def ba3():
        return _serve("brain_anatomy_3d.json", "application/json")

    @app.get("/brain_139k_pos.bin")
    async def bpos():
        return _serve("brain_139k_pos.bin", "application/octet-stream")

    @app.get("/brain_139k_col.bin")
    async def bcol():
        return _serve("brain_139k_col.bin", "application/octet-stream")

    @app.get("/synapses_top5k.bin")
    async def syn():
        return _serve("synapses_top5k.bin", "application/octet-stream")

    @app.get("/js/three.min.js")
    async def three():
        return _serve(os.path.join("js", "three.min.js"), "application/javascript")

    @app.get("/js/OrbitControls.js")
    async def orbit():
        return _serve(os.path.join("js", "OrbitControls.js"), "application/javascript")

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

    return app


GradioApp.create_app = _patched_create_app


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[HF Space] Launching DOOM-FlyWire on 0.0.0.0:{port}...")
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False, quiet=False)
