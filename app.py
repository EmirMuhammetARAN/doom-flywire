"""
DOOM-FlyWire Hugging Face Space Entry Point
============================================
Architecture: Gradio SDK, single server (NO double-bind).

Uses gradio.routes.App.create_app(demo) to get Gradio's internal FastAPI app,
then adds our game routes (/doom, /ws/game, assets) directly to it.
No gr.mount_gradio_app — that injects a startup handler that tries to bind
port 7860 a second time, crashing the server.

ZeroGPU: @spaces.GPU is registered at import time (before uvicorn starts).
"""

# ── 1. Import spaces FIRST — HF detects @spaces.GPU at import time ────────────
import spaces

import os
import asyncio
import json
import time
import threading
import gradio as gr
from gradio.routes import App as GradioApp
from fastapi.responses import FileResponse, JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# ── 2. ZeroGPU stub ───────────────────────────────────────────────────────────
@spaces.GPU
def _gpu_ready():
    """ZeroGPU activation stub — must be connected to a Gradio event."""
    import torch
    return f"GPU available: {torch.cuda.is_available()}"


# ── 3. Gradio UI (Gradio serves at "/" and handles ZeroGPU coordination) ──────
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
os.makedirs(WEB_DIR, exist_ok=True)

with gr.Blocks(title="DOOM-FlyWire Connectome") as demo:
    # Embed the DOOM visualizer as iframe (served at /doom by our custom route)
    gr.HTML(
        '<iframe src="/doom" style="width:100%;height:78vh;border:none;" '
        'id="doom-frame"></iframe>'
    )
    with gr.Row():
        gpu_btn = gr.Button("🚀 Activate GPU", variant="primary", scale=1)
        gpu_out = gr.Textbox(label="GPU Status", interactive=False, scale=3)
    gpu_btn.click(fn=_gpu_ready, inputs=[], outputs=[gpu_out])
    gr.Markdown(
        "*The full visualizer lives in the iframe above. "
        "If it does not load, [open it directly](/doom).*"
    )


# ── 4. Build ONE app — Gradio's internal FastAPI + our game routes ─────────────
# App.create_app(demo) is what demo.launch() uses internally.
# We extend it with our custom routes instead of running a second server.
app = GradioApp.create_app(demo)


# ── 5. Game assets & web UI ───────────────────────────────────────────────────
@app.get("/doom")
async def doom_page():
    f = os.path.join(WEB_DIR, "index.html")
    return FileResponse(f) if os.path.exists(f) else JSONResponse({"error": "index.html not found"})

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})

@app.get("/brain_anatomy.json")
async def brain_anatomy():
    f = os.path.join(WEB_DIR, "brain_anatomy.json")
    return FileResponse(f) if os.path.exists(f) else JSONResponse({"error": "not found"})

@app.get("/brain_anatomy_3d.json")
async def brain_anatomy_3d():
    f = os.path.join(WEB_DIR, "brain_anatomy_3d.json")
    return FileResponse(f) if os.path.exists(f) else JSONResponse({"error": "not found"})

@app.get("/brain_139k_pos.bin")
async def brain_pos():
    f = os.path.join(WEB_DIR, "brain_139k_pos.bin")
    return FileResponse(f, media_type="application/octet-stream") if os.path.exists(f) else JSONResponse({"error": "not found"})

@app.get("/brain_139k_col.bin")
async def brain_col():
    f = os.path.join(WEB_DIR, "brain_139k_col.bin")
    return FileResponse(f, media_type="application/octet-stream") if os.path.exists(f) else JSONResponse({"error": "not found"})

@app.get("/synapses_top5k.bin")
async def synapses():
    f = os.path.join(WEB_DIR, "synapses_top5k.bin")
    return FileResponse(f, media_type="application/octet-stream") if os.path.exists(f) else JSONResponse({"error": "not found"})

@app.get("/js/three.min.js")
async def three_js():
    f = os.path.join(WEB_DIR, "js", "three.min.js")
    return FileResponse(f, media_type="application/javascript") if os.path.exists(f) else JSONResponse({"error": "not found"})

@app.get("/js/OrbitControls.js")
async def orbit_controls():
    f = os.path.join(WEB_DIR, "js", "OrbitControls.js")
    return FileResponse(f, media_type="application/javascript") if os.path.exists(f) else JSONResponse({"error": "not found"})


# ── 6. Game agent (lazy init — NOT at startup, avoids crash on cold boot) ─────
_agent = None
_agent_lock = threading.Lock()
_agent_error = None

def _init_agent():
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
                device=device
            )
            print("[Agent] Ready.")
            return _agent
        except Exception as e:
            _agent_error = str(e)
            print(f"[Agent] Init failed: {e}")
            return None


# ── 7. WebSocket game stream ──────────────────────────────────────────────────
@app.websocket("/ws/game")
async def ws_game_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected.")

    current_agent = _init_agent()
    if current_agent is None:
        await websocket.send_json({"type": "error", "message": _agent_error or "Agent init failed"})
        await websocket.close()
        return

    current_agent.is_running = True
    fps_base = 20.0
    speed_multiplier = 1.0
    frame_interval = (1.0 / fps_base) / speed_multiplier

    try:
        while True:
            t_start = time.perf_counter()
            try:
                data_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(data_text)
                if msg.get("type") == "update_params":
                    current_agent.update_sandbox(msg.get("params", {}))
                elif msg.get("type") in ("reset", "start"):
                    current_agent.reset_episode()
                elif msg.get("type") == "toggle_pause":
                    current_agent.toggle_pause()
                elif msg.get("type") == "set_speed":
                    speed_multiplier = max(0.25, min(3.0, float(msg.get("speed", 1.0))))
                    frame_interval = (1.0 / fps_base) / speed_multiplier
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            step_data = current_agent.step()
            if not step_data:
                await asyncio.sleep(0.01)
                continue

            t_step = time.perf_counter() - t_start
            step_data["instant_fps"] = round(1.0 / max(t_step, 1e-4), 1)
            await websocket.send_json(step_data)

            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.0, frame_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS Error] {e}")


# ── 8. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"[HF Space] Launching DOOM-FlyWire on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
