"""
DOOM-FlyWire Real-Time Web Server
=================================
FastAPI + WebSockets server streaming live ViZDoom gameplay, 139k-neuron
connectome telemetry, and accepting real-time neuroscientist sandbox controls.

HF ZeroGPU note: Agent is initialized lazily on first WebSocket connection,
NOT during FastAPI startup, so the app stays alive even if GPU is unavailable
at startup time.
"""

import os
import sys
import json
import time
import asyncio
import threading
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(title="DOOM-FlyWire Connectome Sandbox")

# Global agent — initialized lazily on first WebSocket connection
agent = None
agent_lock = threading.Lock()
agent_init_error = None  # Stores initialization error if any


def _init_agent():
    """Initialize the DoomConnectomeAgent. Called lazily, not at startup."""
    global agent, agent_init_error
    with agent_lock:
        if agent is not None:
            return agent
        try:
            from src.doom_agent import DoomConnectomeAgent
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Server] Lazy-initializing DOOM-FlyWire Agent on {device}...")
            agent = DoomConnectomeAgent(
                scenario_name="defend_the_center.cfg",
                window_visible=False,
                device=device
            )
            print("[Server] Agent ready!")
            return agent
        except Exception as e:
            agent_init_error = str(e)
            print(f"[Server] Agent initialization failed: {e}")
            return None


@app.on_event("shutdown")
def shutdown_event():
    global agent
    if agent is not None:
        try:
            agent.close()
        except Exception:
            pass


# Serve static web assets
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
os.makedirs(WEB_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def get_index():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "DOOM-FlyWire Server Running. Web UI not found."})


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "agent_ready": agent is not None})


@app.get("/brain_anatomy.json")
async def get_brain_anatomy():
    f = os.path.join(WEB_DIR, "brain_anatomy.json")
    return FileResponse(f) if os.path.exists(f) else JSONResponse({"error": "Not found"})


@app.get("/brain_anatomy_3d.json")
async def get_brain_anatomy_3d():
    f = os.path.join(WEB_DIR, "brain_anatomy_3d.json")
    return FileResponse(f) if os.path.exists(f) else JSONResponse({"error": "Not found"})


@app.get("/brain_139k_pos.bin")
async def get_brain_pos():
    f = os.path.join(WEB_DIR, "brain_139k_pos.bin")
    return FileResponse(f, media_type="application/octet-stream") if os.path.exists(f) else JSONResponse({"error": "Not found"})


@app.get("/brain_139k_col.bin")
async def get_brain_col():
    f = os.path.join(WEB_DIR, "brain_139k_col.bin")
    return FileResponse(f, media_type="application/octet-stream") if os.path.exists(f) else JSONResponse({"error": "Not found"})


@app.get("/synapses_top5k.bin")
async def get_synapses():
    f = os.path.join(WEB_DIR, "synapses_top5k.bin")
    return FileResponse(f, media_type="application/octet-stream") if os.path.exists(f) else JSONResponse({"error": "Not found"})


@app.get("/js/three.min.js")
async def get_three_js():
    f = os.path.join(WEB_DIR, "js", "three.min.js")
    return FileResponse(f, media_type="application/javascript") if os.path.exists(f) else JSONResponse({"error": "Not found"})


@app.get("/js/OrbitControls.js")
async def get_orbit_controls():
    f = os.path.join(WEB_DIR, "js", "OrbitControls.js")
    return FileResponse(f, media_type="application/javascript") if os.path.exists(f) else JSONResponse({"error": "Not found"})


@app.websocket("/ws/game")
async def websocket_game_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WebSocket] Client connected to live neural stream.")

    # Lazy agent initialization — safe to do outside startup
    current_agent = _init_agent()
    if current_agent is None:
        await websocket.send_json({
            "type": "error",
            "message": f"Agent initialization failed: {agent_init_error}"
        })
        await websocket.close()
        return

    current_agent.is_running = True
    fps_base = 20.0
    speed_multiplier = 1.0
    frame_interval = (1.0 / fps_base) / speed_multiplier

    try:
        while True:
            t_start = time.perf_counter()

            # Check for incoming parameter adjustments
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

            # Step the 139k-neuron closed-loop agent
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
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket Error] {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"[Server] Starting DOOM-FlyWire on 0.0.0.0:{port}...")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
