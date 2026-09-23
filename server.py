"""
DOOM-FlyWire Real-Time Web Server
=================================
FastAPI + WebSockets server streaming live ViZDoom gameplay, 139k-neuron
connectome telemetry, and accepting real-time neuroscientist sandbox controls.
"""

import os
import sys
import json
import time
import asyncio
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.doom_agent import DoomConnectomeAgent

app = FastAPI(title="DOOM-FlyWire Connectome Sandbox")

# Global Agent Instance
agent = None

@app.on_event("startup")
def startup_event():
    global agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Server] Initializing DOOM-FlyWire Agent on {device} (defend_the_center)...")
    agent = DoomConnectomeAgent(scenario_name="defend_the_center.cfg", window_visible=False, device=device)
    print("[Server] Agent ready! Listening for web client connections.")

@app.on_event("shutdown")
def shutdown_event():
    global agent
    if agent is not None:
        agent.close()

# Serve static web assets
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
os.makedirs(WEB_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
async def get_index():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "DOOM-FlyWire Server Running. Web UI not found in web/."}

@app.get("/brain_anatomy.json")
async def get_brain_anatomy():
    anatomy_file = os.path.join(WEB_DIR, "brain_anatomy.json")
    if os.path.exists(anatomy_file):
        return FileResponse(anatomy_file)
    return {"error": "Not found"}

@app.get("/brain_anatomy_3d.json")
async def get_brain_anatomy_3d():
    anatomy_file = os.path.join(WEB_DIR, "brain_anatomy_3d.json")
    if os.path.exists(anatomy_file):
        return FileResponse(anatomy_file)
    return {"error": "Not found"}

@app.get("/brain_139k_pos.bin")
async def get_brain_139k_pos():
    pos_file = os.path.join(WEB_DIR, "brain_139k_pos.bin")
    if os.path.exists(pos_file):
        return FileResponse(pos_file, media_type="application/octet-stream")
    return {"error": "Not found"}

@app.get("/brain_139k_col.bin")
async def get_brain_139k_col():
    col_file = os.path.join(WEB_DIR, "brain_139k_col.bin")
    if os.path.exists(col_file):
        return FileResponse(col_file, media_type="application/octet-stream")
    return {"error": "Not found"}

@app.get("/synapses_top5k.bin")
async def get_synapses_top5k():
    syn_file = os.path.join(WEB_DIR, "synapses_top5k.bin")
    if os.path.exists(syn_file):
        return FileResponse(syn_file, media_type="application/octet-stream")
    return {"error": "Not found"}


@app.get("/js/three.min.js")
async def get_three_js():
    js_file = os.path.join(WEB_DIR, "js", "three.min.js")
    if os.path.exists(js_file):
        return FileResponse(js_file, media_type="application/javascript")
    return {"error": "Not found"}

@app.get("/js/OrbitControls.js")
async def get_orbit_controls():
    js_file = os.path.join(WEB_DIR, "js", "OrbitControls.js")
    if os.path.exists(js_file):
        return FileResponse(js_file, media_type="application/javascript")
    return {"error": "Not found"}


@app.websocket("/ws/game")
async def websocket_game_stream(websocket: WebSocket):
    global agent
    await websocket.accept()
    print("[WebSocket] Client connected to live neural stream.")
    if agent is not None:
        agent.is_running = True

    fps_base = 20.0  # smooth natural human viewing speed
    speed_multiplier = 1.0
    frame_interval = (1.0 / fps_base) / speed_multiplier

    try:
        while True:
            t_start = time.perf_counter()

            # 1. Check for incoming parameter adjustments from web UI
            try:
                # Non-blocking receive
                data_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(data_text)
                if msg.get("type") == "update_params":
                    agent.update_sandbox(msg.get("params", {}))
                elif msg.get("type") in ("reset", "start"):
                    agent.reset_episode()
                elif msg.get("type") == "toggle_pause":
                    agent.toggle_pause()
                elif msg.get("type") == "set_speed":
                    speed_multiplier = max(0.25, min(3.0, float(msg.get("speed", 1.0))))
                    frame_interval = (1.0 / fps_base) / speed_multiplier
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass

            # 2. Step the 139k-neuron closed-loop agent
            step_data = agent.step()
            if not step_data:
                await asyncio.sleep(0.01)
                continue

            # 3. Add timestamp and compute instantaneous FPS
            t_step = time.perf_counter() - t_start
            step_data["instant_fps"] = round(1.0 / max(t_step, 1e-4), 1)

            # 4. Stream frame & brain telemetry to client
            await websocket.send_json(step_data)

            # Maintain smooth frame rate
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
