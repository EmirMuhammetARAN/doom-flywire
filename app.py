"""
DOOM-FlyWire Hugging Face Space Entry Point
============================================
Gradio SDK mode: HF runs `python app.py`.

Architecture:
  - spaces.GPU stub satisfies HF ZeroGPU detection at import time
  - FastAPI (server.py) serves /, /ws/game, /static
  - Gradio is mounted at /gradio for SDK healthcheck
  - Agent is initialized LAZILY on first WebSocket connect (not at startup)
  - uvicorn serves the combined app on port 7860
"""

# spaces MUST be imported first — HF injects this module and registers
# @spaces.GPU functions at import time for ZeroGPU detection.
import spaces

import os
import gradio as gr
import uvicorn

# ── ZeroGPU stub ──────────────────────────────────────────────────────────────
@spaces.GPU
def _gpu_ready():
    """
    ZeroGPU activation stub.
    Real GPU work happens in the connectome engine on WebSocket connect.
    This function must exist and be reachable from a Gradio event for HF
    ZeroGPU detection to pass.
    """
    import torch
    return f"GPU available: {torch.cuda.is_available()}"


# ── Import FastAPI game server (no startup side-effects) ──────────────────────
from server import app as fastapi_app


# ── Gradio shim ───────────────────────────────────────────────────────────────
with gr.Blocks(title="DOOM-FlyWire Connectome") as demo:
    gr.Markdown(
        "## 🧠 DOOM-FlyWire: 139k-Neuron Connectome\n"
        "The live DOOM + brain visualizer is at the **root path** — "
        "[open it here](/)."
    )
    with gr.Row():
        gpu_btn = gr.Button("🚀 Test GPU", variant="primary")
        gpu_out = gr.Textbox(label="GPU Status", interactive=False)
    gpu_btn.click(fn=_gpu_ready, inputs=[], outputs=[gpu_out])

# Mount Gradio at /gradio; FastAPI handles everything else
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[Hugging Face Space] Launching DOOM-FlyWire on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
