"""
DOOM-FlyWire Hugging Face Space Entry Point
============================================
Gradio SDK mode: HF runs `python app.py` directly.

Architecture:
  - FastAPI serves /, /ws, /static  (the actual DOOM + brain UI)
  - Gradio is mounted at /gradio     (satisfies HF SDK healthcheck)
  - uvicorn serves the combined app on port 7860

ZeroGPU: HF injects the `spaces` module in Gradio SDK containers.
         @spaces.GPU on any function satisfies the startup check.
"""

import os
import gradio as gr
import uvicorn
from server import app as fastapi_app

# ── ZeroGPU stub ──────────────────────────────────────────────────────────────
# HF injects `spaces` automatically in Gradio SDK ZeroGPU containers.
# We must NOT install it from PyPI — that overrides the injected version.
import spaces

@spaces.GPU
def _gpu_ready():
    """ZeroGPU activation stub — actual GPU work is in the connectome engine."""
    return True

# ── Gradio shim ───────────────────────────────────────────────────────────────
with gr.Blocks(title="DOOM-FlyWire Connectome") as demo:
    gr.Markdown(
        "## 🧠 DOOM-FlyWire Connectome\n"
        "The live visualizer is served at the root path. "
        "[Open the demo](/)"
    )
    run_btn = gr.Button("Activate GPU")
    run_btn.click(fn=_gpu_ready, inputs=[], outputs=[])

# Mount Gradio under /gradio — FastAPI handles /, /ws, /static
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[Hugging Face Space] Launching DOOM-FlyWire on 0.0.0.0:{port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
