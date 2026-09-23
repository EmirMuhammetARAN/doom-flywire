"""
DOOM-FlyWire Hugging Face Space Entry Point
============================================
Wraps the FastAPI/WebSocket server with a Gradio shell so that HF Spaces
SDK health-checks pass. The actual UI is served as static HTML by FastAPI
(at /), Gradio is mounted at /gradio only as a thin shim.

ZeroGPU requirement: at least one @spaces.GPU decorated function must exist.
"""

import os
import spaces
import gradio as gr
import uvicorn
from server import app as fastapi_app


# ── ZeroGPU stub ──────────────────────────────────────────────────────────────
# HF ZeroGPU requires at least one @spaces.GPU function to be defined at
# import time. The real GPU work happens inside DoomConnectomeAgent (server.py),
# but we need this stub so the Space does not crash on startup.
@spaces.GPU
def _gpu_ready():
    """ZeroGPU activation stub — real GPU work is in the connectome engine."""
    return True


# ── Gradio shim ───────────────────────────────────────────────────────────────
# The full UI lives in web/index.html (served by FastAPI at "/").
# This Gradio block is only here to satisfy the HF SDK healthcheck.
_REDIRECT_JS = """
() => {
    if (window.location.pathname.startsWith('/gradio')) {
        window.location.href = '/';
    }
}
"""

with gr.Blocks(title="DOOM-FlyWire Connectome", js=_REDIRECT_JS) as demo:
    gr.Markdown(
        "## DOOM-FlyWire\n"
        "Loading the connectome visualizer... "
        "If you are not redirected automatically, [click here](/)."
    )

# Mount Gradio under /gradio; FastAPI handles everything else
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[Hugging Face Space] Launching DOOM-FlyWire on 0.0.0.0:{port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
