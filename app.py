"""
DOOM-FlyWire Hugging Face Space Entry Point
============================================
Wraps the FastAPI/WebSocket server with a Gradio shell so that HF Spaces
SDK health-checks pass. The actual UI is served as static HTML by FastAPI
(at /), Gradio is mounted at /gradio only as a thin shim.

ZeroGPU note: HF injects its own `spaces` module at runtime — do NOT install
it from PyPI (that breaks the @spaces.GPU registration mechanism). We import
it with a try/except so local runs still work.
"""

import os
import gradio as gr
import uvicorn
from server import app as fastapi_app


# ── ZeroGPU stub ──────────────────────────────────────────────────────────────
# HF's runtime injects `spaces` automatically in ZeroGPU containers.
# Installing it from PyPI would override that injection and break detection.
try:
    import spaces

    @spaces.GPU
    def _gpu_ready():
        """ZeroGPU activation stub — real GPU work is in the connectome engine."""
        return True

except ImportError:
    # Local development without HF runtime injection — no-op
    pass


# ── Gradio shim ───────────────────────────────────────────────────────────────
# Gradio 6.0 moved `js` from Blocks() to launch(). Since we use
# mount_gradio_app (no launch()), we skip the redirect JS entirely and
# just show a plain markdown fallback link.
with gr.Blocks(title="DOOM-FlyWire Connectome") as demo:
    gr.Markdown(
        "## DOOM-FlyWire Connectome\n"
        "The visualizer loads at the root path. "
        "[Open the live demo](/)"
    )

# Mount Gradio under /gradio; FastAPI serves /, /ws, /static
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[Hugging Face Space] Launching DOOM-FlyWire on 0.0.0.0:{port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
