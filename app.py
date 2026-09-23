import os
import gradio as gr
import uvicorn
from server import app as fastapi_app

# Mount Gradio Blocks for Hugging Face SDK healthcheck compatibility
demo = gr.Blocks(title="DOOM-FlyWire Connectome")
with demo:
    gr.HTML("<script>if (window.location.pathname.startsWith('/gradio')) { window.location.href = '/'; }</script>")

# Combine FastAPI and Gradio
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[Hugging Face Space] Launching DOOM-FlyWire on 0.0.0.0:{port}...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
