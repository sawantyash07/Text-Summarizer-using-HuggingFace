import os
import contextlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, constr
import uvicorn

# 🚀 Environment Setup for Performance & Stability
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TRANSFORMERS_NO_TENSORFLOW'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 📂 Global Variables
templates = Jinja2Templates(directory="templates")
# Global container for the model pipe
summarizer_pipe = {}

# ⚡ Lifespan Manager: Loads the model (LAZY LOADING FOR FAST STARTUP)
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server started! Initializing AI Model in background...")
    try:
        # Move heavy import inside to make server startup nearly instant
        from transformers import pipeline
        
        # Using a proven high-quality T5 SAMSum model
        # For local use it looks in the current dir, for Render it downloads from HF
        model_path = "knkarthick/SAMSum-T5" 
        
        summarizer_pipe["instance"] = pipeline(
            "summarization", 
            model=model_path, 
            device=-1 # Ensure CPU usage
        )
        print(f"✅ AI Model [{model_path}] loaded successfully!")
    except Exception as e:
        print(f"❌ Error during model loading: {e}")
    yield
    summarizer_pipe.clear()

# ⚙️ Initialize FastAPI
app = FastAPI(title="AI Text Summarizer - T5 Edition", lifespan=lifespan)

# ✅ Input Validation Schema
class SummarizationRequest(BaseModel):
    text: constr(min_length=20, max_length=5000)

# 🌐 API Endpoints - Home Page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 🌐 API Endpoints - Summarization Action
@app.post("/summarize")
async def summarize_text(payload: SummarizationRequest):
    text_content = payload.text
    
    if "instance" not in summarizer_pipe:
        raise HTTPException(status_code=503, detail="The AI model is still initializing. Please wait about 30 seconds and try again.")

    try:
        result = summarizer_pipe["instance"](
            text_content, 
            max_length=150, 
            min_length=30, 
            do_sample=False
        )
        summary_text = result[0]['summary_text']
        
        return {
            "summary": summary_text,
            "original_word_count": len(text_content.split()),
            "summary_word_count": len(summary_text.split())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

# ▶️ RUN CONFIGURATION (OPTIMIZED FOR BOTH LOCAL & RENDER)
if __name__ == "__main__":
    # Check if we are running on Render (Render sets the PORT environment variable)
    RENDER_PORT = os.environ.get("PORT")
    
    if RENDER_PORT:
        # DEPLOYMENT SETTINGS
        uvicorn.run(app, host="0.0.0.0", port=int(RENDER_PORT))
    else:
        # LOCAL SETTINGS (Fixed for Chrome error 10048 and 0.0.0.0 issues)
        print("🏠 Running locally on http://127.0.0.1:8080")
        uvicorn.run(app, host="127.0.0.1", port=8080)
