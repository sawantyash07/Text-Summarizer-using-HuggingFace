import os
import contextlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, constr
import uvicorn
import socket

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
    print("\n" + "="*50)
    print("🚀 API SERVER IS NOW AWAKE!")
    print("🏠 Access UI at: http://127.0.0.1:8000")
    print("="*50 + "\n")
    print("🔄 Initializing AI Intelligence in background...")
    
    try:
        from transformers import pipeline
        # Remote model path for better compatibility
        model_name = "knkarthick/SAMSum-T5" 
        
        summarizer_pipe["instance"] = pipeline(
            "summarization", 
            model=model_name, 
            device=-1 # CPU
        )
        print(f"✅ AI Brain [{model_name}] connected successfully!")
    except Exception as e:
        print(f"❌ Error during AI initialization: {e}")
        
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
        raise HTTPException(
            status_code=503, 
            detail="The AI engine is warming up. Please wait 30 seconds and try again."
        )

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

# ▶️ RUN CONFIGURATION
if __name__ == "__main__":
    # Check for Render environment
    PORT = os.environ.get("PORT")
    
    if PORT:
        # DEPLOYMENT MODE
        uvicorn.run(app, host="0.0.0.0", port=int(PORT))
    else:
        # LOCAL MODE
        # Using Port 8000 and 127.0.0.1 to avoid Chrome security errors
        uvicorn.run(app, host="127.0.0.1", port=8000)
