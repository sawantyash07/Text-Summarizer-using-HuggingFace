import os
import contextlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, constr
from transformers import pipeline
import uvicorn

# 🚀 Environment Setup for Performance & Stability
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TRANSFORMERS_NO_TENSORFLOW'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 📂 Global Variables
templates = Jinja2Templates(directory="templates")
# Global container for the model pipe
summarizer_pipe = {}

# ⚡ Lifespan Manager: Loads the model from Hugging Face Hub (FOR RENDER DEPLOYMENT)
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔄 Initializing deployment and downloading model from Hugging Face...")
    try:
        # Using a proven high-quality T5 SAMSum model from the community
        # Replace with your own model name if you have uploaded it to HF Hub
        model_name = "knkarthick/SAMSum-T5" 
        
        summarizer_pipe["instance"] = pipeline(
            "summarization", 
            model=model_name, 
            device=-1 # Ensure CPU usage for Render free tier
        )
        print(f"✅ AI Model [{model_name}] loaded successfully from HF Hub!")
    except Exception as e:
        print(f"❌ Critical Error during model loading: {e}")
    yield
    summarizer_pipe.clear()

# ⚙️ Initialize FastAPI with Lifespan
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
        raise HTTPException(status_code=503, detail="The AI model is still initializing. Please try again in moments.")

    try:
        # Standard parameters for SAMSum dialogue summarization
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

# ▶️ Deployment Server Entry
if __name__ == "__main__":
    # Render uses special port binding ($PORT env var)
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
