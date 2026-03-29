import os
import contextlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, constr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn

# 🚀 Environment Setup for Performance & Stability
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['TRANSFORMERS_NO_TENSORFLOW'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 📂 Global Variables
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="templates")
# Global container for the local model pipe
summarizer_pipe = {}

# ⚡ Lifespan Manager: Loads the model ONLY after the port is successfully bound
# This makes startup much faster if there is a port conflict!
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🔄 Analyzing project and loading model from: {MODEL_DIR}")
    try:
        # Loading resources only when the server is actually starting
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
        summarizer_pipe["instance"] = pipeline("summarization", model=model, tokenizer=tokenizer)
        print("✅ AI Model & Tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Critical Error during model initialization: {e}")
    yield
    # Cleanup logic (if any) can go here
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
        raise HTTPException(status_code=503, detail="The AI model is still initializing. Please try again in a few seconds.")

    try:
        # Execute the summarization using the pre-loaded instance
        result = summarizer_pipe["instance"](text_content, max_length=150, min_length=30, do_sample=False)
        summary_text = result[0]['summary_text']
        
        return {
            "summary": summary_text,
            "original_word_count": len(text_content.split()),
            "summary_word_count": len(summary_text.split())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

# ▶️ Development Server Entry
if __name__ == "__main__":
    # Switching to Port 8080 to avoid the "Socket Address In Use" error (10048)
    print("🚀 Starting Summarizer API at http://127.0.0.1:8080 ...")
    uvicorn.run(app, host="127.0.0.1", port=8080)
