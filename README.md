# ⚡ AI Text Summarizer with T5 Transformer

A high-performance, modern, and interactive text summarization application built with **FastAPI** and **Hugging Face Transformers**. The project features a premium "Red Noir" design with a sleek, responsive user interface.

---

## 🎨 Design Aesthetic: Red Noir Edition
- **Sleek Dark Mode:** Deep black backgrounds with subtle geometric grids.
- **Glassmorphism UI:** Translucent cards with real-time blur and glowing red accents.
- **Interactive Animations:** Fade-up transitions, pulsing sparkles, and dynamic "shake" validation.
- **Micro-interactions:** Live word count tracking, smart processing states, and clipboard copy confirmation.

---

## 🛠️ Tech Stack
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Async Web Framework)
- **Validation:** [Pydantic](https://docs.pydantic.dev/) (Data Modeling & Logic)
- **Core AI:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (T5-SAMSum Architecture)
- **Frontend:** HTML5, [Tailwind CSS](https://tailwindcss.com/), Vanilla JavaScript, [Iconify](https://icon-sets.iconify.design/)
- **Templating:** Jinja2

---

## 📂 Project Structure
```text
final_t5_samsum_model/
├── app.py                  # FastAPI Backend & Model Loading
├── .gitignore              # Git Ignore Rules
├── README.md               # Documentation
├── templates/
│   └── index.html          # Premium Frontend UI
└── [Model Files]           # Fine-tuned T5 weights (Locally hosted)
```

---

## 🚀 How to Run

### 1. **Clone & Setup Environment**
First, ensure you have Python 3.9+ installed.
```bash
git clone https://github.com/sawantyash07/Text-Summarizer-using-HuggingFace.git
cd Text-Summarizer-using-HuggingFace
```

### 2. **Install Dependencies**
Install all required libraries using pip:
```bash
pip install fastapi uvicorn pydantic jinja2 aiofiles transformers torch 
```
> **Note:** Since this project uses the `PyTorch` backend for Transformers, `tensorflow` is not required and may cause conflicts on Windows.

### 3. **Run the Application**
Launch the FastAPI dev server using the optimized `app.py`:
```bash
python app.py
```

### 4. **Access the App**
Open your favorite browser and navigate to:
**[http://127.0.0.1:8080](http://127.0.0.1:8080)**

---

## 💡 Key Features
- **Fast Model Loading:** Uses FastAPI's `lifespan` manager for asynchronous loading.
- **Word Reduction Stats:** Live calculation of original vs. summary word counts.
- **Copy to Clipboard:** One-tap export of generated intelligence.
- **Error Handling:** Robust Pydantic-driven validation for both client and server sides.

---

### 📬 Contact
Created by **Yash Sawant** - [GitHub](https://github.com/sawantyash07)
