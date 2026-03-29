# ⚡ AI Text Summarizer with T5 Transformer (FastAPI)

A high-performance, modern, and interactive text summarization application built with **FastAPI** and **Hugging Face Transformers**. The project features a premium "Red Noir" design with a sleek, responsive user interface.

---

## 🎨 Design Aesthetic: Red Noir Edition
- **Sleek Dark Mode:** Deep black backgrounds with subtle geometric grids and red accents.
- **Glassmorphism UI:** Translucent cards with real-time blur and glowing crimson borders.
- **Interactive Animations:** Fade-up transitions, pulsing sparkles, and dynamic "shake" validation.
- **Live Stats:** Character tracker, word count, and word reduction metrics.

---

## 🛠️ Tech Stack
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Async Web Framework)
- **Validation:** [Pydantic](https://docs.pydantic.dev/) (Data Modeling)
- **Core AI:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) (T5-SAMSum Architecture)
- **Frontend:** HTML5, [Tailwind CSS](https://tailwindcss.com/), Vanilla JavaScript, [Iconify](https://icon-sets.iconify.design/)
- **Templating:** Jinja2

---

## 🚀 How to Run Locally

### 1. **Clone & Setup Environment**
```bash
git clone https://github.com/sawantyash07/Text-Summarizer-using-HuggingFace.git
cd Text-Summarizer-using-HuggingFace
```

### 2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Run the Application**
Launch the FastAPI server (using the optimized `app.py` on port 8080):
```bash
python app.py
```

### 4. **Access the App**
Open your favorite browser and navigate to:
**[http://127.0.0.1:8080](http://127.0.0.1:8080)**

---

## 🌐 Deploying on Render

This project is ready for deployment on [Render](https://render.com/). 
The `render.yaml` and `requirements.txt` files are included for quick setup. Note that the application is currently configured to load the model from the **Hugging Face Hub** (`knkarthick/SAMSum-T5`) for optimized cloud deployment.

---

### 📬 Contact
Created by **Yash Sawant** - [GitHub](https://github.com/sawantyash07)
