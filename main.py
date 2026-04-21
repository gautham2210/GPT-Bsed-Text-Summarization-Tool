import os

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import evaluate

# Initialize FastAPI app
app = FastAPI(title="AI Text Summarizer with ROUGE")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load summarization model
summarizer = pipeline(
    "summarization",
    model="google/pegasus-xsum",
    framework="pt"
)

# Load ROUGE metric
rouge = evaluate.load("rouge")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "summary": "",
            "input_text": "",
            "reference_summary": "",
            "rouge_scores": None,
            "error": ""
        }
    )


@app.post("/", response_class=HTMLResponse)
async def summarize(
    request: Request,
    text: str = Form(...),
    reference_summary: str = Form("")
):
    summary = ""
    rouge_scores = None
    error = ""

    cleaned_text = text.strip()
    cleaned_reference = reference_summary.strip()

    if not cleaned_text:
        error = "Please enter some text to summarize."
    else:
        try:
            result = summarizer(
                cleaned_text,
                max_length=80,
                min_length=25,
                do_sample=False,
                num_beams=6,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True
            )

            summary = result[0]["summary_text"]

            if cleaned_reference:
                scores = rouge.compute(
                    predictions=[summary],
                    references=[cleaned_reference],
                    use_stemmer=True
                )

                rouge_scores = {
                    "rouge1": round(scores["rouge1"], 4),
                    "rouge2": round(scores["rouge2"], 4),
                    "rougeL": round(scores["rougeL"], 4),
                    "rougeLsum": round(scores["rougeLsum"], 4)
                }

        except Exception as e:
            error = f"Error: {str(e)}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "summary": summary,
            "input_text": text,
            "reference_summary": reference_summary,
            "rouge_scores": rouge_scores,
            "error": error
        }
    )