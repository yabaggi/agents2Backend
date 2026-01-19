# app.py
"""
Combined FastAPI application for Educational & Development Tools.
Serves both Study and Code tabs with dynamic model configuration.
"""

# LOAD ENVIRONMENT VARIABLES FIRST
from dotenv import load_dotenv
load_dotenv()  # This loads .env file

import json
import os
from textwrap import dedent
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openrouter_client import (
    call_openrouter,
    OpenRouterError,
    OpenRouterResponseFormatError,
)
from models_config import (
    AVAILABLE_MODELS,
    ModelInfo,
    get_all_models,
    get_models_by_category,
    get_free_models,
    get_premium_models,
    get_model_info,
    get_default_model,
)

# --- Verify API Key on Startup ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("=" * 60)
    print("⚠️  WARNING: OPENROUTER_API_KEY not found!")
    print("=" * 60)
    print("Please create a .env file with:")
    print("OPENROUTER_API_KEY=your_key_here")
    print("=" * 60)

# --- Environment Variables ---
GENERAL_MODEL = os.getenv("OPENROUTER_MODEL_GENERAL", get_default_model("general"))
CODE_MODEL = os.getenv("OPENROUTER_MODEL_CODE", get_default_model("code"))

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Educational & Development Tools API",
    description="AI-powered tools for learning and development",
    version="1.0.0"
)

# app.py - Update CORS section
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://yourusername.github.io",  # Your GitHub Pages URL
        "https://your-frontend.vercel.app",  # Your Vercel URL
        "*"  # Remove this in production, specify exact origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... rest of your app.py code stays the same ...






# --- Initialize FastAPI App ---
app = FastAPI(
    title="Educational & Development Tools API",
    description="AI-powered tools for learning and development",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/models/list")
def list_models(
    category: Optional[str] = Query(None, description="Filter by category: study, code, or general"),
    free_only: bool = Query(False, description="Show only free models")
):
    """
    Get list of available models with optional filtering.
    
    Query Parameters:
    - category: Filter by model category (study/code/general)
    - free_only: If true, only return free models
    """
    models = get_all_models()
    
    if category:
        models = [m for m in models if m.category == category]
    
    if free_only:
        models = [m for m in models if m.is_free]
    
    return {
        "models": [m.dict() for m in models],
        "total": len(models)
    }


@app.get("/models/{model_id}")
def get_model_details(model_id: str):
    """Get detailed information about a specific model"""
    info = get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return info.dict()


@app.get("/models/defaults")
def get_default_models():
    """Get default models for each category"""
    return {
        "general": get_default_model("general"),
        "code": get_default_model("code"),
        "study": get_default_model("study"),
    }


# ============================================================================
# STUDY TAB MODELS & ENDPOINTS
# ============================================================================

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_index: int
    explanation: str


class QuizRequest(BaseModel):
    lesson_text: str
    num_questions: int = 3
    difficulty: str = "beginner"
    model: Optional[str] = None


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]


class ExplainRequest(BaseModel):
    text: str
    level: str = "beginner"
    model: Optional[str] = None


class ExplainResponse(BaseModel):
    explanation: str


class StudyGuideRequest(BaseModel):
    lesson_text: str
    model: Optional[str] = None


class StudyGuideResponse(BaseModel):
    summary: str
    key_terms: str
    self_check_questions: str


# --- Helper Functions ---

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from text"""
    s = text.strip()
    
    if s.startswith("```"):
        parts = s.split("\n", 1)
        s = parts[1] if len(parts) > 1 else ""
        if "```" in s:
            s = s.rsplit("```", 1)[0]
    return s.strip()


def parse_quiz_json(raw_content: str) -> QuizResponse:
    """Parse quiz JSON response from model"""
    cleaned = _strip_code_fences(raw_content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise OpenRouterResponseFormatError(
            f"Quiz output is not valid JSON: {e}\nRaw: {raw_content[:2000]}"
        )
    try:
        return QuizResponse(**data)
    except Exception as e:
        raise OpenRouterResponseFormatError(
            f"Quiz JSON does not match expected schema: {e}\nData: {data}"
        )


# --- Study Endpoints ---

@app.post("/quiz/from-lesson", response_model=QuizResponse)
def generate_quiz(req: QuizRequest):
    """Generate a quiz from lesson text"""
    system_prompt = (
        "You are an assessment designer for corporate training.\n"
        "You ONLY use the provided lesson text; do not invent facts.\n"
        "Output STRICTLY valid JSON in this format:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question": string,\n'
        '      "options": [string, string, string, string],\n'
        '      "correct_index": integer,  // 0-3\n'
        '      "explanation": string\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    user_prompt = (
        f"Create {req.num_questions} {req.difficulty}-level multiple-choice questions "
        f"based ONLY on this lesson content.\n\n"
        f"Lesson content:\n-----\n{req.lesson_text}\n-----"
    )

    model_to_use = req.model or GENERAL_MODEL

    try:
        raw = call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model_to_use,
            temperature=0.4,
            max_tokens=1500,
        )
        quiz = parse_quiz_json(raw)
        return quiz

    except OpenRouterResponseFormatError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except OpenRouterError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e}")


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    """Explain text at specified level"""
    system_prompt = (
        "You are a helpful tutor for adult professionals.\n"
        "Adapt explanations to the learner's level.\n"
        "Keep answers under 200 words.\n"
        "Use short paragraphs and bullets when helpful."
    )

    user_prompt = (
        f"Learner level: {req.level}\n\n"
        f"Explain the following text so that someone at this level can understand it.\n"
        f"If already simple, clarify further or give examples.\n\n"
        f"Text:\n-----\n{req.text}\n-----"
    )

    model_to_use = req.model or GENERAL_MODEL

    try:
        raw = call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model_to_use,
            temperature=0.3,
            max_tokens=400,
        )
        return ExplainResponse(explanation=raw.strip())

    except OpenRouterError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e}")


@app.post("/study-guide", response_model=StudyGuideResponse)
def study_guide(req: StudyGuideRequest):
    """Generate a study guide from lesson text"""
    system_prompt = (
        "You generate concise study guides for short lessons.\n"
        "Output in markdown with three sections:\n"
        "1. **Summary** (3–5 bullet points)\n"
        "2. **Key Terms and Definitions** (markdown list)\n"
        "3. **Self-Check Questions** (3 open-ended questions, no answers)"
    )

    user_prompt = (
        "Create a study guide for the following lesson:\n"
        "-----\n"
        f"{req.lesson_text}\n"
        "-----"
    )

    model_to_use = req.model or GENERAL_MODEL

    try:
        raw = call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model_to_use,
            temperature=0.3,
            max_tokens=800,
        )

        return StudyGuideResponse(
            summary=raw.strip(),
            key_terms="(see markdown above)",
            self_check_questions="(see markdown above)",
        )

    except OpenRouterError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e}")


# ============================================================================
# CODE TAB MODELS & ENDPOINTS
# ============================================================================

class CommitMessageRequest(BaseModel):
    diff: str
    model: Optional[str] = None


class CommitMessageResponse(BaseModel):
    message: str


class ExplainCodeRequest(BaseModel):
    code: str
    language: Optional[str] = None
    model: Optional[str] = None


class ExplainCodeResponse(BaseModel):
    explanation: str


class DocstringRequest(BaseModel):
    function_source: str
    style: str = "google"
    model: Optional[str] = None


class DocstringResponse(BaseModel):
    docstring: str


# --- Code Service Functions ---

def generate_commit_message(diff: str, model: str) -> str:
    """Generate a git commit message from a diff"""
    system_prompt = dedent("""
        You are an assistant that writes high-quality Git commit messages.
        Use conventional commits when possible: feat, fix, chore, refactor, docs, test.
        Respond with:
        - First line: subject (<= 72 chars, imperative mood)
        - Optional body: bullet points summarizing key changes.
    """)

    user_prompt = dedent(f"""
        Generate a commit message for the following staged diff.

        Diff:
        -----
        {diff}
        -----
    """)

    return call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        model=model,
        temperature=0.2,
        max_tokens=256,
    ).strip()


def explain_code_snippet(code: str, language: Optional[str], model: str) -> str:
    """Explain a code snippet"""
    lang_info = f"Language: {language}\n" if language else ""
    system_prompt = dedent("""
        You are a senior software engineer helping teammates understand code.
        For the given snippet, output three sections:

        1) High-level summary (1–3 sentences)
        2) Potential issues or edge cases (bullet list, can be empty)
        3) Suggested tests (bullet list of test scenarios)

        Assume the reader knows the language syntax but not this codebase.
    """)

    user_prompt = dedent(f"""
        {lang_info}
        Explain the following code snippet:

        -----
        {code}
        -----

        If something is unclear due to missing context, say so explicitly.
    """)

    return call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        model=model,
        temperature=0.3,
        max_tokens=512,
    ).strip()


def generate_docstring(function_source: str, style: str, model: str) -> str:
    """Generate a docstring for a function"""
    style_map = {
        "google": "Google-style",
        "numpy": "NumPy-style",
        "sphinx": "Sphinx-style",
    }
    style_text = style_map.get(style, "a clear and standard")

    system_prompt = dedent(f"""
        You are a code documentation assistant.
        Given a function in Python, you write {style_text} docstring for it.
        Infer parameter types and return value from context when possible.
        If something is unknown, be generic rather than guessing.
        Respond with ONLY the docstring, including the opening and closing triple quotes.
    """)

    user_prompt = dedent(f"""
        Write a docstring for this function:

        -----
        {function_source}
        -----
    """)

    return call_openrouter(
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        model=model,
        temperature=0.2,
        max_tokens=256,
    ).strip()


# --- Code Endpoints ---

@app.post("/code/commit-message", response_model=CommitMessageResponse)
def commit_message(req: CommitMessageRequest):
    """Generate a git commit message from a diff"""
    model_to_use = req.model or CODE_MODEL
    try:
        msg = generate_commit_message(req.diff, model_to_use)
        return CommitMessageResponse(message=msg)
    except OpenRouterError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e}")


@app.post("/code/explain", response_model=ExplainCodeResponse)
def explain_code(req: ExplainCodeRequest):
    """Explain a code snippet"""
    model_to_use = req.model or CODE_MODEL
    try:
        text = explain_code_snippet(req.code, req.language, model_to_use)
        return ExplainCodeResponse(explanation=text)
    except OpenRouterError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e}")


@app.post("/code/docstring", response_model=DocstringResponse)
def docstring(req: DocstringRequest):
    """Generate a docstring for a function"""
    model_to_use = req.model or CODE_MODEL
    try:
        doc = generate_docstring(req.function_source, req.style, model_to_use)
        return DocstringResponse(docstring=doc)
    except OpenRouterError as e:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e}")


# --- Health Check & Info ---

@app.get("/")
def root():
    """API information and health check"""
    return {
        "message": "Educational & Development Tools API",
        "version": "1.0.0",
        "endpoints": {
            "models": ["/models/list", "/models/{model_id}", "/models/defaults"],
            "study": ["/quiz/from-lesson", "/explain", "/study-guide"],
            "code": ["/code/commit-message", "/code/explain", "/code/docstring"]
        },
        "documentation": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    return {
        "status": "healthy",
        "api_key_configured": bool(api_key),
        "models_available": len(AVAILABLE_MODELS)
    }


