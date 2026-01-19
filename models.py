# models.py
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError


# --- Quiz generation ---

class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(..., min_items=2)
    correct_index: int
    explanation: str


class QuizRequest(BaseModel):
    lesson_text: str
    num_questions: int = Field(5, ge=1, le=20)
    difficulty: str = Field("beginner")
    model: Optional[str] = None # allow passing a model ID

class QuizResponse(BaseModel):
    questions: List[QuizQuestion]


# --- Explanation ---

class ExplainRequest(BaseModel):
    text: str
    level: str = Field("beginner", pattern="^(beginner|intermediate|expert)$")
    model: Optional[str] = None # allow passing a model ID

class ExplainResponse(BaseModel):
    explanation: str


# --- Study guide ---

class StudyGuideRequest(BaseModel):
    lesson_text: str
    model: Optional[str] = None # allow passing a model ID

class StudyGuideResponse(BaseModel):
    summary: str
    key_terms: str
    self_check_questions: str

