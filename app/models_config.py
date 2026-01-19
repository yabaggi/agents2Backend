# models_config.py
"""
Centralized model configuration for the application.
Add new models here - they'll automatically appear in the frontend dropdown.
"""

from typing import List, Optional
from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Model information schema"""
    id: str
    name: str
    provider: str
    category: str  # "study", "code", "general"
    is_free: bool
    description: Optional[str] = None
    context_window: Optional[int] = None
    pricing: Optional[str] = None
    recommended_for: Optional[str] = None


# ============================================================================
# MODEL CATALOG - Add new models here!
# ============================================================================

AVAILABLE_MODELS: List[ModelInfo] = [
    # ========== FREE MODELS ==========
    ModelInfo(
        id="qwen/qwen3-coder:free",
        name="Qwen 3 Coder",
        provider="Qwen",
        category="code",
        is_free=True,
        description="Optimized for coding tasks, great for code explanation and generation",
        context_window=32768,
        pricing="Free",
        recommended_for="Code generation, debugging, documentation"
    ),
    ModelInfo(
        id="google/gemini-2.0-flash-exp:free",
        name="Gemini 2.0 Flash",
        provider="Google",
        category="general",
        is_free=True,
        description="Fast and efficient, good for general tasks",
        context_window=1000000,
        pricing="Free",
        recommended_for="Study guides, explanations, general Q&A"
    ),
    ModelInfo(
        id="google/gemma-3-27b-it:free",
        name="Gemma 3 27B",
        provider="Google",
        category="general",
        is_free=True,
        description="Balanced performance for various tasks",
        context_window=8192,
        pricing="Free",
        recommended_for="Quiz generation, text explanations"
    ),
    ModelInfo(
        id="qwen/qwen3-4b:free",
        name="Qwen 3 4B",
        provider="Qwen",
        category="general",
        is_free=True,
        description="Lightweight and fast, good for simple tasks",
        context_window=32768,
        pricing="Free",
        recommended_for="Quick explanations, simple Q&A"
    ),
    ModelInfo(
        id="moonshotai/kimi-k2:free",
        name="Kimi K2",
        provider="Moonshot AI",
        category="general",
        is_free=True,
        description="Efficient model with good reasoning",
        context_window=128000,
        pricing="Free",
        recommended_for="Long context tasks, study materials"
    ),
    ModelInfo(
        id="openai/gpt-oss-20b:free",
        name="GPT OSS 20B",
        provider="OpenAI",
        category="general",
        is_free=True,
        description="Open source GPT variant",
        context_window=8192,
        pricing="Free",
        recommended_for="General purpose tasks"
    ),
    ModelInfo(
        id="google/gemini-2.0-flash-thinking-exp:free",
        name="Gemini 2.0 Flash Thinking",
        provider="Google",
        category="general",
        is_free=True,
        description="Enhanced reasoning capabilities",
        context_window=32768,
        pricing="Free",
        recommended_for="Complex problem solving, detailed explanations"
    ),
    
    # ========== PREMIUM MODELS ==========
    ModelInfo(
        id="anthropic/claude-3.5-sonnet",
        name="Claude 3.5 Sonnet",
        provider="Anthropic",
        category="general",
        is_free=False,
        description="Best-in-class reasoning, coding, and analysis",
        context_window=200000,
        pricing="$3/$15 per million tokens",
        recommended_for="Complex coding, detailed analysis, high-quality content"
    ),
    ModelInfo(
        id="openai/gpt-4o",
        name="GPT-4o",
        provider="OpenAI",
        category="general",
        is_free=False,
        description="Latest GPT-4 optimized model with multimodal capabilities",
        context_window=128000,
        pricing="$2.50/$10 per million tokens",
        recommended_for="Complex reasoning, creative writing, coding"
    ),
    ModelInfo(
        id="openai/gpt-4o-mini",
        name="GPT-4o Mini",
        provider="OpenAI",
        category="general",
        is_free=False,
        description="Faster and more affordable GPT-4o variant",
        context_window=128000,
        pricing="$0.15/$0.60 per million tokens",
        recommended_for="Fast responses, cost-effective tasks"
    ),
    ModelInfo(
        id="deepseek/deepseek-chat",
        name="DeepSeek Chat",
        provider="DeepSeek",
        category="code",
        is_free=False,
        description="Excellent for complex coding and technical tasks",
        context_window=64000,
        pricing="$0.27/$1.10 per million tokens",
        recommended_for="Advanced coding, system architecture"
    ),
    ModelInfo(
        id="meta-llama/llama-3.1-405b-instruct",
        name="Llama 3.1 405B",
        provider="Meta",
        category="general",
        is_free=False,
        description="Largest open-source model with excellent performance",
        context_window=131072,
        pricing="$2.70/$2.70 per million tokens",
        recommended_for="Complex tasks, long context understanding"
    ),
    ModelInfo(
        id="meta-llama/llama-3.1-70b-instruct",
        name="Llama 3.1 70B",
        provider="Meta",
        category="general",
        is_free=False,
        description="Balanced performance and cost",
        context_window=131072,
        pricing="$0.52/$0.75 per million tokens",
        recommended_for="General purpose, good value"
    ),
    ModelInfo(
        id="google/gemini-pro-1.5",
        name="Gemini Pro 1.5",
        provider="Google",
        category="general",
        is_free=False,
        description="Long context and multimodal capabilities",
        context_window=2000000,
        pricing="$1.25/$5 per million tokens",
        recommended_for="Very long documents, complex analysis"
    ),
    ModelInfo(
        id="mistralai/mistral-large-2411",
        name="Mistral Large",
        provider="Mistral AI",
        category="general",
        is_free=False,
        description="High-performance European model",
        context_window=128000,
        pricing="$2/$6 per million tokens",
        recommended_for="Multilingual tasks, code generation"
    ),
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_models() -> List[ModelInfo]:
    """Get all available models"""
    return AVAILABLE_MODELS


def get_models_by_category(category: str) -> List[ModelInfo]:
    """Get models filtered by category (study, code, general)"""
    return [m for m in AVAILABLE_MODELS if m.category == category]


def get_free_models() -> List[ModelInfo]:
    """Get only free models"""
    return [m for m in AVAILABLE_MODELS if m.is_free]


def get_premium_models() -> List[ModelInfo]:
    """Get only premium (paid) models"""
    return [m for m in AVAILABLE_MODELS if not m.is_free]


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get detailed information about a specific model"""
    return next((m for m in AVAILABLE_MODELS if m.id == model_id), None)


def get_default_model(category: str = "general") -> str:
    """Get default model ID for a category"""
    defaults = {
        "study": "google/gemini-2.0-flash-exp:free",
        "code": "qwen/qwen3-coder:free",
        "general": "google/gemini-2.0-flash-exp:free",
    }
    return defaults.get(category, "google/gemini-2.0-flash-exp:free")
