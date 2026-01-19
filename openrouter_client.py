# openrouter_client.py
"""
OpenRouter API client for making LLM requests.
Handles authentication, rate limiting, and error handling.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file

import requests
from typing import List, Dict, Optional


class OpenRouterError(Exception):
    """Base exception for OpenRouter errors"""
    pass


class OpenRouterResponseFormatError(OpenRouterError):
    """Exception for invalid response format"""
    pass


def call_openrouter(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 1.0,
) -> str:
    """
    Call OpenRouter API with the given parameters.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model ID (e.g., 'qwen/qwen3-coder:free')
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
    
    Returns:
        Model response as string
    
    Raises:
        OpenRouterError: If API call fails
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Please create a .env file with your API key. "
            "Get one free at: https://openrouter.ai/keys"
        )
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
        "X-Title": "Educational Tools App",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        # Check for 401 Unauthorized
        if response.status_code == 401:
            raise OpenRouterError(
                "Invalid or missing API key. Please check your OPENROUTER_API_KEY in .env file. "
                "Get a free key at: https://openrouter.ai/keys"
            )
        
        response.raise_for_status()
        
        data = response.json()
        
        # Check for error in response
        if "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            raise OpenRouterError(f"API error: {error_msg}")
        
        # Extract content
        if "choices" not in data or len(data["choices"]) == 0:
            raise OpenRouterError("No choices in API response")
        
        content = data["choices"][0]["message"]["content"]
        return content
    
    except requests.exceptions.Timeout:
        raise OpenRouterError("Request timeout - model took too long to respond")
    except requests.exceptions.RequestException as e:
        raise OpenRouterError(f"Request failed: {str(e)}")
    except KeyError as e:
        raise OpenRouterError(f"Unexpected response format: missing key {e}")
    except OpenRouterError:
        raise  # Re-raise our custom errors
    except Exception as e:
        raise OpenRouterError(f"Unexpected error: {str(e)}")
