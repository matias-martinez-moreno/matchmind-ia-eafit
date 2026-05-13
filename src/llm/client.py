"""Cliente Groq API para generar análisis con Llama-3.1-8b."""
import os
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL = "llama-3.1-8b-instant"


class GroqClient:
    """Wrapper sobre la API de Groq con manejo simple de errores."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY no encontrada. Configura el archivo .env "
                "siguiendo .env.example."
            )
        self.client = Groq(api_key=key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.4,
                 max_tokens: int = 400) -> str:
        """Envía un prompt al LLM y devuelve la respuesta como string."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
