"""Evaluación cuantitativa del componente LLM sobre 20 casos de prueba.

La guía exige NO solo mostrar ejemplos positivos: se mide la calidad del
LLM con un set de preguntas con respuesta esperada.

Criterios automatizados:
1. ¿La respuesta menciona el equipo predicho como favorito?
2. ¿La respuesta menciona al menos uno de los top-3 factores SHAP?
3. ¿La respuesta está en español?
4. ¿La respuesta cabe en ≤200 palabras (cumple la restricción)?
"""
from dataclasses import dataclass, field


@dataclass
class LLMTestCase:
    home_team: str
    away_team: str
    expected_favorite: str  # equipo que debería mencionar como favorito
    expected_factors: list[str] = field(default_factory=list)  # palabras clave esperadas


def check_mentions_favorite(text: str, favorite: str) -> bool:
    return favorite.lower() in text.lower()


def check_mentions_factor(text: str, factors: list[str]) -> bool:
    text_l = text.lower()
    return any(f.lower() in text_l for f in factors)


def check_is_spanish(text: str) -> bool:
    """Heurística simple: presencia de palabras frecuentes en español."""
    keywords = [" el ", " la ", " de ", " que ", " es ", " los ", " las ", " un ", " una "]
    score = sum(1 for kw in keywords if kw in text.lower())
    return score >= 3


def check_within_length(text: str, max_words: int = 200) -> bool:
    return len(text.split()) <= max_words


def evaluate_response(text: str, test_case: LLMTestCase) -> dict:
    """Aplica los 4 criterios a una respuesta del LLM."""
    return {
        "mentions_favorite": check_mentions_favorite(text, test_case.expected_favorite),
        "mentions_factor": check_mentions_factor(text, test_case.expected_factors),
        "is_spanish": check_is_spanish(text),
        "within_length": check_within_length(text),
    }


def summarize(results: list[dict]) -> dict:
    """Calcula porcentajes de aciertos para cada criterio."""
    if not results:
        return {}
    keys = results[0].keys()
    return {k: sum(r[k] for r in results) / len(results) for k in keys}
