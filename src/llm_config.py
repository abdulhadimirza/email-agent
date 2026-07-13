import os
from crewai import LLM

def get_groq_llm(**kwargs) -> LLM:
    """Helper to instantiate the GROQ LLM with environment variable fallbacks."""
    default_model = os.environ.get("MODEL", "groq/llama-3.1-8b-instant")
    api_key = os.environ.get("GROQ_API_KEY")
    
    # Allow overriding model, but default to the environment/hardcoded one
    model = kwargs.pop("model", default_model)
    
    return LLM(
        model=model,
        api_key=api_key,
        **kwargs
    )
