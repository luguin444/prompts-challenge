"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

PROMPT_NAME = "leonanluppi/bug_to_user_story_v1"
OUTPUT_PATH = "prompts/bug_to_user_story_v1.yml"


def serialize_chat_prompt(prompt) -> dict:
    """Extrai system_prompt/user_prompt de um ChatPromptTemplate vindo do Hub."""
    # Fallback: se vier um RunnableSequence (prompt + model), pega o passo com .messages
    if not hasattr(prompt, "messages"):
        candidates = getattr(prompt, "steps", None) or [getattr(prompt, "first", None)]
        for step in candidates:
            if hasattr(step, "messages"):
                prompt = step
                break

    system_prompt, user_prompt = "", ""
    for message in getattr(prompt, "messages", []):
        template_text = getattr(getattr(message, "prompt", None), "template", "")
        role = type(message).__name__.lower()
        if "system" in role:
            system_prompt = template_text
        elif "human" in role or "user" in role:
            user_prompt = template_text

    return {
        "bug_to_user_story_v1": {
            "description": "Prompt inicial (v1) de baixa qualidade puxado do LangSmith Prompt Hub",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "version": "v1",
            "input_variables": list(getattr(prompt, "input_variables", [])),
            "source": PROMPT_NAME,
        }
    }


def pull_prompts_from_langsmith() -> bool:
    """Puxa o prompt do Hub e salva em prompts/bug_to_user_story_v1.yml."""
    try:
        print(f"Puxando prompt do LangSmith Hub: {PROMPT_NAME}")
        prompt = hub.pull(PROMPT_NAME)
        print("   ✓ Prompt carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao puxar prompt do LangSmith Hub: {e}")
        return False

    data = serialize_chat_prompt(prompt)
    if not data["bug_to_user_story_v1"]["system_prompt"]:
        print("⚠️  Não foi possível extrair o system_prompt do template puxado.")

    if save_yaml(data, OUTPUT_PATH):
        print(f"   ✓ Prompt salvo localmente em: {OUTPUT_PATH}")
        return True
    return False


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")
    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1
    return 0 if pull_prompts_from_langsmith() else 1


if __name__ == "__main__":
    sys.exit(main())
