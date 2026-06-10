"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

SIMPLIFICADO: Código mais limpo e direto ao ponto.
"""

import os
import sys
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

PROMPT_YAML = "prompts/bug_to_user_story_v2.yml"
PROMPT_KEY = "bug_to_user_story_v2"
PROMPT_BASENAME = "bug_to_user_story_v2"


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    errors = []

    for field in ("description", "system_prompt", "user_prompt", "version"):
        if not prompt_data.get(field):
            errors.append(f"Campo obrigatório faltando ou vazio: {field}")

    system_prompt = (prompt_data.get("system_prompt") or "").strip()
    if "TODO" in system_prompt:
        errors.append("system_prompt ainda contém TODOs")

    user_prompt = prompt_data.get("user_prompt") or ""
    if "{bug_report}" not in user_prompt:
        errors.append("user_prompt deve conter a variável {bug_report}")

    techniques = prompt_data.get("techniques_applied", [])
    if len(techniques) < 2:
        errors.append(f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}")

    return (len(errors) == 0, errors)


def _build_readme(prompt_data: dict) -> str:
    """Monta um README com os metadados/técnicas para acompanhar o prompt no Hub."""
    lines = [
        f"# {prompt_data.get('description', PROMPT_BASENAME)}",
        "",
        f"Versão: {prompt_data.get('version', 'v2')}",
        "",
        "## Técnicas aplicadas",
    ]
    for technique in prompt_data.get("techniques_applied", []):
        lines.append(f"- {technique}")
    return "\n".join(lines)


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt (ex: "{username}/bug_to_user_story_v2")
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    try:
        template = ChatPromptTemplate.from_messages([
            ("system", prompt_data["system_prompt"]),
            ("user", prompt_data["user_prompt"]),
        ])

        print(f"Fazendo push (público) para: {prompt_name}")
        url = hub.push(
            prompt_name,
            template,
            new_repo_is_public=True,
            new_repo_description=prompt_data.get("description", ""),
            readme=_build_readme(prompt_data),
            tags=prompt_data.get("tags", []),
        )

        print("   ✓ Push concluído com sucesso")
        print(f"   URL: {url}")
        return True

    except Exception as e:
        print(f"❌ Erro ao fazer push do prompt '{prompt_name}': {e}")
        print("   Verifique LANGSMITH_API_KEY/USERNAME_LANGSMITH_HUB no .env e sua conexão.")
        return False


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS PARA O LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    data = load_yaml(PROMPT_YAML)
    if not data or PROMPT_KEY not in data:
        print(f"❌ Não foi possível carregar '{PROMPT_KEY}' de {PROMPT_YAML}")
        return 1

    prompt_data = data[PROMPT_KEY]

    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("❌ Prompt inválido:")
        for error in errors:
            print(f"   - {error}")
        return 1
    print("   ✓ Prompt validado")

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    prompt_name = f"{username}/{PROMPT_BASENAME}"

    if not push_prompt_to_langsmith(prompt_name, prompt_data):
        return 1

    print("\n" + "-" * 50)
    print("✅ Push concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
