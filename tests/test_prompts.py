"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

# Caminho absoluto (independe do diretório de onde o pytest é executado)
PROMPT_FILE = str(Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml")
PROMPT_KEY = "bug_to_user_story_v2"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_prompt_data():
    """Carrega e retorna os dados do prompt v2 (dicionário interno)."""
    data = load_prompts(PROMPT_FILE)
    assert data is not None, f"Não foi possível carregar o YAML: {PROMPT_FILE}"
    assert PROMPT_KEY in data, f"Chave '{PROMPT_KEY}' não encontrada no YAML"
    return data[PROMPT_KEY]


class TestPrompts:
    def test_prompt_has_system_prompt(self):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        data = get_prompt_data()
        assert "system_prompt" in data, "Campo 'system_prompt' ausente"
        assert isinstance(data["system_prompt"], str), "'system_prompt' deve ser uma string"
        assert data["system_prompt"].strip() != "", "'system_prompt' está vazio"

    def test_prompt_has_role_definition(self):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = get_prompt_data()["system_prompt"].lower()
        assert ("você é um" in system_prompt or "você é uma" in system_prompt), (
            "O prompt não define uma persona (esperado algo como 'Você é um Product Manager')"
        )

    def test_prompt_mentions_format(self):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        system_prompt = get_prompt_data()["system_prompt"].lower()
        assert ("markdown" in system_prompt or "user story" in system_prompt), (
            "O prompt não exige formato Markdown nem User Story padrão"
        )

    def test_prompt_has_few_shot_examples(self):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = get_prompt_data()["system_prompt"]
        assert "Bug report:" in system_prompt and "User Story:" in system_prompt, (
            "O prompt não contém pares de entrada/saída (Bug report: / User Story:)"
        )
        assert system_prompt.count("Bug report:") >= 2, (
            "Few-shot exige pelo menos 2 exemplos de entrada/saída"
        )

    def test_prompt_no_todos(self):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        data = get_prompt_data()
        full_text = (data.get("system_prompt", "") or "") + (data.get("user_prompt", "") or "")
        assert "TODO" not in full_text, "O prompt ainda contém [TODO] no texto"

    def test_minimum_techniques(self):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = get_prompt_data().get("techniques_applied", [])
        assert isinstance(techniques, list), "'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
