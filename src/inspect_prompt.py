"""
Ferramenta de DEBUG para iterar o prompt v2 LOCALMENTE (sem precisar fazer push).

Lê prompts/bug_to_user_story_v2.yml direto do disco, roda o modelo nos exemplos
escolhidos do dataset e imprime, lado a lado:
  - bug report
  - resposta gerada pelo modelo
  - referência (ground truth)
  - reasoning do avaliador F1 (precision / recall) -> aponta o que FALTA (recall)
    ou SOBRA/está errado (precision)

Fluxo de iteração rápido:
  1. edita prompts/bug_to_user_story_v2.yml
  2. roda este script (não precisa push)
  3. ajusta de novo; quando estiver bom -> push_prompts.py + evaluate.py oficial

Uso:
  python src/inspect_prompt.py            # todos os 15 exemplos
  python src/inspect_prompt.py 4 9 15     # só os exemplos 4, 9 e 15 (1-based)
"""

import os
import sys
import time
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, get_llm
from evaluate import load_dataset_from_jsonl
from metrics import evaluate_f1_score

load_dotenv()

YAML = "prompts/bug_to_user_story_v2.yml"
DATASET = "datasets/bug_to_user_story.jsonl"


def build_chain():
    """Monta a chain a partir do YAML LOCAL (mesmo template que vai pro Hub)."""
    data = load_yaml(YAML)["bug_to_user_story_v2"]
    template = ChatPromptTemplate.from_messages([
        ("system", data["system_prompt"]),
        ("user", data["user_prompt"]),
    ])
    return template | get_llm(temperature=0)


def main():
    chain = build_chain()
    examples = load_dataset_from_jsonl(DATASET)

    # Seleção via argumentos (1-based); sem argumentos = todos os exemplos
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    indices = [i - 1 for i in args] if args else range(len(examples))

    for i in indices:
        example = examples[i]
        bug = example["inputs"]["bug_report"]
        reference = example["outputs"]["reference"]
        complexity = example.get("metadata", {}).get("complexity", "?")

        answer = chain.invoke({"bug_report": bug}).content
        f1 = evaluate_f1_score(bug, answer, reference)

        print("=" * 80)
        print(f"EXEMPLO #{i + 1}  |  complexidade: {complexity}")
        print(f"F1: {f1['score']:.2f}  (precision={f1['precision']:.2f}  recall={f1['recall']:.2f})")
        print("-" * 80)
        print("BUG REPORT:\n" + bug)
        print("-" * 80)
        print("RESPOSTA DO MODELO:\n" + answer)
        print("-" * 80)
        print("REFERÊNCIA (esperada):\n" + reference)
        print("-" * 80)
        print("REASONING DO JUIZ (F1):\n" + f1.get("reasoning", ""))
        print()

        time.sleep(1)  # respiro pequeno para o rate limit (TPM) do juiz

    return 0


if __name__ == "__main__":
    sys.exit(main())
