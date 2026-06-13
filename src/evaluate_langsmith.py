"""
Avaliação dos prompts otimizados COMO EXPERIMENT no LangSmith.

Diferente do src/evaluate.py (que calcula as métricas localmente e só imprime no
terminal), este script usa `langsmith.evaluation.evaluate` para rodar um experiment
de verdade: cada exemplo vira uma execução logada no LangSmith, com as 5 métricas
como feedback e o RACIOCÍNIO do avaliador (campo `comment`) anexado.

Resultado:
- Um experiment compartilhável em "Datasets & Experiments" (entrega B do desafio).
- Tracing detalhado de cada exemplo (clicável a partir do experiment).
- Mesmo veredito aprovado/reprovado do evaluate.py (reusa display_results).

NÃO altera os arquivos "prontos" — apenas reaproveita as funções deles.
"""

import os
import sys
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

# Reaproveita a lógica já pronta (evaluate.py / metrics.py / utils.py)
from evaluate import (
    create_evaluation_dataset,
    pull_prompt_from_langsmith,
    get_llm,
    display_results,
)
from metrics import evaluate_f1_score, evaluate_clarity, evaluate_precision
from utils import check_env_vars, print_section_header

load_dotenv()


def make_target(prompt_template, llm):
    """Cria a função-alvo: roda o mesmo pipeline (prompt | llm) do evaluate.py."""
    chain = prompt_template | llm

    def target(inputs: dict) -> dict:
        return {"answer": chain.invoke(inputs).content}

    return target


# Acumula os scores por métrica para o resumo final no terminal.
# list.append é thread-safe sob o GIL, então funciona mesmo com max_concurrency > 1.
collected = {
    "f1_score": [],
    "clarity": [],
    "precision": [],
    "helpfulness": [],
    "correctness": [],
}


def metrics_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    Avaliador único que calcula as 5 métricas para um exemplo:
    - F1, Clarity, Precision via LLM-as-Judge (EVAL_MODEL)
    - Helpfulness e Correctness derivadas
    Retorna todos os scores de uma vez; o `comment` carrega o reasoning do juiz.
    """
    bug = inputs.get("bug_report", "")
    answer = outputs.get("answer", "")
    reference = reference_outputs.get("reference", "")

    f1 = evaluate_f1_score(bug, answer, reference)
    clarity = evaluate_clarity(bug, answer, reference)
    precision = evaluate_precision(bug, answer, reference)

    helpfulness = round((clarity["score"] + precision["score"]) / 2, 4)
    correctness = round((f1["score"] + precision["score"]) / 2, 4)

    collected["f1_score"].append(f1["score"])
    collected["clarity"].append(clarity["score"])
    collected["precision"].append(precision["score"])
    collected["helpfulness"].append(helpfulness)
    collected["correctness"].append(correctness)

    return {
        "results": [
            {"key": "f1_score", "score": f1["score"], "comment": f1.get("reasoning", "")},
            {"key": "clarity", "score": clarity["score"], "comment": clarity.get("reasoning", "")},
            {"key": "precision", "score": precision["score"], "comment": precision.get("reasoning", "")},
            {"key": "helpfulness", "score": helpfulness},
            {"key": "correctness", "score": correctness},
        ]
    }


def main():
    """
    Fluxo principal:
    1) Valida env e inicializa o client do LangSmith
    2) Garante o dataset {LANGSMITH_PROJECT}-eval (reusa create_evaluation_dataset)
    3) Puxa o prompt {username}/bug_to_user_story_v2 e monta o target
    4) Roda evaluate() -> loga o experiment no LangSmith
    5) Agrega os scores e exibe o mesmo veredito do evaluate.py + URL do experiment
    """
    print_section_header("AVALIAÇÃO COMO EXPERIMENT NO LANGSMITH")

    provider = os.getenv("LLM_PROVIDER", "openai")
    print(f"Provider: {provider}")
    print(f"Modelo Principal: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")
    print(f"Modelo de Avaliação: {os.getenv('EVAL_MODEL', 'gpt-4o')}\n")

    required_vars = ["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", "prompt-optimization-challenge-resolved")
    dataset_name = f"{project_name}-eval"

    # Garante que o dataset com os 15 exemplos existe no LangSmith
    create_evaluation_dataset(client, dataset_name, "datasets/bug_to_user_story.jsonl")

    username = os.getenv("USERNAME_LANGSMITH_HUB")
    prompt_name = f"{username}/bug_to_user_story_v2"

    prompt_template = pull_prompt_from_langsmith(prompt_name)
    target = make_target(prompt_template, get_llm())

    print(f"\n🚀 Rodando experiment para: {prompt_name}\n")

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=[metrics_evaluator],
        experiment_prefix="bug_to_user_story_v2",
        metadata={
            "prompt": prompt_name,
            "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            "eval_model": os.getenv("EVAL_MODEL", "gpt-4o"),
        },
        client=client,
        # gpt-4o (juiz) tem TPM baixo em contas Tier 1 da OpenAI (30k tokens/min).
        # Concorrência alta estoura 429; default 1 (sequencial). Suba via
        # EVAL_MAX_CONCURRENCY se sua conta tiver TPM maior (ou se o juiz for gpt-4o-mini).
        max_concurrency=int(os.getenv("EVAL_MAX_CONCURRENCY", "1")),
    )

    # Resumo no terminal (mesmo critério de aprovação do evaluate.py)
    scores = {
        key: round(sum(values) / len(values), 4)
        for key, values in collected.items()
        if values
    }

    if scores:
        passed = display_results(prompt_name, scores)
    else:
        print("⚠️  Nenhum score coletado — verifique se o prompt gerou respostas.")
        passed = False

    experiment_name = getattr(results, "experiment_name", None)
    print("\n" + "-" * 50)
    print("✅ Experiment logado no LangSmith.")
    if experiment_name:
        print(f"   Experiment: {experiment_name}")
    print("   Veja em 'Datasets & Experiments':")
    print(f"   https://smith.langchain.com/projects")
    print("   (abra o dataset -> aba Experiments para ver notas + reasoning por exemplo)")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
