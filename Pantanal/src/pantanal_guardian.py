"""Pantanal Guardian AI - versão didática 0.2.

Comandos disponíveis:

* ``demo``: testa apenas os modelos com uma base já tratada e simulada;
* ``demo-completo``: cria arquivos brutos simulados, prepara e treina;
* ``preparar``: transforma arquivos do INMET e INPE em base diária;
* ``treinar``: compara regressão logística e Random Forest;
* ``prever``: usa o modelo escolhido em uma nova situação.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preparar_bases import gerar_brutos_demonstracao, preparar_base_oficial

COLUNAS_X = ["temperatura_maxima", "umidade_minima", "dias_sem_chuva"]
COLUNA_DATA = "data"
COLUNA_ALVO = "foco_dia_seguinte"
LIMIAR = 0.50


def validar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o formato mínimo e devolve os registros em ordem cronológica."""
    obrigatorias = [COLUNA_DATA, *COLUNAS_X, COLUNA_ALVO]
    faltantes = [c for c in obrigatorias if c not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes: {', '.join(faltantes)}")

    base = df[obrigatorias].copy()
    base[COLUNA_DATA] = pd.to_datetime(base[COLUNA_DATA], errors="coerce")
    if base[COLUNA_DATA].isna().any():
        raise ValueError("Existem datas inválidas na base.")

    for coluna in COLUNAS_X:
        base[coluna] = pd.to_numeric(base[coluna], errors="coerce")

    alvo = pd.to_numeric(base[COLUNA_ALVO], errors="coerce")
    if alvo.isna().any() or not set(alvo.unique()).issubset({0, 1}):
        raise ValueError("foco_dia_seguinte deve conter somente 0 ou 1.")
    base[COLUNA_ALVO] = alvo.astype(int)

    base = base.sort_values(COLUNA_DATA).drop_duplicates(COLUNA_DATA, keep="last")
    if len(base) < 30:
        raise ValueError("A base precisa de pelo menos 30 dias para esta demonstração.")
    return base.reset_index(drop=True)


def dividir_cronologicamente(df: pd.DataFrame):
    """Usa 70% para treino, 15% para validação e 15% para teste, sem embaralhar."""
    fim_treino = int(len(df) * 0.70)
    fim_validacao = int(len(df) * 0.85)
    partes = df.iloc[:fim_treino], df.iloc[fim_treino:fim_validacao], df.iloc[fim_validacao:]
    for nome, parte in zip(("treino", "validação", "teste"), partes):
        if parte[COLUNA_ALVO].nunique() < 2:
            raise ValueError(f"O conjunto de {nome} precisa conter dias das classes 0 e 1.")
    return partes


def criar_modelos() -> dict[str, Pipeline]:
    """Cria dois pipelines; a mediana é aprendida somente no conjunto de treino."""
    preparo_logistico = ColumnTransformer([
        ("numericas", Pipeline([
            ("mediana", SimpleImputer(strategy="median")),
            ("escala", StandardScaler()),
        ]), COLUNAS_X)
    ])
    preparo_floresta = ColumnTransformer([
        ("numericas", SimpleImputer(strategy="median"), COLUNAS_X)
    ])
    return {
        "regressao_logistica": Pipeline([
            ("preparo", preparo_logistico),
            ("modelo", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
        ]),
        "random_forest": Pipeline([
            ("preparo", preparo_floresta),
            ("modelo", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ]),
    }


def calcular_metricas(y_real, probabilidades, limiar=LIMIAR) -> dict[str, float]:
    """Calcula métricas, incluindo a taxa de falsos alertas FP/(FP+TN)."""
    previstos = (probabilidades >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, previstos, labels=[0, 1]).ravel()
    taxa_falsos_alertas = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "acuracia": accuracy_score(y_real, previstos),
        "precisao": precision_score(y_real, previstos, zero_division=0),
        "sensibilidade": recall_score(y_real, previstos, zero_division=0),
        "f1_score": f1_score(y_real, previstos, zero_division=0),
        "roc_auc": roc_auc_score(y_real, probabilidades),
        "taxa_falsos_alertas": taxa_falsos_alertas,
        "verdadeiros_negativos": int(tn), "falsos_positivos": int(fp),
        "falsos_negativos": int(fn), "verdadeiros_positivos": int(tp),
    }


def salvar_matriz(y_real, probabilidades, nome, pasta, limiar=LIMIAR):
    previstos = (probabilidades >= limiar).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_real, previstos, display_labels=["sem foco", "com foco"], cmap="YlOrRd")
    plt.title(f"Matriz de confusão — {nome.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(pasta / f"matriz_confusao_{nome}.png", dpi=160)
    plt.close()


def treinar(arquivo: Path, pasta_resultados: Path) -> None:
    base = validar_base(pd.read_csv(arquivo))
    treino, validacao, teste = dividir_cronologicamente(base)
    X_treino, y_treino = treino[COLUNAS_X], treino[COLUNA_ALVO]

    modelos = criar_modelos()
    linhas = []
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        for conjunto_nome, conjunto in (("validacao", validacao), ("teste", teste)):
            probs = modelo.predict_proba(conjunto[COLUNAS_X])[:, 1]
            linhas.append({"modelo": nome, "conjunto": conjunto_nome, **calcular_metricas(conjunto[COLUNA_ALVO], probs)})

    metricas = pd.DataFrame(linhas)
    melhor_nome = metricas.query("conjunto == 'validacao'").sort_values("f1_score", ascending=False).iloc[0]["modelo"]

    # Depois da escolha, o melhor algoritmo reaprende usando treino + validação.
    treino_final = pd.concat([treino, validacao], ignore_index=True)
    melhor = criar_modelos()[melhor_nome]
    melhor.fit(treino_final[COLUNAS_X], treino_final[COLUNA_ALVO])
    probs_teste = melhor.predict_proba(teste[COLUNAS_X])[:, 1]

    pasta_resultados.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(pasta_resultados / "metricas.csv", index=False, float_format="%.4f")
    saida = teste[[COLUNA_DATA, *COLUNAS_X, COLUNA_ALVO]].copy()
    saida["probabilidade"] = probs_teste
    saida["previsao"] = (probs_teste >= LIMIAR).astype(int)
    saida.to_csv(pasta_resultados / "previsoes_teste.csv", index=False, float_format="%.4f")
    joblib.dump(melhor, pasta_resultados / "modelo_escolhido.joblib")
    salvar_matriz(teste[COLUNA_ALVO], probs_teste, melhor_nome, pasta_resultados)

    configuracao = {
        "arquivo": str(arquivo), "registros": len(base), "variaveis": COLUNAS_X,
        "alvo": COLUNA_ALVO, "divisao": {"treino": "70%", "validacao": "15%", "teste": "15%"},
        "limiar": LIMIAR, "modelo_escolhido": melhor_nome,
        "periodo": {"inicio": str(base[COLUNA_DATA].min().date()), "fim": str(base[COLUNA_DATA].max().date())},
    }
    (pasta_resultados / "configuracao.json").write_text(json.dumps(configuracao, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nTreinamento concluído.")
    print(metricas[["modelo", "conjunto", "acuracia", "precisao", "sensibilidade", "f1_score", "taxa_falsos_alertas"]].to_string(index=False))
    print(f"\nModelo escolhido pela validação: {melhor_nome}")
    print(f"Resultados salvos em: {pasta_resultados.resolve()}")


def classificar_risco(probabilidade: float) -> str:
    """Faixas provisórias; deverão ser revistas após validação científica."""
    if probabilidade < 0.25:
        return "baixo"
    if probabilidade < 0.50:
        return "moderado"
    if probabilidade < 0.75:
        return "alto"
    return "crítico"


def prever(pasta_resultados: Path, temperatura: float, umidade: float, dias_sem_chuva: int):
    caminho = pasta_resultados / "modelo_escolhido.joblib"
    if not caminho.exists():
        raise FileNotFoundError("Treine o modelo antes de solicitar uma previsão.")
    modelo = joblib.load(caminho)
    entrada = pd.DataFrame([[temperatura, umidade, dias_sem_chuva]], columns=COLUNAS_X)
    prob = float(modelo.predict_proba(entrada)[0, 1])
    print(f"Temperatura máxima: {temperatura:.1f} °C")
    print(f"Umidade mínima: {umidade:.1f}%")
    print(f"Dias consecutivos sem chuva: {dias_sem_chuva}")
    print(f"Probabilidade experimental: {prob:.1%}")
    print(f"Classificação provisória: risco {classificar_risco(prob)}")
    print("AVISO: resultado experimental; não substitui alertas oficiais.")


def gerar_base_demonstracao(caminho: Path, quantidade=1800) -> None:
    """Gera dados artificiais somente para testar o funcionamento do código."""
    rng = np.random.default_rng(42)
    datas = pd.date_range("2020-01-01", periods=quantidade, freq="D")
    ciclo = np.sin(2 * np.pi * datas.dayofyear.to_numpy() / 365.25)
    temperatura = 31 + 5 * ciclo + rng.normal(0, 2, quantidade)
    umidade = np.clip(58 - 19 * ciclo + rng.normal(0, 8, quantidade), 12, 98)
    chuva = rng.random(quantidade) < np.clip((umidade - 25) / 100, 0.05, 0.55)
    dias_secos, contador = [], 0
    for choveu in chuva:
        contador = 0 if choveu else contador + 1
        dias_secos.append(contador)
    logit = -5.6 + 0.085 * temperatura - 0.042 * umidade + 0.075 * np.array(dias_secos)
    prob = 1 / (1 + np.exp(-logit))
    alvo = rng.binomial(1, np.clip(prob, 0.01, 0.80))
    df = pd.DataFrame({COLUNA_DATA: datas, "temperatura_maxima": temperatura.round(1), "umidade_minima": umidade.round(1), "dias_sem_chuva": dias_secos, COLUNA_ALVO: alvo})
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=False)
    print(f"Base SIMULADA criada em: {caminho.resolve()}")


def main():
    raiz = Path(__file__).resolve().parents[1]
    resultados = raiz / "resultados"
    parser = argparse.ArgumentParser(description="Pantanal Guardian AI - versão experimental 0.2")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("demo", help="gera dados simulados e testa o fluxo completo")
    sub.add_parser(
        "demo-completo",
        help="gera dois arquivos brutos simulados, prepara a base e treina os modelos",
    )

    # O comando preparar é usado quando já temos os downloads do INMET e INPE.
    p_preparo = sub.add_parser("preparar", help="integra os arquivos brutos do INMET e INPE")
    p_preparo.add_argument("--inmet", type=Path, required=True, help="CSV bruto do INMET")
    p_preparo.add_argument("--inpe", type=Path, required=True, help="CSV bruto do INPE")
    p_preparo.add_argument("--saida", type=Path, required=True, help="nome do CSV tratado")
    p_preparo.add_argument("--municipio", help="filtro opcional, por exemplo: Corumbá")
    p_preparo.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="caixa geográfica opcional; confirme as coordenadas antes de usar",
    )
    p_preparo.add_argument("--geojson", type=Path, help="polígono opcional da área de estudo")
    p_treino = sub.add_parser("treinar", help="treina usando uma base tratada")
    p_treino.add_argument("--arquivo", type=Path, required=True)
    p_prev = sub.add_parser("prever", help="faz uma previsão com o modelo salvo")
    p_prev.add_argument("--temperatura", type=float, required=True)
    p_prev.add_argument("--umidade", type=float, required=True)
    p_prev.add_argument("--dias-sem-chuva", type=int, required=True)
    args = parser.parse_args()

    if args.comando == "demo":
        arquivo = raiz / "dados" / "base_demonstracao.csv"
        gerar_base_demonstracao(arquivo)
        treinar(arquivo, resultados)
    elif args.comando == "demo-completo":
        # Esta opção permite aos alunos observar o caminho completo antes de
        # trabalhar com os downloads oficiais.
        pasta_brutos = raiz / "dados" / "brutos_demo"
        inmet, inpe = gerar_brutos_demonstracao(pasta_brutos)
        arquivo = raiz / "dados" / "base_tratada_demo.csv"
        preparar_base_oficial(inmet, inpe, arquivo, municipio="Corumbá")
        treinar(arquivo, resultados)
    elif args.comando == "preparar":
        bbox = tuple(args.bbox) if args.bbox else None
        preparar_base_oficial(
            args.inmet,
            args.inpe,
            args.saida,
            municipio=args.municipio,
            bbox=bbox,
            geojson=args.geojson,
        )
    elif args.comando == "treinar":
        treinar(args.arquivo, resultados)
    else:
        prever(resultados, args.temperatura, args.umidade, args.dias_sem_chuva)


if __name__ == "__main__":
    main()
