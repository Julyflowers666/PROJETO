"""Preparação das bases oficiais do Pantanal Guardian AI.

Este módulo transforma dois arquivos brutos em uma tabela diária:

1. INMET: temperatura, umidade e precipitação medidas ao longo do dia;
2. INPE: registros de focos de calor, com data e localização.

O resultado pode ser usado diretamente pelo arquivo ``pantanal_guardian.py``.
Os nomes das colunas públicas podem variar entre downloads. Por isso, o código
normaliza os textos e procura os nomes mais comuns antes de iniciar o cálculo.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


def normalizar_texto(valor: object) -> str:
    """Remove acentos e símbolos para facilitar a comparação de nomes.

    Exemplo: ``UMIDADE RELATIVA DO AR, HORARIA (%)`` vira
    ``umidade relativa do ar horaria``.
    """
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(letra for letra in texto if not unicodedata.combining(letra))
    texto = texto.lower().strip()
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def detectar_cabecalho_e_separador(caminho: Path) -> tuple[int, str]:
    """Descobre onde começa o cabeçalho e qual separador foi utilizado.

    Alguns arquivos anuais do INMET possuem linhas de informações da estação
    antes da tabela. O programa procura, nas primeiras 40 linhas, a primeira
    linha que começa com uma coluna de data.
    """
    texto = caminho.read_text(encoding="utf-8-sig", errors="replace")
    linhas = texto.splitlines()
    for indice, linha in enumerate(linhas[:40]):
        normalizada = normalizar_texto(linha)
        if normalizada.startswith("data") or "data hora gmt" in normalizada:
            # O CSV do INMET normalmente usa ponto e vírgula; o INPE, vírgula.
            contagens = {";": linha.count(";"), ",": linha.count(","), "\t": linha.count("\t")}
            separador = max(contagens, key=contagens.get)
            if contagens[separador] > 0:
                return indice, separador

    # Se não houver metadados, usamos o analisador padrão do Python.
    amostra = "\n".join(linhas[:10])
    try:
        separador = csv.Sniffer().sniff(amostra, delimiters=";,\t").delimiter
    except csv.Error:
        separador = ","
    return 0, separador


def ler_csv_flexivel(caminho: Path) -> pd.DataFrame:
    """Lê CSV em UTF-8 ou Latin-1 e aceita vírgula, ponto e vírgula ou tabulação."""
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    pular, separador = detectar_cabecalho_e_separador(caminho)
    ultimo_erro = None
    for codificacao in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                caminho,
                sep=separador,
                skiprows=pular,
                encoding=codificacao,
                dtype=str,
                low_memory=False,
            )
        except UnicodeDecodeError as erro:
            ultimo_erro = erro
    raise ValueError(f"Não foi possível ler {caminho}: {ultimo_erro}")


def mapa_colunas(df: pd.DataFrame) -> dict[str, str]:
    """Relaciona o nome normalizado ao nome original presente no arquivo."""
    return {normalizar_texto(coluna): coluna for coluna in df.columns}


def localizar_coluna(df: pd.DataFrame, alternativas: list[str]) -> str:
    """Encontra uma coluna por nome exato ou por trecho do nome normalizado."""
    mapa = mapa_colunas(df)
    alternativas_norm = [normalizar_texto(item) for item in alternativas]

    # Primeiro tentamos igualdade exata para evitar escolher uma coluna parecida.
    for alternativa in alternativas_norm:
        if alternativa in mapa:
            return mapa[alternativa]

    # Depois aceitamos nomes maiores que contenham a expressão procurada.
    for alternativa in alternativas_norm:
        for nome_norm, nome_original in mapa.items():
            if alternativa in nome_norm:
                return nome_original

    disponiveis = ", ".join(str(c) for c in df.columns)
    raise ValueError(
        f"Nenhuma coluna corresponde a {alternativas}. "
        f"Colunas encontradas no arquivo: {disponiveis}"
    )


def converter_numero(serie: pd.Series) -> pd.Series:
    """Converte números brasileiros, como ``35,7``, para valores numéricos."""
    texto = serie.astype("string").str.strip()
    texto = texto.replace({"": pd.NA, "null": pd.NA, "NaN": pd.NA, "-9999": pd.NA})
    texto = texto.str.replace(",", ".", regex=False)
    return pd.to_numeric(texto, errors="coerce")


def converter_data(serie: pd.Series) -> pd.Series:
    """Converte datas do INMET/INPE, aceitando data isolada ou data e hora."""
    # dayfirst=True aceita o formato brasileiro; anos no início também funcionam.
    # ``format='mixed'`` é importante: sem ele, uma data como 2026/01/02 pode
    # ser interpretada como 1 de fevereiro quando usamos ``dayfirst=True``.
    return pd.to_datetime(
        serie.astype("string").str.slice(0, 19),
        errors="coerce",
        format="mixed",
        dayfirst=True,
    )


def calcular_dias_sem_chuva(precipitacao: pd.Series, limite_mm: float = 1.0) -> pd.Series:
    """Conta dias seguidos com precipitação inferior a 1 mm.

    Um dia sem medição não é considerado dia seco. Nesse caso, o resultado fica
    ausente e a contagem recomeça, evitando transformar falta de dado em falta
    de chuva.
    """
    resultado: list[float] = []
    contador = 0
    for valor in precipitacao:
        if pd.isna(valor):
            resultado.append(np.nan)
            contador = 0
        elif float(valor) < limite_mm:
            contador += 1
            resultado.append(float(contador))
        else:
            contador = 0
            resultado.append(0.0)
    return pd.Series(resultado, index=precipitacao.index, dtype="float64")


def preparar_inmet(df_bruto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Agrupa as medições do INMET e produz um registro por dia."""
    col_data = localizar_coluna(df_bruto, ["data", "data medicao"])
    col_temp = localizar_coluna(df_bruto, ["temperatura maxima", "temp max"])
    col_umidade = localizar_coluna(
        df_bruto,
        ["umidade relativa minima", "umidade minima", "umidade relativa do ar horaria", "umidade relativa"],
    )
    col_chuva = localizar_coluna(df_bruto, ["precipitacao total", "precipitacao"])

    trabalho = pd.DataFrame({
        "data": converter_data(df_bruto[col_data]).dt.normalize(),
        "temperatura": converter_numero(df_bruto[col_temp]),
        "umidade": converter_numero(df_bruto[col_umidade]),
        "precipitacao": converter_numero(df_bruto[col_chuva]),
    }).dropna(subset=["data"])

    # Uma mesma data pode aparecer várias vezes, pois há medições horárias.
    diario = trabalho.groupby("data", as_index=False).agg(
        temperatura_maxima=("temperatura", "max"),
        umidade_minima=("umidade", "min"),
        precipitacao_total=("precipitacao", lambda s: s.sum(min_count=1)),
    )

    # Criamos todos os dias do intervalo para deixar lacunas visíveis.
    intervalo = pd.date_range(diario["data"].min(), diario["data"].max(), freq="D")
    diario = diario.set_index("data").reindex(intervalo).rename_axis("data").reset_index()
    diario["dias_sem_chuva"] = calcular_dias_sem_chuva(diario["precipitacao_total"])

    relatorio = {
        "linhas_brutas": int(len(df_bruto)),
        "dias_gerados": int(len(diario)),
        "inicio": str(diario["data"].min().date()),
        "fim": str(diario["data"].max().date()),
        "ausentes_temperatura": int(diario["temperatura_maxima"].isna().sum()),
        "ausentes_umidade": int(diario["umidade_minima"].isna().sum()),
        "ausentes_precipitacao": int(diario["precipitacao_total"].isna().sum()),
        "colunas_utilizadas": {
            "data": col_data,
            "temperatura": col_temp,
            "umidade": col_umidade,
            "precipitacao": col_chuva,
        },
    }
    return diario, relatorio


def filtrar_inpe(
    df: pd.DataFrame,
    municipio: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    geojson: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Padroniza os focos e aplica os filtros territoriais solicitados.

    ``bbox`` usa a ordem: longitude mínima, latitude mínima, longitude máxima,
    latitude máxima. Um arquivo GeoJSON é preferível porque acompanha melhor o
    contorno real da área de estudo.
    """
    col_data = localizar_coluna(df, ["data hora gmt", "data hora", "data"])
    trabalho = df.copy()
    trabalho["data"] = converter_data(trabalho[col_data]).dt.normalize()
    trabalho = trabalho.dropna(subset=["data"])
    filtros = []

    if municipio:
        col_municipio = localizar_coluna(trabalho, ["municipio"])
        alvo = normalizar_texto(municipio)
        trabalho = trabalho[
            trabalho[col_municipio].map(normalizar_texto) == alvo
        ]
        filtros.append(f"município={municipio}")

    # Latitude e longitude são necessárias para caixa geográfica ou GeoJSON.
    if bbox is not None or geojson is not None:
        col_lat = localizar_coluna(trabalho, ["latitude", "lat"])
        col_lon = localizar_coluna(trabalho, ["longitude", "lon"])
        trabalho["latitude_num"] = converter_numero(trabalho[col_lat])
        trabalho["longitude_num"] = converter_numero(trabalho[col_lon])
        trabalho = trabalho.dropna(subset=["latitude_num", "longitude_num"])

    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        trabalho = trabalho[
            trabalho["longitude_num"].between(lon_min, lon_max)
            & trabalho["latitude_num"].between(lat_min, lat_max)
        ]
        filtros.append(f"bbox={bbox}")

    if geojson is not None:
        try:
            import geopandas as gpd
        except ImportError as erro:
            raise ImportError(
                "Para filtrar por GeoJSON, instale: pip install geopandas"
            ) from erro
        if not geojson.exists():
            raise FileNotFoundError(f"GeoJSON não encontrado: {geojson}")
        area = gpd.read_file(geojson).to_crs("EPSG:4326")
        pontos = gpd.GeoDataFrame(
            trabalho,
            geometry=gpd.points_from_xy(trabalho["longitude_num"], trabalho["latitude_num"]),
            crs="EPSG:4326",
        )
        trabalho = gpd.sjoin(pontos, area[["geometry"]], predicate="within", how="inner")
        filtros.append(f"geojson={geojson}")

    # Retiramos duplicações exatas para não contar duas vezes o mesmo registro.
    chaves = ["data"]
    for coluna in ("latitude_num", "longitude_num"):
        if coluna in trabalho.columns:
            chaves.append(coluna)
    trabalho = trabalho.drop_duplicates(subset=chaves)

    relatorio = {
        "linhas_brutas": int(len(df)),
        "focos_apos_filtros": int(len(trabalho)),
        "dias_com_foco": int(trabalho["data"].nunique()),
        "filtros": filtros or ["nenhum filtro territorial aplicado"],
        "coluna_data": col_data,
    }
    return trabalho, relatorio


def integrar_bases(inmet_diario: pd.DataFrame, focos: pd.DataFrame) -> pd.DataFrame:
    """Une meteorologia e focos e cria o alvo das próximas 24 horas."""
    focos_dia = (
        focos.groupby("data").size().rename("quantidade_focos").reset_index()
        if not focos.empty
        else pd.DataFrame(columns=["data", "quantidade_focos"])
    )
    base = inmet_diario.merge(focos_dia, on="data", how="left")
    base["quantidade_focos"] = base["quantidade_focos"].fillna(0).astype(int)
    base["foco_no_dia"] = (base["quantidade_focos"] > 0).astype(int)

    # Em cada linha de hoje, colocamos a ocorrência observada amanhã.
    base["foco_dia_seguinte"] = base["foco_no_dia"].shift(-1)
    base = base.iloc[:-1].copy()  # o último dia não possui amanhã conhecido
    base["foco_dia_seguinte"] = base["foco_dia_seguinte"].astype(int)
    return base[
        [
            "data", "temperatura_maxima", "umidade_minima", "precipitacao_total",
            "dias_sem_chuva", "quantidade_focos", "foco_no_dia", "foco_dia_seguinte",
        ]
    ]


def preparar_base_oficial(
    arquivo_inmet: Path,
    arquivo_inpe: Path,
    saida: Path,
    municipio: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    geojson: Path | None = None,
) -> pd.DataFrame:
    """Executa a preparação completa e salva CSV e relatório de qualidade."""
    bruto_inmet = ler_csv_flexivel(arquivo_inmet)
    bruto_inpe = ler_csv_flexivel(arquivo_inpe)
    diario, rel_inmet = preparar_inmet(bruto_inmet)
    focos, rel_inpe = filtrar_inpe(bruto_inpe, municipio, bbox, geojson)
    base = integrar_bases(diario, focos)

    saida.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(saida, index=False, date_format="%Y-%m-%d", float_format="%.2f")

    relatorio = {
        "aviso": "Confira os filtros territoriais antes de tratar a base como Serra do Amolar.",
        "inmet": rel_inmet,
        "inpe": rel_inpe,
        "base_final": {
            "linhas": int(len(base)),
            "dias_com_alvo_positivo": int(base["foco_dia_seguinte"].sum()),
            "percentual_positivo": round(float(base["foco_dia_seguinte"].mean() * 100), 4),
        },
    }
    relatorio_path = saida.with_name(saida.stem + "_relatorio.json")
    relatorio_path.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Base tratada salva em: {saida.resolve()}")
    print(f"Relatório de qualidade: {relatorio_path.resolve()}")
    print(f"Dias na base: {len(base)} | Alvos positivos: {int(base['foco_dia_seguinte'].sum())}")
    return base


def gerar_brutos_demonstracao(pasta: Path, quantidade_dias: int = 1200) -> tuple[Path, Path]:
    """Cria arquivos parecidos com os portais apenas para a aula prática.

    Os registros são artificiais. Eles permitem estudar a preparação sem
    confundir os resultados com observações reais do Pantanal.
    """
    rng = np.random.default_rng(2026)
    datas = pd.date_range("2021-01-01", periods=quantidade_dias, freq="D")
    linhas_inmet = []
    focos_inpe = []
    dias_secos = 0

    for data in datas:
        ciclo = np.sin(2 * np.pi * data.dayofyear / 365.25)
        temp_base = 31 + 5 * ciclo + rng.normal(0, 1.5)
        umid_base = np.clip(58 - 18 * ciclo + rng.normal(0, 6), 15, 95)
        choveu = rng.random() < np.clip((umid_base - 20) / 110, 0.08, 0.55)
        chuva_total = float(rng.gamma(2, 4)) if choveu else 0.0
        dias_secos = 0 if chuva_total >= 1 else dias_secos + 1

        # Quatro medições no dia simulam o formato horário do INMET.
        for hora in (0, 6, 12, 18):
            linhas_inmet.append({
                "Data": data.strftime("%Y/%m/%d"),
                "Hora UTC": f"{hora:02d}00 UTC",
                "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)": chuva_total / 4,
                "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)": temp_base + rng.normal(0, 1),
                "UMIDADE RELATIVA DO AR, HORÁRIA (%)": umid_base + rng.normal(0, 3),
            })

        # A chance artificial aumenta em dias quentes, secos e pouco úmidos.
        # O ponto inicial foi escolhido para gerar eventos suficientes em uma
        # aula. Ele não representa uma equação científica do Pantanal.
        logit = -3.3 + 0.08 * temp_base - 0.035 * umid_base + 0.065 * dias_secos
        chance = float(1 / (1 + np.exp(-logit)))
        if rng.random() < np.clip(chance, 0.015, 0.65):
            focos_inpe.append({
                "data_hora_gmt": f"{data:%Y-%m-%d} 16:30:00",
                "latitude": -18.10 + rng.normal(0, 0.08),
                "longitude": -57.50 + rng.normal(0, 0.08),
                "municipio": "Corumbá",
                "satelite": "DEMO",
            })

    pasta.mkdir(parents=True, exist_ok=True)
    inmet_path = pasta / "inmet_bruto_demo.csv"
    inpe_path = pasta / "inpe_bruto_demo.csv"
    pd.DataFrame(linhas_inmet).to_csv(inmet_path, sep=";", index=False, decimal=",")
    pd.DataFrame(focos_inpe).to_csv(inpe_path, index=False)
    print("Arquivos brutos SIMULADOS criados para a aula:")
    print(f"- {inmet_path.resolve()}")
    print(f"- {inpe_path.resolve()}")
    return inmet_path, inpe_path
