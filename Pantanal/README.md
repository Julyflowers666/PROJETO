# Pantanal Guardian AI - versão 0.2

Esta versão prepara arquivos brutos do INMET e INPE, cria a base diária e compara **regressão logística** e **Random Forest** utilizando três variáveis:

- temperatura máxima diária;
- umidade relativa mínima diária;
- quantidade de dias consecutivos sem chuva.

A coluna `foco_dia_seguinte` é o alvo: `1` quando houve ao menos um foco no dia seguinte e `0` quando não houve.

> Atenção: a base gerada pelo comando `demo` é **simulada e apenas didática**. As métricas produzidas com ela não podem ser apresentadas como resultados científicos da Serra do Amolar.

## 1. Preparar o ambiente

No terminal, dentro desta pasta:

```bash
python -m venv .venv
```

No Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Executar a demonstração completa, desde os arquivos brutos

```bash
python src/pantanal_guardian.py demo-completo
```

O programa cria arquivos brutos simulados, prepara `dados/base_tratada_demo.csv`, treina os dois algoritmos e salva resultados em `resultados/`.

Para estudar cada etapa com explicações, leia `GUIA_PARA_ALUNOS.md`.

## 3. Formato da base tratada oficial

O arquivo final, depois do cruzamento INMET–INPE, deverá possuir:

```csv
data,temperatura_maxima,umidade_minima,dias_sem_chuva,foco_dia_seguinte
2024-08-10,35.4,28,8,0
2024-08-11,36.7,23,9,1
```

Regras:

- uma linha representa um dia;
- `data` deve usar `AAAA-MM-DD`;
- as três variáveis devem ser numéricas;
- o alvo deve conter somente `0` ou `1`;
- os registros devem estar em ordem cronológica;
- valores meteorológicos ausentes são preenchidos com a mediana aprendida apenas no treinamento.

## 4. Treinar com uma base tratada

```bash
python src/pantanal_guardian.py treinar --arquivo dados/base_tratada.csv
```

Divisão cronológica adotada na versão 0.1:

- 70% dos primeiros registros: treinamento;
- 15% seguintes: validação;
- 15% finais: teste.

Os modelos usam `class_weight="balanced"` para reduzir o efeito do desbalanceamento entre dias com e sem foco.

## 5. Fazer uma previsão experimental

Depois do treinamento:

```bash
python src/pantanal_guardian.py prever --temperatura 36.8 --umidade 24 --dias-sem-chuva 12
```

As faixas baixo, moderado, alto e crítico são provisórias e não representam alertas oficiais.

## 6. Arquivos gerados

- `resultados/metricas.csv`: comparação dos algoritmos;
- `resultados/previsoes_teste.csv`: valores reais, probabilidades e previsões;
- `resultados/matriz_confusao_*.png`: acertos e erros de cada modelo;
- `resultados/modelo_escolhido.joblib`: modelo com melhor F1 na validação;
- `resultados/configuracao.json`: variáveis, divisão e limiar utilizados.

## 7. Testes automáticos

```bash
python -m unittest discover -s tests -v
```

## Próxima etapa científica

Para obter resultados reais, ainda será necessário construir `base_tratada.csv` com os arquivos oficiais do INMET e do INPE, filtrar os focos pela área da Serra do Amolar e registrar no diário de bordo todas as decisões de limpeza.
