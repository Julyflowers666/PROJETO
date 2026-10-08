# Pantanal Guardian AI 0.2 - guia para os alunos

## 1. O que mudou nesta versão

A versão 0.1 começava com uma base já organizada. A versão 0.2 também prepara os arquivos brutos:

```text
INMET bruto + INPE bruto
          ↓
limpeza e padronização
          ↓
uma linha para cada dia
          ↓
criação do alvo foco_dia_seguinte
          ↓
regressão logística e Random Forest
          ↓
métricas e previsão experimental
```

O arquivo principal continua sendo:

```text
src/pantanal_guardian.py
```

O novo arquivo `src/preparar_bases.py` cuida da limpeza e da integração.

## 2. Primeiro teste recomendado

Abra a pasta do projeto no VS Code, ative o ambiente virtual e execute:

```powershell
python src/pantanal_guardian.py demo-completo
```

Esse comando realiza cinco ações:

1. cria um CSV bruto parecido com o INMET;
2. cria um CSV bruto parecido com o INPE;
3. transforma as medições em registros diários;
4. cria a coluna `foco_dia_seguinte`;
5. treina e compara os modelos.

Tudo é simulado e deve ser identificado como atividade didática.

## 3. Entendendo o arquivo do INMET

O INMET pode apresentar várias medições no mesmo dia. O programa procura:

- data;
- temperatura máxima;
- umidade mínima;
- precipitação.

Depois, agrupa as linhas pela data:

```python
diario = trabalho.groupby("data").agg(...)
```

Para cada dia, ele guarda a maior temperatura, a menor umidade e a soma da chuva.

## 4. Cálculo dos dias sem chuva

Se a precipitação do dia for menor que 1 mm, o contador aumenta. Quando a chuva chega a 1 mm ou mais, ele volta a zero.

Um dado ausente não é tratado como dia seco. A contagem é interrompida, pois não podemos afirmar que não choveu quando a medição está faltando.

## 5. Entendendo o arquivo do INPE

Cada linha pode representar um foco detectado. O programa:

1. converte a data;
2. aplica o filtro territorial solicitado;
3. remove registros repetidos;
4. conta quantos focos aparecem em cada dia.

Para afirmar que o foco pertence à Serra do Amolar, use o polígono GeoJSON validado. O filtro apenas por município representa Corumbá, uma área muito maior.

## 6. Criação do alvo

O modelo usa as condições de hoje para tentar estimar amanhã:

```python
base["foco_dia_seguinte"] = base["foco_no_dia"].shift(-1)
```

Exemplo:

| Data | Condições meteorológicas | Foco amanhã |
|---|---|---:|
| 10/08 | 36 °C, 25%, 8 dias secos | 1 |
| 11/08 | 35 °C, 30%, 9 dias secos | 0 |

## 7. Preparação com os arquivos oficiais

Coloque os arquivos na pasta `dados/originais` e execute:

```powershell
python src/pantanal_guardian.py preparar `
  --inmet dados/originais/inmet.csv `
  --inpe dados/originais/inpe.csv `
  --saida dados/base_tratada.csv `
  --municipio "Corumbá"
```

O comando acima filtra o município, mas ainda não delimita a Serra do Amolar.

Com um GeoJSON validado e o GeoPandas instalado:

```powershell
pip install geopandas

python src/pantanal_guardian.py preparar `
  --inmet dados/originais/inmet.csv `
  --inpe dados/originais/inpe.csv `
  --saida dados/base_tratada.csv `
  --geojson dados/serra_do_amolar.geojson
```

## 8. Conferência obrigatória

Depois da preparação, abra:

```text
dados/base_tratada_relatorio.json
```

Confira:

- período inicial e final;
- quantidade de dias;
- valores ausentes;
- filtros territoriais;
- quantidade de dias com foco;
- percentual de alvos positivos.

Registre essas informações no diário de bordo.

## 9. Treinamento

Depois de conferir a base:

```powershell
python src/pantanal_guardian.py treinar --arquivo dados/base_tratada.csv
```

O programa utiliza os primeiros 70% para treinamento, os 15% seguintes para validação e os 15% finais para teste. Os registros não são embaralhados.

## 10. Previsão pelo terminal

```powershell
python src/pantanal_guardian.py prever `
  --temperatura 36.8 `
  --umidade 24 `
  --dias-sem-chuva 12
```

As faixas de risco são provisórias. Elas deverão ser revistas depois da validação com dados oficiais.

## 11. O que explicar na feira

- O Figma mostra a interface planejada.
- O Python executa o núcleo de análise.
- Os testes simulados comprovam o funcionamento técnico.
- Somente os testes com as bases oficiais poderão produzir resultados científicos sobre a área estudada.
- A ferramenta é experimental e não substitui alertas oficiais.
