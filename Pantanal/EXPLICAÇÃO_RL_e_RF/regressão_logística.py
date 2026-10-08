# ============================================================
# EXEMPLO DIDÁTICO DE REGRESSÃO LOGÍSTICA
# PANTANAL GUARDIAN AI
# ============================================================
#
# OBJETIVO:
# Criar um modelo de Regressão Logística capaz de aprender
# com condições climáticas e tentar prever se existe ou não
# risco de ocorrência de um foco de calor.
#
# IMPORTANTE:
# Os dados abaixo são APENAS exemplos para aprender como
# a Regressão Logística funciona.
# Eles NÃO são dados reais do Pantanal.
# ============================================================


# ------------------------------------------------------------
# 1. IMPORTANDO AS BIBLIOTECAS
# ------------------------------------------------------------

# NumPy é uma biblioteca muito utilizada para trabalhar
# com números e estruturas de dados.
import numpy as np

# O train_test_split serve para separar nossos dados em:
#
# - dados de treinamento
# - dados de teste
#
# Assim podemos treinar o modelo com uma parte dos dados
# e depois verificar se ele consegue fazer previsões
# utilizando dados que ele ainda não viu.
from sklearn.model_selection import train_test_split

# LogisticRegression é justamente o modelo de
# Regressão Logística que vamos utilizar.
from sklearn.linear_model import LogisticRegression

# Estas funções servem para avaliar o desempenho do modelo.
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# ------------------------------------------------------------
# 2. CRIANDO OS DADOS
# ------------------------------------------------------------

# Vamos criar alguns exemplos de condições climáticas.
#
# Cada linha representa um dia.
#
# As colunas representam:
#
# 1ª coluna = temperatura máxima
# 2ª coluna = umidade relativa mínima
# 3ª coluna = quantidade de dias consecutivos com
#             precipitação inferior a 1 mm
#
# Exemplo:
#
# [38, 25, 8]
#
# significa:
#
# temperatura = 38°C
# umidade = 25%
# dias secos = 8
#
# Em um projeto real, esses dados viriam das bases
# utilizadas pelo Pantanal Guardian AI.
X = np.array([
    [38, 25, 8],
    [36, 30, 6],
    [39, 20, 10],
    [35, 32, 7],
    [37, 28, 9],
    [34, 35, 5],
    [31, 60, 2],
    [29, 70, 1],
    [28, 75, 1],
    [30, 65, 2],
    [32, 55, 3],
    [27, 80, 0],
    [33, 50, 3],
    [26, 85, 0],
    [30, 72, 1],
    [35, 40, 5]
])


# ------------------------------------------------------------
# 3. CRIANDO OS RESULTADOS
# ------------------------------------------------------------

# Agora precisamos informar ao computador qual foi o
# resultado de cada um desses dias.
#
# Vamos utilizar:
#
# 0 = não ocorreu foco de calor
# 1 = ocorreu foco de calor
#
# Cada número corresponde à mesma linha dos dados acima.
#
# Por exemplo:
#
# X[0] = [38, 25, 8]
# y[0] = 1
#
# Isso significa que, nesse exemplo, um dia com:
#
# 38°C
# 25% de umidade
# 8 dias secos
#
# foi classificado como tendo foco de calor.
y = np.array([
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1
])


# ------------------------------------------------------------
# 4. ENTENDENDO X E Y
# ------------------------------------------------------------

# É importante entender esses dois nomes.
#
# X = características que o modelo vai analisar.
#
# No nosso exemplo:
#
# temperatura
# umidade
# dias secos
#
# y = resposta que queremos que o modelo aprenda a prever.
#
# No nosso caso:
#
# 0 = não há foco
# 1 = há foco
#
# Podemos pensar assim:
#
# X → informações sobre o dia
#
# y → resultado daquele dia


# ------------------------------------------------------------
# 5. SEPARANDO TREINAMENTO E TESTE
# ------------------------------------------------------------

# Agora vamos separar nossos dados.
#
# Uma parte será utilizada para ENSINAR o modelo.
#
# Outra parte será utilizada para TESTAR o modelo.
#
# test_size=0.25 significa que aproximadamente 25%
# dos dados serão utilizados para teste.
#
# random_state=42 serve para que a divisão seja sempre
# semelhante quando executarmos o programa novamente.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


# ------------------------------------------------------------
# 6. CRIANDO O MODELO
# ------------------------------------------------------------

# Aqui estamos criando nosso modelo de Regressão Logística.
#
# Ainda não estamos fazendo nenhuma previsão.
#
# Estamos apenas criando o "modelo vazio" que posteriormente
# será treinado utilizando nossos dados.
modelo = LogisticRegression()


# ------------------------------------------------------------
# 7. TREINANDO O MODELO
# ------------------------------------------------------------

# Agora acontece uma das partes mais importantes.
#
# O comando fit() faz o treinamento.
#
# Estamos dizendo:
#
# "Modelo, analise os dados X_treino e os resultados
# y_treino e tente aprender a relação entre eles."
#
# O modelo procura padrões nos dados.
#
# Por exemplo, ele pode perceber que:
#
# temperaturas mais altas
# +
# umidade mais baixa
# +
# muitos dias secos
#
# aparecem com mais frequência quando o resultado é 1.
modelo.fit(X_treino, y_treino)


# ------------------------------------------------------------
# 8. FAZENDO PREVISÕES
# ------------------------------------------------------------

# Agora vamos utilizar os dados de teste.
#
# Esses dados não foram utilizados diretamente para
# ensinar o modelo.
#
# O comando predict() faz uma classificação.
#
# O resultado será:
#
# 0 → não há foco
# 1 → há foco
previsoes = modelo.predict(X_teste)


# ------------------------------------------------------------
# 9. MOSTRANDO AS PREVISÕES
# ------------------------------------------------------------

print("==========================================")
print("PREVISÕES DO MODELO")
print("==========================================")

print("Resultados reais:")
print(y_teste)

print()

print("Resultados previstos:")
print(previsoes)


# ------------------------------------------------------------
# 10. CALCULANDO A ACURÁCIA
# ------------------------------------------------------------

# Agora queremos saber quantas previsões o modelo acertou.
#
# accuracy_score compara:
#
# y_teste      → resultado verdadeiro
#
# previsoes    → resultado previsto pelo modelo
#
# A acurácia representa a proporção de previsões corretas.
acuracia = accuracy_score(y_teste, previsoes)


# ------------------------------------------------------------
# 11. MOSTRANDO A ACURÁCIA
# ------------------------------------------------------------

print()
print("==========================================")
print("AVALIAÇÃO DO MODELO")
print("==========================================")

print("Acurácia:", round(acuracia * 100, 2), "%")


# ------------------------------------------------------------
# 12. RELATÓRIO COMPLETO
# ------------------------------------------------------------

# classification_report apresenta várias métricas.
#
# Entre elas:
#
# precision  → precisão
#
# recall     → sensibilidade
#
# f1-score   → equilíbrio entre precisão e sensibilidade
#
# support    → quantidade de exemplos utilizados
#
# Essas métricas são importantes para o Pantanal Guardian AI,
# porque não queremos avaliar o modelo somente pela acurácia.
print()
print("==========================================")
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("==========================================")

print(classification_report(y_teste, previsoes))


# ------------------------------------------------------------
# 13. TESTANDO UM NOVO DIA
# ------------------------------------------------------------

# Agora vamos imaginar que chegou um novo conjunto de
# informações meteorológicas.
#
# Temperatura máxima = 37°C
# Umidade mínima = 27%
# Dias consecutivos com chuva inferior a 1 mm = 8
#
# Esse dia ainda não possui um resultado conhecido.
#
# Portanto, vamos pedir para o modelo fazer uma previsão.
novo_dia = np.array([
    [37, 27, 8]
])


# ------------------------------------------------------------
# 14. PREVISÃO DO NOVO DIA
# ------------------------------------------------------------

# predict() vai retornar:
#
# 0 → não há risco classificado
# 1 → há risco classificado
#
# O resultado é armazenado na variável resultado.
resultado = modelo.predict(novo_dia)


# ------------------------------------------------------------
# 15. CALCULANDO A PROBABILIDADE
# ------------------------------------------------------------

# Aqui temos algo muito interessante.
#
# predict() simplesmente retorna a classe:
#
# 0 ou 1
#
# Já predict_proba() retorna as probabilidades.
#
# O resultado terá duas probabilidades:
#
# primeira → probabilidade de ser classe 0
#
# segunda → probabilidade de ser classe 1
probabilidades = modelo.predict_proba(novo_dia)


# ------------------------------------------------------------
# 16. PEGANDO A PROBABILIDADE DE FOCO
# ------------------------------------------------------------

# probabilidades[0] significa:
#
# "pegue as probabilidades do primeiro novo dia"
#
# probabilidades[0][1] significa:
#
# "pegue a probabilidade de ele pertencer à classe 1"
#
# Como classe 1 significa "foco de calor",
# esse valor representa a probabilidade estimada
# de ocorrência de foco.
probabilidade_foco = probabilidades[0][1]


# ------------------------------------------------------------
# 17. MOSTRANDO O RESULTADO
# ------------------------------------------------------------

print()
print("==========================================")
print("PREVISÃO PARA UM NOVO DIA")
print("==========================================")

print("Temperatura máxima:", novo_dia[0][0], "°C")
print("Umidade mínima:", novo_dia[0][1], "%")
print("Dias secos:", novo_dia[0][2])

print()

print("Resultado previsto:", resultado[0])

print(
    "Probabilidade de foco:",
    round(probabilidade_foco * 100, 2),
    "%"
)


# ------------------------------------------------------------
# 18. TRANSFORMANDO A PREVISÃO EM UMA MENSAGEM
# ------------------------------------------------------------

# Vamos deixar o resultado mais fácil de entender.
#
# Se o modelo retornar 1:
# existe uma classificação de ocorrência de foco.
#
# Se retornar 0:
# não existe essa classificação.
if resultado[0] == 1:

    print()
    print("ALERTA: condições associadas a risco de foco de calor.")

else:

    print()
    print("Sem alerta: condições classificadas como baixo risco.")


# ============================================================
# FIM DO PROGRAMA
# ============================================================