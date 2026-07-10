"""
=============================================================
  Classificador de Cogumelos - Árvore de Decisão
  Dataset: mushrooms_pt.csv (8124 amostras, 22 features)
  Objetivo: prever se um cogumelo é venenoso ou comestível
=============================================================
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# ============================================================
#  1. CARREGANDO E EXPLORANDO OS DADOS
# ============================================================

df = pd.read_csv('mushrooms_pt.csv')  # le o CSV e transforma em uma tabela, um dataframe

print(df.shape)                       # mostra quantas linhas e colunas tem o dataframe
print(df['classe'].value_counts())    # conta quantos de cada classe (venenoso/comestivel)
print(df.isnull().sum().sum())        # soma todos os valores vazios do dataset

# agora, vamos olhar as features. O arquivo é em texto, e a máquina só entende números, logo:

print(df.dtypes)   # tipo de cada coluna
print(df.head(3))  # as 3 primeiras linhas

# pandas chama texto de Object
# vamos transformar esses objetos em números, usando o LabelEncoder!
# Ex: sino -> 0, convexo -> 2 ....

# ============================================================
#  2. PRÉ-PROCESSAMENTO - ENCODING DAS VARIÁVEIS CATEGÓRICAS
# ============================================================

le = LabelEncoder()

df_encoded = df.copy()  # copia o dataframe inteiro pra não perder os dados originais

for coluna in df_encoded.columns:                        # passa por cada coluna uma por uma
    df_encoded[coluna] = le.fit_transform(df_encoded[coluna])  # aprende as categorias da coluna e transforma em numeros

print(df_encoded.head(3))

# ============================================================
#  3. SEPARANDO FEATURES (X) E TARGET (y)
# ============================================================

# X e y: convenção universal em ML
# X -> Features, o que a máquina vai usar pra aprender,
#      nesse caso as 22 colunas com as categorias já encodadas pra número
# y -> Target, o que está sendo ensinado a prever:
#      se é venenoso ou comestível

X = df_encoded.drop(columns=['classe'])  # pega tudo menos a coluna classe (pro modelo não espionar a resposta)
y = df_encoded['classe']       # pega só a coluna classe

print(X.shape)
print(y.shape)

# ============================================================
#  4. DIVIDINDO TREINO E TESTE
# ============================================================

# dividindo treino de teste! treino fica com 80% dos dados e teste 20%, pra ver se ele realmente aprendeu.

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42  # test_size=0.2 são os 20% de dados pro teste
)

print(X_treino.shape)
print(X_teste.shape)

# ============================================================
#  5. TREINANDO O MODELO
# ============================================================

# agora que temos os dados separadinhos, com um total de 6499 pra treino
# e 1625 pra teste, vamos treinar o nosso modelo com esses dados!

modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_treino, y_treino)  # aqui ele olha as features e aprende as regras!

print("Modelo treinado com sucesso!")

# ============================================================
#  6. AVALIANDO O MODELO
# ============================================================

# agora, vamos ver se ele aprendeu mesmo!

y_pred = modelo.predict(X_teste)  # o modelo vai tentar adivinhar a classe

print(f"Acuracia do modelo: {accuracy_score(y_teste, y_pred):.2%}")
print(classification_report(y_teste, y_pred, target_names=['comestivel', 'venenoso']))

# accuracy_score é a porcentagem de acerto!
# classification_report é o relatório detalhado por classe.

print(f"Profundidade da árvore: {modelo.get_depth()}")
print(f"Número de folhas: {modelo.get_n_leaves()}")

# ============================================================
#  7. IMPORTÂNCIA DAS FEATURES
# ============================================================

importancias = pd.Series(
    modelo.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# feature_importances_ -> o modelo atribui uma pontuação pra cada feature
# que ajudou a árvore na decisão, ordena da mais importante pra menos

print(importancias.head(5))

# cor_esporos foi a primeira, o que é interessante e acurado:
# na natureza, cor é um sinal de perigo em diversas espécies,
# não só de fungos, mas de animais e plantas também!

# ============================================================
#  8. VISUALIZANDO AS REGRAS DA ÁRVORE
# ============================================================

# hora divertida! vamos ver a árvore, ou seja,
# as perguntas que ela faz pra tomar decisões

regras = export_text(modelo, feature_names=list(X.columns))
print(regras)

# legal, criamos uma árvore legível e a primeira pergunta é sempre
# sobre a cor dos esporos! mas, agora vamos deixar isso menos matemático e mais legível:

for feature in ['cor_esporos', 'numero_aneis', 'tamanho_lamina']:
    categorias = df[feature].unique()
    codigos = le.fit_transform(categorias)
    legenda = {int(codigo): categoria for codigo, categoria in zip(codigos, categorias)}
    print(f"\n{feature}: {dict(sorted(legenda.items()))}")

# ============================================================
#  9. VISUALIZAÇÃO GRÁFICA DA ÁRVORE
# ============================================================
# Ler é legal, mas dá pra melhorar! Agora, vamos VER a árvore!

plt.figure(figsize=(20, 10))
plot_tree(
    modelo,
    feature_names=list(X.columns),
    class_names=['comestivel', 'venenoso'],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.savefig('arvore_decisao_cogumelos.png', dpi=150, bbox_inches='tight')
plt.show()
print("Árvore salva como arvore_decisao_cogumelos.png!")