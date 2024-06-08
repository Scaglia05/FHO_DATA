import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Visualizar as primeiras linhas dos dados
print(train.head())

# Informações sobre os dados
print(train.info())

# Estatísticas descritivas
print(train.describe())

# Análise de valores ausentes
missing_values = train.isnull().sum()
print(missing_values[missing_values > 0])

# Visualização da distribuição dos preços das casas
sns.histplot(train['SalePrice'], kde=True)
plt.title('Distribuição dos Preços das Casas')
plt.xlabel('Preço')
plt.ylabel('Frequência')
plt.show()

# Correlação entre variáveis
corr_matrix = train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

# Selecionar a feature
feature = 'OverallQual'

# Criar uma instância do modelo de regressão linear
model = LinearRegression()

# Separar a variável alvo e a feature selecionada
X_train = train[[feature]]
y_train = train['SalePrice']
X_test = test[[feature]]

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Exibir resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Dados de Treinamento')
plt.plot(X_test, y_pred, color='red', linewidth=2,
         label='Modelo de Regressão Linear')
plt.title('Regressão Linear - Preços das Casas em função de OverallQual')
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.legend()
plt.show()