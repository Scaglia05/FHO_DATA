# Previsão de Preços de Casas usando Regressão Linear

Este projeto utiliza regressão linear para prever os preços das casas com base na qualidade geral da casa (OverallQual). A análise e visualização de dados foram feitas usando pandas, numpy, seaborn e matplotlib.

## Descrição do Projeto

O objetivo é analisar um conjunto de dados de preços de casas, entender a correlação entre as variáveis e aplicar um modelo de regressão linear para prever os preços das casas com base na qualidade geral da casa.

## Estrutura do Código

1. **Importação de Bibliotecas:**
   - pandas, numpy, seaborn, matplotlib.pyplot, LinearRegression, mean_squared_error, r2_score

2. **Carregar os Dados:**
   - Carregar os conjuntos de dados `train.csv` e `test.csv`.

3. **Visualizar os Dados:**
   - Exibir as primeiras linhas, informações gerais e estatísticas descritivas dos dados.
   - Identificar e exibir colunas com valores ausentes.

4. **Visualização da Distribuição dos Preços das Casas:**
   - Plotar um histograma da variável `SalePrice`.

5. **Correlação entre Variáveis:**
   - Calcular e plotar a matriz de correlação.

6. **Seleção da Feature e Treinamento do Modelo:**
   - Selecionar `OverallQual` como a feature, criar e treinar o modelo de regressão linear.

7. **Fazer Previsões:**
   - Usar o modelo treinado para fazer previsões com os dados de teste.

8. **Exibir Resultados:**
   - Plotar um gráfico de dispersão dos dados de treinamento e a linha de regressão linear.

## Requisitos

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## Como Executar

1. Certifique-se de ter todas as bibliotecas instaladas.
2. Coloque os arquivos `train.csv` e `test.csv` no mesmo diretório do script.
3. Execute o script Python.

## Resultados

O gráfico de regressão linear mostra a relação entre a qualidade geral da casa e o preço de venda, permitindo prever os preços das casas com base nessa característica.
