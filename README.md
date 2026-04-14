# python-machine-learning-xgboost-previsao-tempo

# 🌡️ Previsão de Temperatura com Machine Learning (XGBoost)

## 📌 Sobre o Projeto

Este projeto tem como objetivo aplicar conceitos de Machine Learning para previsão de temperatura com base em dados históricos.

O modelo utilizado foi o **XGBoost Regressor**, amplamente utilizado em problemas de séries temporais e regressão devido à sua alta performance.

> ⚠️ Este projeto foi desenvolvido com fins educacionais, com base em estudos e prática de conceitos fundamentais de ciência de dados.

---

## 🧠 Tecnologias utilizadas

* Python
* Pandas
* Seaborn
* Matplotlib
* Scikit-learn
* XGBoost

---

## 📊 Etapas do Projeto

### 1. Importação e análise dos dados

* Leitura do dataset com `pandas`
* Conversão da coluna de tempo para índice temporal
* Visualização da série temporal
* Análise da distribuição dos dados (histograma + KDE)

---

### 2. Separação de dados

* Target (y): variável a ser prevista (`TARGET`)
* Features (X): variáveis preditoras

Divisão baseada em tempo:

* Treino: 1991 até 2015 (~70%)
* Teste: 2016 até 2025 (~30%)

✔️ Importante: Em séries temporais, NÃO se usa embaralhamento (shuffle)

---

### 3. Treinamento do modelo

Utilização do modelo:

```python
XGBRegressor
```

Com hiperparâmetros ajustados manualmente.

---

### 4. Geração de previsões

* Previsões para treino (`pred_train`)
* Previsões para teste (`pred_test`)

---

### 5. Avaliação do modelo

#### 📈 R² (Coeficiente de Determinação)

* Mede o quanto o modelo explica os dados
* Quanto mais próximo de 1, melhor

#### 📉 MAPE (Erro Percentual Médio Absoluto)

* Mede o erro médio percentual
* Quanto menor, melhor

---

## ⚠️ Overfitting em Machine Learning

Um ponto importante observado neste projeto é o conceito de **Overfitting**.

👉 Ocorre quando:

* O modelo aprende MUITO bem os dados de treino
* Mas tem desempenho ruim em dados novos (teste)

### 🔍 Como identificar:

* R² treino muito alto
* R² teste significativamente menor

### 🛠️ Possíveis soluções:

* Regularização
* Redução da complexidade do modelo
* Mais dados
* Validação cruzada

---

## 📌 Resultados

O modelo foi capaz de capturar padrões relevantes da série temporal, porém é importante analisar possíveis sinais de overfitting ao comparar treino vs teste.

