import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor

#Importando base de dados
df = pd.read_csv(
    r'dados_preprocessados_temp_rj.csv',
    index_col='time',
    parse_dates=True
                 )

#Breve análise descritiva com estatísticas básicas dos dados
#print(df[['TARGET']].describe())

#Visualizar dados de forma visual (Gráfico de linha)
df[['TARGET']].plot(figsize=(12,4))
#plt.show()

#Visualizar distribuição dos dados (Gráfico de histograma)
sns.histplot(df[["TARGET"]], kde=True)
#plt.show()

#Criando uma cópia dos dados
df_preprocessado = df.copy()

#Separação dos dados em Target (y) e Features (x)
#Target = Variável Prevista = Y
y_target = df_preprocessado[["TARGET"]]

#Features = Variáveis (is) Preditora(s) = X
#Temperatura diária registrada 1 mês atrás
x_features = df_preprocessado.drop(columns=["TARGET"])

#Separação em treino e testes
#Seguiremos com uma separação padrão de ~70% para treino e 30% para teste
#Primeiros 25 anos destinados para treinamento (1991 - 2015)
y_train = y_target['1991':'2015']
x_train = x_features['1991':'2015']

#Últimos 10 anos destinados para teste/validação (2016 - 2025)
y_test = y_target['2016':]
x_test = x_features["2016":]

#Visualizando percentuais das amostras
pct_treino = (len(y_train) / len(y_target)) * 100
print(f'Percentual de dados para treinamento: {pct_treino:.0f}%')
pct_teste = (len(y_test) / len(y_target)) * 100
print(f'Percentual de dados para teste: {pct_teste:.0f}%')

#Modelo escolhido -> XGBosst -> Modelo gradiente
#Muito utilizado em séries temporais

#Modelo de configuração (parametrizado)
modelo = XGBRegressor(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    colsample_bytree=0.8,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
)

#Treinar o modelo com dados
modelo.fit(x_train, y_train)

#Visualizar previsões
#print(modelo.predict(x_test))

#Melhorar visualização das previsões
pred_train = pd.DataFrame(modelo.predict(x_train), index=x_train.index, columns=['Y_PRED'])
pred_test = pd.DataFrame(modelo.predict(x_test), index=x_test.index, columns=['Y_PRED'])
#print(prend_test)

#Plotar previsões vs Dados reais
#Criando figura
fig, ax = plt.subplots(figsize=(12,4))

#Plotar dados
ax.plot(y_target, label = 'Valores Reais')
ax.plot(pred_train, label = 'Previsões - Treino')
ax.plot(pred_test, label = 'Previsões - Teste')

ax.legend(ncols=3)
plt.show()

# R2 -> coeficiente de determinação
# -> mostra quanto da variância total dos seus dados seu modelo consegue explicar
# -> mostra quão bem seu modelo explica o comportamento da sua série
# -> quanto maior melhor

r2_test = r2_score(y_true=y_test, y_pred=pred_test)
r2_train = r2_score(y_true=y_train, y_pred=pred_train)

print(f" -- Valores de R2 -> Treino: {r2_train:.2f} | Teste: {r2_test:.2f}")

# MAPE -> erro médio absoluto percentual
# -> medida de erro global do seu modelo
# -> quanto menor melhor

mape_test = mean_absolute_percentage_error(y_true=y_test, y_pred=pred_test) * 100
mape_train = mean_absolute_percentage_error(y_true=y_train, y_pred=pred_train) * 100

print(f" -- Valores de MAPE -> Treino: {mape_train:.2f}% | Teste: {mape_test:.2f}%")