import pandas as pd
import utils.dataset as dt
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose

dataset_path = "dataset/openpowerlifting-2024-01-06-4c732975.csv"

if not os.path.exists(dataset_path):
    dt.download_dataset()

try:
    dataset = pd.read_csv(dataset_path)
    print(dataset.head())
except Exception as e:
    print("Ocorreu um erro ao carregar o dataset:", e)


# Análise de Tendência da Série Temporal
dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
df = dataset.dropna(subset=['Date'])

monthly = df.resample('M', on='Date').size()

stl = STL(monthly, period=12)
result = stl.fit()

plt.figure(figsize=(14,5))
plt.plot(result.trend)
plt.title("Tendência da Série Temporal - Número de Competidores por Mês")
plt.xlabel("Ano")
plt.ylabel("Tendência (valores suavizados)")
plt.grid(True)
plt.show()


# Análise Sazonal da Série Temporal
dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
df = dataset.dropna(subset=['Date'])

# Série mensal
monthly = df.resample('M', on='Date').size()

# Decomposição clássica de sazonalidade
decomp = seasonal_decompose(monthly, model='additive', period=12)

plt.figure(figsize=(14,5))
plt.plot(decomp.seasonal)
plt.title("Sazonalidade - Série Mensal de Competidores")
plt.xlabel("Ano")
plt.ylabel("Impacto Sazonal")
plt.grid(True)
plt.show()


# Análise de Ciclicidade da Série Temporal
dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
df = dataset.dropna(subset=['Date'])

# Série mensal
monthly = df.resample('M', on='Date').size()

# Decomposição STL
stl = STL(monthly, period=12)
result = stl.fit()

# O componente "resid" contém ciclos e impactos externos
plt.figure(figsize=(14,5))
plt.plot(result.resid)
plt.title("Ciclos / Componente Residual – Competidores por Mês")
plt.xlabel("Ano")
plt.ylabel("Resíduos (Ciclos + Ruído)")
plt.grid(True)
plt.show()