import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose

dataset_path = "dataset/openpowerlifting-2024-01-06-4c732975.csv"

dataset = pd.read_csv(dataset_path, low_memory=False)

dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')
df = dataset.dropna(subset=['Date'])

monthly = df.resample('M', on='Date').size()

# 1) TENDÊNCIA – Número de Competidores (STL)
stl = STL(monthly, period=12)
result = stl.fit()

plt.figure(figsize=(14, 5))
plt.plot(result.trend, color="blue")
plt.title("Tendência – Competidores por Mês (STL)")
plt.xlabel("Ano")
plt.ylabel("Tendência")
plt.grid(True)
plt.show()


# 2) SAZONALIDADE – Competidores por Mês
decomp = seasonal_decompose(monthly, model='additive', period=12)

plt.figure(figsize=(14, 5))
plt.plot(decomp.seasonal, color="green")
plt.title("Sazonalidade – Competidores por Mês")
plt.xlabel("Ano")
plt.ylabel("Impacto Sazonal")
plt.grid(True)
plt.show()


# 3) CICLOS – Resíduos (STL)
plt.figure(figsize=(14, 5))
plt.plot(result.resid, color="red")
plt.title("Ciclicidade / Resíduos – Competidores por Mês")
plt.xlabel("Ano")
plt.ylabel("Resíduos (Ciclos + Ruído)")
plt.grid(True)
plt.show()


# 4) Número de competições por mês
meets_per_month = df.groupby(df['Date'].dt.to_period('M'))['MeetName'].nunique()
meets_per_month.index = meets_per_month.index.to_timestamp()

plt.figure(figsize=(14,5))
plt.plot(meets_per_month)
plt.title("Número de Competições por Mês")
plt.xlabel("Ano")
plt.ylabel("Competições")
plt.grid(True)
plt.show()


# 5) Número de competidores por mês
competitors_per_month = monthly

plt.figure(figsize=(14,5))
plt.plot(competitors_per_month)
plt.title("Número de Competidores por Mês")
plt.xlabel("Ano")
plt.ylabel("Competidores")
plt.grid(True)
plt.show()


# Verificando propocionalidade de número de atletas por competição
fig, ax1 = plt.subplots(figsize=(14,6))

ax1.plot(meets_per_month, color='blue', label='Competições')
ax1.set_ylabel('Competições', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(competitors_per_month, color='red', alpha=0.6, label='Competidores')
ax2.set_ylabel('Competidores', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title("Competições vs Competidores por Mês")
plt.grid(True)
plt.show()



# 6) Participação por sexo ao longo do tempo
gender_monthly = df.groupby([df['Date'].dt.to_period('M'), 'Sex']).size().unstack(fill_value=0)
gender_monthly.index = gender_monthly.index.to_timestamp()

plt.figure(figsize=(14,5))
plt.plot(gender_monthly['M'], label='Masculino')
plt.plot(gender_monthly['F'], label='Feminino')
plt.title("Participação Masculina vs Feminina por Mês")
plt.xlabel("Ano")
plt.ylabel("Competidores")
plt.legend()
plt.grid(True)
plt.show()


# 7) Raw vs Equipped ao longo do tempo (Qual modalidade tem crescido mais?)
equipment_monthly = df.groupby([df['Date'].dt.to_period('M'), 'Equipment']).size().unstack(fill_value=0)
equipment_monthly.index = equipment_monthly.index.to_timestamp()

plt.figure(figsize=(14,5))
for col in equipment_monthly.columns:
    plt.plot(equipment_monthly[col], label=col)

plt.title("Participação por Tipo de Equipamento (Raw vs Equipped)")
plt.xlabel("Ano")
plt.ylabel("Competidores")
plt.legend()
plt.grid(True)
plt.show()


# 8) Média dos Totais por mês
valid_totals = df.dropna(subset=['TotalKg'])
avg_total_monthly = valid_totals.groupby(valid_totals['Date'].dt.to_period('M'))['TotalKg'].mean()
avg_total_monthly.index = avg_total_monthly.index.to_timestamp()

plt.figure(figsize=(14,5))
plt.plot(avg_total_monthly)
plt.title("Média do Total (Kg) dos Atletas por Mês")
plt.xlabel("Ano")
plt.ylabel("Total Médio (Kg)")
plt.grid(True)
plt.show()


# 9) Novos atletas por mês
df_sorted = df.sort_values('Date')
first_appearance = df_sorted.groupby('Name')['Date'].first()

new_lifters_monthly = first_appearance.groupby(first_appearance.dt.to_period('M')).size()
new_lifters_monthly.index = new_lifters_monthly.index.to_timestamp()

plt.figure(figsize=(14,5))
plt.plot(new_lifters_monthly)
plt.title("Número de Novos Atletas por Mês")
plt.xlabel("Ano")
plt.ylabel("Novos Atletas")
plt.grid(True)
plt.show()


# 10) Recordes – Maior Total por mês
max_total_monthly = valid_totals.groupby(valid_totals['Date'].dt.to_period('M'))['TotalKg'].max()
max_total_monthly.index = max_total_monthly.index.to_timestamp()

plt.figure(figsize=(14,5))
plt.plot(max_total_monthly)
plt.title("Recorde de Total (Kg) por Mês")
plt.xlabel("Ano")
plt.ylabel("Maior Total (Kg)")
plt.grid(True)
plt.show()
