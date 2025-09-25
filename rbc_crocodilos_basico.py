import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import os

# -------------------------------
# 1️⃣ Definir CSV
# -------------------------------
caminho_csv = "data/crocodile_dataset.csv"  # Ajuste o caminho se necessário
if not os.path.isfile(caminho_csv):
    raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv}")

df = pd.read_csv(caminho_csv)
print(f"Colunas disponíveis no CSV: {list(df.columns)}")

# -------------------------------
# 2️⃣ Transformar colunas categóricas
# -------------------------------
le_habitat = LabelEncoder()
df['Habitat_num'] = le_habitat.fit_transform(df['Habitat Type'])

le_genus = LabelEncoder()
df['Genus_num'] = le_genus.fit_transform(df['Genus'])

# -------------------------------
# 3️⃣ Preparar features e rótulo
# -------------------------------
X = df[['Habitat_num','Observed Length (m)','Observed Weight (kg)','Genus_num']]
y = df['Conservation Status']

# -------------------------------
# 4️⃣ Treinar modelo k-NN
# -------------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# -------------------------------
# 5️⃣ Receber características do usuário
# -------------------------------
print("\nDigite as características do crocodilo:")
habitat_input = input(f"Habitat ({list(df['Habitat Type'].unique())}): ").strip()
length_input = float(input("Comprimento (m): ").strip())
weight_input = float(input("Peso (kg): ").strip())
genus_input = input(f"Gênero ({list(df['Genus'].unique())}): ").strip()

# Transformar entradas categóricas
habitat_num = le_habitat.transform([habitat_input])[0]
genus_num = le_genus.transform([genus_input])[0]

novo_caso = {
    'Habitat_num': habitat_num,
    'Observed Length (m)': length_input,
    'Observed Weight (kg)': weight_input,
    'Genus_num': genus_num
}

# -------------------------------
# 6️⃣ Previsão do status
# -------------------------------
predicao = knn.predict(np.array([list(novo_caso.values())]))[0]
print("\nStatus previsto para o crocodilo:", predicao)

# -------------------------------
# 7️⃣ Vizinhos mais próximos
# -------------------------------
distancias, indices = knn.kneighbors(np.array([list(novo_caso.values())]))
vizinhos = df.iloc[indices[0]]

print("\nCrocodilos mais parecidos:")
for i, vizinho in vizinhos.iterrows():
    print(f"- Nome: {vizinho['Common Name']}, Gênero: {vizinho['Genus']}, "
          f"Habitat: {vizinho['Habitat Type']}, Comprimento: {vizinho['Observed Length (m)']} m, "
          f"Peso: {vizinho['Observed Weight (kg)']} kg, Status: {vizinho['Conservation Status']}")
