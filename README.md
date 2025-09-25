Raciocínio Baseado em Casos (RBC) para Classificação de Crocodilos
Este projeto é uma demonstração prática do conceito de Raciocínio Baseado em Casos (RBC) aplicado a um problema de classificação. Utilizamos o algoritmo k-Nearest Neighbors (k-NN) para prever o status de conservação de uma espécie de crocodilo com base em suas características, comparando-a com um dataset de espécies já conhecidas.

👥 Autores
Miguel Schneiders Flach

Felipe Eduardo Bohnen

Roney Bieger Anshau

📖 Conceito: Raciocínio Baseado em Casos (RBC)
O Raciocínio Baseado em Casos é uma metodologia de Inteligência Artificial que foca na resolução de novos problemas a partir da análise e adaptação de soluções de problemas passados. O processo é cíclico e segue quatro etapas principais:

Retrieve (Recuperar): Dado um novo problema (um novo caso), o sistema busca em sua base de conhecimento (o dataset) os casos mais similares.

Reuse (Reutilizar): A solução do caso recuperado é adaptada para o novo problema.

Revise (Revisar): A solução proposta é avaliada e, se necessário, corrigida para garantir sua eficácia.

Retain (Reter): O novo problema, junto com sua solução validada, é armazenado na base de conhecimento, enriquecendo-a para futuras consultas.

📊 Dataset
Utilizamos o dataset "Global Crocodile Species", disponível publicamente no Kaggle. Ele contém informações sobre diversas espécies de crocodilos, e as principais colunas usadas foram:

Habitat

Comprimento (m)

Peso (kg)

Gênero

Status de Conservação (nossa variável alvo)

🛠️ Implementação
A lógica do RBC foi implementada em Python, utilizando o algoritmo k-Nearest Neighbors (k-NN) da biblioteca scikit-learn para executar a etapa de Recuperação.

Ferramentas: Python 3, Pandas, Scikit-learn.

Lógica:

Os dados categóricos (como Habitat e Gênero) são convertidos para formato numérico usando LabelEncoder.

Um modelo k-NN (com k=3) é treinado com os dados existentes.

Para classificar um novo crocodilo, o k-NN identifica os 3 casos mais "próximos" (similares) no dataset.

A classe (status de conservação) mais comum entre os vizinhos é atribuída ao novo caso.

Exemplo de Código
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Leitura e preparação dos dados
df = pd.read_csv('Global_Crocodile_Species.csv')
le = LabelEncoder()

# Converte colunas de texto para números
for col in ['Habitat', 'Gênero', 'Status de Conservação']:
    df[col] = le.fit_transform(df[col])

# Separa features (X) e target (y)
X = df[['Habitat', 'Comprimento', 'Peso', 'Gênero']]
y = df['Status de Conservação']

# 2. Treinamento do modelo k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# 3. Novo crocodilo para predição
# (Valores já convertidos para o formato numérico)
novo_crocodilo = [[0, 4.5, 400, 1]] # Ex: Freshwater, 4.5m, 400kg, Crocodylus

# 4. Predição
predicao = knn.predict(novo_crocodilo)
status_previsto = le.inverse_transform(predicao)

print(f"Status previsto para o novo crocodilo: {status_previsto[0]}")

🚀 Como Executar
Clone o repositório:

git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
cd seu-repositorio

Instale as dependências:

pip install pandas scikit-learn

Execute o script:

Certifique-se de que o arquivo Global_Crocodile_Species.csv está na mesma pasta.

Rode o script Python para ver a predição.

🏁 Conclusão
Este trabalho demonstra como o k-NN pode ser uma ferramenta simples e eficaz para implementar a fase de recuperação do ciclo RBC. O projeto mostra que, com dados históricos de qualidade, é possível construir sistemas inteligentes capazes de resolver novos problemas com base na experiência acumulada.
