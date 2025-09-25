import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import os
import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedTk

# --- ETAPA 1: Carregar Dados e Treinar o Modelo (executa apenas uma vez) ---

try:
    caminho_csv = "data/crocodile_dataset.csv"
    if not os.path.isfile(caminho_csv):
        raise FileNotFoundError(f"Arquivo não encontrado: '{caminho_csv}'. Certifique-se de que o arquivo está no diretório correto.")

    df = pd.read_csv(caminho_csv)

    le_habitat = LabelEncoder()
    df['Habitat_num'] = le_habitat.fit_transform(df['Habitat Type'])

    le_genus = LabelEncoder()
    df['Genus_num'] = le_genus.fit_transform(df['Genus'])

    X = df[['Habitat_num', 'Observed Length (m)', 'Observed Weight (kg)', 'Genus_num']]
    y = df['Conservation Status']

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

except FileNotFoundError as e:
    messagebox.showerror("Erro de Arquivo", str(e))
    exit()
except Exception as e:
    messagebox.showerror("Erro de Inicialização", f"Ocorreu um erro ao carregar o modelo: {e}")
    exit()


# --- ETAPA 2: Classe da Aplicação com Interface Gráfica ---

class CrocClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Classificador de Crocodilos")
        self.master.geometry("700x750")
        self.master.minsize(600, 700)

        self.style = ttk.Style(self.master)
        self.style.theme_use('arc')

        # Paleta de cores
        self.colors = {
            "primary": "#3498db",    # Azul
            "secondary": "#2c3e50",  # Azul escuro
            "background": "#ecf0f1", # Cinza claro
            "text": "#34495e",       # Cinza escuro
            "light_text": "#ffffff"  # Branco
        }

        self.master.configure(bg=self.colors["background"])
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.master, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_label = ttk.Label(
            header_frame,
            text="Classificador de Crocodilos (RBC)",
            font=("Segoe UI", 22, "bold"),
            foreground=self.colors["secondary"]
        )
        header_label.pack()
        
        sub_header_label = ttk.Label(
            header_frame,
            text="Insira as características da espécie para prever seu status de conservação.",
            font=("Segoe UI", 11),
            foreground=self.colors["text"]
        )
        sub_header_label.pack()

        input_frame = ttk.Frame(main_frame, padding=20)
        input_frame.pack(fill=tk.X, pady=10)
        
        input_frame.columnconfigure(1, weight=1)

        font_label = ("Segoe UI", 12)
        font_entry = ("Segoe UI", 11)
        
        fields = {
            "Habitat:": list(df['Habitat Type'].unique()),
            "Comprimento (m):": None,
            "Peso (kg):": None,
            "Gênero:": list(df['Genus'].unique())
        }

        self.entries = {}
        for i, (text, values) in enumerate(fields.items()):
            label = ttk.Label(input_frame, text=text, font=font_label, foreground=self.colors["text"])
            label.grid(row=i, column=0, sticky=tk.W, padx=10, pady=10)

            if values:
                entry = ttk.Combobox(input_frame, values=values, font=font_entry, state="readonly")
            else:
                entry = ttk.Entry(input_frame, font=font_entry)
            
            entry.grid(row=i, column=1, sticky=tk.EW, padx=10, pady=10)
            self.entries[text.split(':')[0]] = entry

        # --- Botão (COM A COR DO TEXTO ALTERADA) ---
        self.style.configure(
            'Accent.TButton',
            font=('Segoe UI', 12, 'bold'),
            background=self.colors["primary"],
            foreground=self.colors["secondary"]  # <-- Alteração aqui!
        )
        
        classify_button = ttk.Button(
            main_frame,
            text="Classificar Espécie",
            style='Accent.TButton',
            command=self.realizar_predicao,
            cursor="hand2"
        )
        classify_button.pack(pady=20, ipady=5)

        result_frame = ttk.Frame(main_frame, padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(
            result_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            state=tk.DISABLED,
            bg="white",
            relief=tk.SOLID,
            bd=1,
            padx=10,
            pady=10
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

    def realizar_predicao(self):
        try:
            habitat = self.entries["Habitat"].get()
            length_str = self.entries["Comprimento (m)"].get()
            weight_str = self.entries["Peso (kg)"].get()
            genus = self.entries["Gênero"].get()

            if not all([habitat, length_str, weight_str, genus]):
                messagebox.showwarning("Campos Vazios", "Por favor, preencha todos os campos.")
                return

            length = float(length_str)
            weight = float(weight_str)

            habitat_num = le_habitat.transform([habitat])[0]
            genus_num = le_genus.transform([genus])[0]
            
            novo_caso_features = np.array([[habitat_num, length, weight, genus_num]])
            predicao = knn.predict(novo_caso_features)[0]

            distancias, indices = knn.kneighbors(novo_caso_features)
            vizinhos = df.iloc[indices[0]]

            resultado = f"Status de Conservação Previsto: {predicao}\n\n"
            resultado += "--- Espécies Mais Semelhantes Encontradas ---\n\n"
            for _, vizinho in vizinhos.iterrows():
                resultado += (
                    f"• Nome: {vizinho['Common Name']} (Status: {vizinho['Conservation Status']})\n"
                    f"  Detalhes: {vizinho['Genus']}, {vizinho['Habitat Type']}, {vizinho['Observed Length (m)']}m, {vizinho['Observed Weight (kg)']}kg\n\n"
                )

            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, resultado)
            self.result_text.config(state=tk.DISABLED)

        except ValueError:
            messagebox.showerror("Erro de Entrada", "Por favor, insira valores numéricos válidos para Comprimento e Peso.")
        except Exception as e:
            messagebox.showerror("Erro Inesperado", f"Ocorreu um erro: {e}")


# --- ETAPA 3: Iniciar a Aplicação ---
if __name__ == "__main__":
    root = ThemedTk(theme="arc")
    app = CrocClassifierApp(root)
    root.mainloop()
