import tkinter as tk
from tkinter import filedialog, messagebox
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import sys

# 🔧 Ajuste para rodar no .exe (PyInstaller)
def caminho_recurso(relativo):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relativo)


# 🚀 Carregar modelo
model_path = caminho_recurso('modelo_laudos')

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


# 🔍 Função para previsão
def prever_laudo(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred


# 🎨 Interface Tkinter
janela = tk.Tk()
janela.title("Classificador de Laudos Veterinários")
janela.geometry("600x400")
janela.resizable(False, False)


# 📝 Campo de texto
label_instrucao = tk.Label(janela, text="Digite o laudo ou carregue um arquivo .txt", font=("Arial", 12))
label_instrucao.pack(pady=10)

caixa_texto = tk.Text(janela, width=70, height=10, font=("Arial", 10))
caixa_texto.pack(pady=10)


# 📂 Função para carregar txt
def carregar_txt():
    arquivo = filedialog.askopenfilename(filetypes=[("Arquivos de texto", "*.txt")])
    if arquivo:
        with open(arquivo, "r", encoding="utf-8") as f:
            conteudo = f.read()
            caixa_texto.delete(1.0, tk.END)
            caixa_texto.insert(tk.END, conteudo)


# 🔍 Função para classificar
def classificar():
    texto = caixa_texto.get(1.0, tk.END).strip()
    if not texto:
        messagebox.showwarning("Aviso", "Por favor, insira um laudo para classificar.")
        return

    resultado = prever_laudo(texto)
    if resultado == 1:
        mensagem = "✅ Laudo COM ALTERAÇÃO"
    else:
        mensagem = "🟢 Laudo SEM ALTERAÇÃO"

    messagebox.showinfo("Resultado", mensagem)


# 🎛️ Botões
frame_botoes = tk.Frame(janela)
frame_botoes.pack(pady=10)

btn_classificar = tk.Button(frame_botoes, text="Classificar Laudo", command=classificar, width=20, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
btn_classificar.grid(row=0, column=0, padx=10)

btn_carregar = tk.Button(frame_botoes, text="Carregar TXT", command=carregar_txt, width=20, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
btn_carregar.grid(row=0, column=1, padx=10)

btn_sair = tk.Button(frame_botoes, text="Sair", command=janela.quit, width=10, bg="#f44336", fg="white", font=("Arial", 10, "bold"))
btn_sair.grid(row=0, column=2, padx=10)


# 🚀 Rodar interface
janela.mainloop()
