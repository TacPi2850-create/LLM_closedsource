from openai import OpenAI
import os
import streamlit as st
import streamlit.components.v1 as components
import socket


# === CONFIGURAZIONE PAGINA STREAMLIT ===
st.set_page_config(page_title="Jarvis", page_icon="🤖", layout="wide")

def check_lmstudio_alive(host="localhost", port=1234, timeout=2):
    """Verifica se il server LM Studio è attivo sulla porta specificata."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False
# === MODELLI DISPONIBILI ===
available_models = {
    "Qwen2.5 7B Instruct 1M": {
        "descrizione": (
            "Gratuito - Locale (LM Studio)\n"
            "Modello open-source eseguito direttamente sul tuo computer tramite LM Studio.\n"
            "Non richiede connessione internet né credenziali esterne.\n"
            "🔍 Ideale per spiegazioni teoriche, analisi aziendali, trend e interpretazione dei dati."
        ),
        "esempi": (
            "- \"Spiegami i KPI di questo report.\"\n"
            "- \"Trova anomalie nei ricavi.\"\n"
            "- \"Spiegami la varianza di questo dataset.\""
        )
    },
    "DeepSeek Math 7B": {
        "descrizione": (
            "Gratuito - Locale (LM Studio)\n"
            "Modello specializzato in calcolo numerico, regressioni, forecasting e statistica.\n"
            "Non usa l'API OpenAI, funziona offline con LM Studio.\n"
            "📈 Ottimo per analisi numeriche e modellazione predittiva (es. Prophet)."
        ),
        "esempi": (
            "- \"Prevedi l'andamento dei ricavi.\"\n"
            "- \"Costruisci un modello Prophet.\"\n"
            "- \"Esegui regressione lineare su questi dati.\""
        )
    },
    "gpt-3.5-turbo": {
        "descrizione": (
            "A pagamento - OpenAI API\n"
            "Modello cloud gestito da OpenAI, più economico rispetto a GPT-4.\n"
            "Richiede una API Key OpenAI e una connessione internet attiva.\n"
            "⚡ Efficiente per task generali, supporto, riassunti, traduzioni e query semplici."
        ),
        "esempi": (
            "- \"Traduci questo testo in inglese.\"\n"
            "- \"Riassumi questi appunti.\"\n"
            "- \"Dammi una definizione semplice di overfitting.\""
        )
    },
    "gpt-4-turbo": {
        "descrizione": (
            "A pagamento - OpenAI API\n"
            "Modello più avanzato di OpenAI, con prestazioni superiori su compiti complessi.\n"
            "Richiede una API Key e connessione internet.\n"
            "🧠 Ideale per data science, analisi di dataset, generazione di codice Python, debug e strategia aziendale."
        ),
        "esempi": (
            "- \"Analizza un file CSV complesso.\"\n"
            "- \"Crea un modello di classificazione con Python.\"\n"
            "- \"Suggerisci una strategia di ottimizzazione KPI.\""
        )
    }
}

# === STILE CSS PER LA TABELLA ===
style_html = """
    <style>
    table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background-color: #1e1e1e;
        border-radius: 0;
        overflow: hidden;
        font-size: 17px;
    }
    th, td {
        border: 1px solid #333;
        padding: 10px;
        text-align: left;
        color: #e0e0e0;
    }
    th {
        background-color: #292929;
        color: #00c8ff;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #2c2c2c;
    }
    tr:hover {
        background-color: #444444;
    }
    </style>
"""

# === CONTENUTO HTML DELLA TABELLA ===
table_html = """
    <table>
        <tr>
            <th>🧠 Modello</th>
            <th>Descrizione Uso Consigliato</th>
            <th>Esempi di Domande Ideali</th>
        </tr>
"""

for modello, dettagli in available_models.items():
    esempi_html = dettagli["esempi"].replace("\n", "<br>")
    table_html += f"""
        <tr>
            <td>{modello}</td>
            <td>{dettagli['descrizione']}</td>
            <td>{esempi_html}</td>
        </tr>
    """

table_html += "</table>"
# === SELEZIONE MODELLO IN SIDEBAR ===
st.sidebar.header("⚙️ Configurazione Modello")
modello_selezionato = st.sidebar.selectbox(
    "Seleziona il modello da utilizzare:",
    list(available_models.keys())
)
st.session_state["model_selected"] = modello_selezionato
# === VERIFICA STATO SERVER LM STUDIO ===
LMSTUDIO_IP = "192.168.1.111"
LMSTUDIO_PORT = 1234

if check_lmstudio_alive(host=LMSTUDIO_IP, port=LMSTUDIO_PORT):
    st.sidebar.success(f"🟢 LM Studio è attivo su {LMSTUDIO_IP}:{LMSTUDIO_PORT}")
else:
    st.sidebar.error(f"🔴 LM Studio non è raggiungibile su {LMSTUDIO_IP}:{LMSTUDIO_PORT}")

MODEL_ID_MAP = {
    "Qwen2.5 7B Instruct 1M": "qwen2.5-7b-instruct-1m",
    "DeepSeek Math 7B": "deepseek-math:7b-instruct",
}

MODEL_NAME = MODEL_ID_MAP.get(st.session_state["model_selected"], "qwen:7b")

st.sidebar.subheader("🌐 Modalità di esecuzione")
modalita_esecuzione = st.sidebar.radio(
    "Scegli il motore:",
    ["LM Studio Locale", "OpenAI API"],
    key="modalita_modello"
)

if modalita_esecuzione == "OpenAI API":
    st.session_state["openai_api_key"] = st.sidebar.text_input("🔑 API Key OpenAI", type="password")
    
    modello_openai = st.sidebar.selectbox(
        "🤖 Scegli il modello OpenAI:",
        ["gpt-3.5-turbo", "gpt-4-turbo"],
        index=1  # Predefinito su GPT-4 Turbo
    )
    st.session_state["modello_openai_selezionato"] = modello_openai



# === VISUALIZZA MESSAGGIO CHIARO SOTTO IL TITOLO ===

# Titolo dinamico per il modello attivo
if st.session_state["modalita_modello"] == "OpenAI API":
    nome_modello_attivo = st.session_state.get("modello_openai_selezionato", "gpt-4-turbo")
else:
    nome_modello_attivo = st.session_state["model_selected"]

st.markdown(f"""
<div style="padding: 10px; border-radius: 10px; background-color: #e6f7ff; color: #005c99; margin-bottom: 20px;">
    <h4 style="margin-bottom:5px;">🧠 Modello Attivo: <b>{nome_modello_attivo}</b></h4>
    <small>Consulta la legenda sotto per suggerimenti sulle domande migliori!</small>
</div>
""", unsafe_allow_html=True)


# === VISUALIZZAZIONE TABELLA ===
with st.expander("Legenda Modelli & Domande Consigliate", expanded=True):
    full_html = f"""
    {style_html}
    {table_html}
    """
    components.html(full_html, height=260, scrolling=True)

import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
import os
import datetime
import re
from prophet import Prophet
from pptx import Presentation
from pptx.util import Inches
import socket
import io
import contextlib

# === CONFIG ===
SERVER_URL = "http://192.168.1.111:1234/v1/chat/completions"
MODEL_ID_MAP = {
    "Qwen2.5 7B Instruct 1M":"qwen2.5-7b-instruct-1m",
    "DeepSeek Math 7B":"deepseek-math-7b-instruct"
}
MODEL_NAME = MODEL_ID_MAP.get(st.session_state["model_selected"], "qwen2.5-7b-instruct-1m")
DB_PATH = "jarvis_brain.sqlite"
LOG_DIR = "chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# === FUNZIONI DI SUPPORTO ===
client = OpenAI(api_key=st.session_state.get("openai_api_key", ""))

def pulisci_testo(testo):
    testo = re.sub(r'\*\*(.*?)\*\*', r'**\1**', testo)
    testo = re.sub(r'\_\_(.*?)\_\_', r'_\1_', testo)
    testo = re.sub(r'<[^>]+>', '', testo)
    testo = ''.join(c for c in testo if (ord(c) < 128 or c in '\n\t .,;:-_+=*/^()[]{}\u20AC€$%'))
    return testo.strip()

def chiedi_a_jarvis_locale(domanda):
    try:
        response = requests.post(SERVER_URL, json={
            "messages": [
                {"role": "system", "content": "Agisci come un esperto analista dati. Rispondi sempre in italiano."},
                {"role": "user", "content": domanda}
            ],
            "model": MODEL_NAME,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": 2000
        })
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "Nessuna risposta dal modello.")
    except Exception as e:
        return f"Errore durante la comunicazione: {e}"

def chiedi_a_openai(prompt_utente):
    modello = st.session_state.get("modello_openai_selezionato", "gpt-4-turbo")
    try:
        # Aggiungi il nuovo messaggio alla cronologia
        st.session_state["chat_history"].append({"role": "user", "content": prompt_utente})

        response = client.chat.completions.create(
            model=modello,
            messages=st.session_state["chat_history"],
            temperature=0.3,
            max_tokens=2000
        )

        reply = response.choices[0].message.content
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        return reply

    except Exception as e:
        return f"Errore OpenAI: {e}"



def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS memoria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domanda TEXT,
            risposta TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def salva_memoria(domanda, risposta):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO memoria (domanda, risposta, timestamp) VALUES (?, ?, ?)",
              (domanda, risposta, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def costruisci_prompt(contenuto, preview):
    return (
        "Agisci come un analista dati aziendale esperto.\n"
        f"- Numero righe: {preview['righe_totali']}\n"
        f"- Numero colonne: {preview['colonne_totali']}\n"
        f"- Percentuale valori nulli: {preview['percentuale_nulli']}%\n"
        f"- Colonne numeriche: {', '.join(preview['colonne_numeriche'])}\n"
        f"- Colonne categoriali: {', '.join(preview['colonne_categoriali'])}\n\n"
        "Analizza:\n"
        "- Tendenze principali\n- Anomalie\n- Suggerimenti pratici\n\n"
        f"Dati:\n\"\"\"\n{contenuto}\n\"\"\""
    )
    return prompt

def calcola_preview(df):
    preview = {
        "righe_totali": df.shape[0],
        "colonne_totali": df.shape[1],
        "percentuale_nulli": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
        "colonne_numeriche": df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
        "colonne_categoriali": df.select_dtypes(include=['object']).columns.tolist()
    }
    return preview


def genera_riassunto_df(df):
    descrizione = df.describe(include='all').transpose()
    descrizione["median"] = df.median(numeric_only=True)
    descrizione["mode"] = df.mode(numeric_only=True).iloc[0]
    descrizione["varianza"] = df.var(numeric_only=True)
    descrizione["skewness"] = df.skew(numeric_only=True)
    descrizione["kurtosis"] = df.kurt(numeric_only=True)
    return descrizione.fillna("-").to_string()

def mostra_risposta_formattata(risposta):
    blocchi = re.split(r'(\$\$.*?\$\$|\$.*?\$|\[.*?\])', risposta, flags=re.DOTALL)
    for blocco in blocchi:
        blocco = blocco.strip()
        if (blocco.startswith('$$') and blocco.endswith('$$')) or \
           (blocco.startswith('$') and blocco.endswith('$')) or \
           (blocco.startswith('[') and blocco.endswith(']')):
            formula = blocco.strip('$').strip('[]')
            st.latex(formula)
        elif blocco.startswith("- ") or blocco.startswith("• ") or blocco.startswith("* "):
            for riga in blocco.split('\n'):
                if riga.strip().startswith(("-", "•", "*")):
                    st.markdown(f"- {riga[1:].strip()}")
                else:
                    st.write(riga)
        elif blocco.startswith("#"):
            st.markdown(blocco)
        else:
            if blocco:
                st.write(blocco)

def forecast_prophet(df, data_col, target_col, periods, freq):
    data = df[[data_col, target_col]].dropna()
    data = data.rename(columns={data_col: 'ds', target_col: 'y'})
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def check_lmstudio_alive(host="localhost", port=1234, timeout=2):
    """Verifica se il server LM Studio è attivo sulla porta specificata."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# Inizializza df se non è ancora presente
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()
    if "chat_history" not in st.session_state:
     st.session_state["chat_history"] = [
        {"role": "system", "content": "Agisci come un analista dati esperto. Rispondi sempre in italiano."}
    ]
if "codici_eseguiti_log" not in st.session_state:
    st.session_state["codici_eseguiti_log"] = []
df = st.session_state["df"]

# Carica CSV
st.sidebar.subheader("Carica un file CSV")
uploaded_file = st.sidebar.file_uploader("Drag & Drop CSV", type=["csv"], key="csv_upload")

# Se è stato caricato un nuovo file, aggiornalo
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df

# === CREA GRAFICO ===
st.sidebar.header("🎨 Crea Grafico")

# Parametri avanzati per il grafico
st.sidebar.subheader("Parametri Avanzati Grafico")
fig_width = st.sidebar.slider("Larghezza Grafico", 5, 20, 10, key="grafico_larghezza")
fig_height = st.sidebar.slider("Altezza Grafico", 4, 15, 6, key="grafico_altezza")
grid_option = st.sidebar.checkbox("Mostra griglia", value=True, key="grafico_griglia")
rotate_xticks = st.sidebar.checkbox("Ruota etichette X", value=True, key="grafico_ruota_xticks")

# === SELEZIONE COLONNE PER GRAFICO ===
# Inizializza df se non è ancora presente
if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

df = st.session_state["df"]

# Se è stato caricato un nuovo file, aggiorna il DataFrame
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df

# Selezione colonne per il grafico
col_x = st.sidebar.selectbox("Colonna X", df.columns.tolist() if not df.empty else ["Nessuna colonna disponibile"], key="col_x")
col_y = st.sidebar.selectbox("Colonna Y (opzionale)", ["None"] + df.columns.tolist() if not df.empty else ["None"], key="col_y")

grafico_tipo = st.sidebar.radio("Tipo di Grafico", ("Linea", "Istogramma", "Dispersione"), key="tipo_grafico")

# === PREVISIONE CON PROPHET ===
st.sidebar.header("🔮 Previsione con Prophet")
st.sidebar.subheader("📈 Parametri Forecast Prophet")
colonna_data = st.sidebar.selectbox("Seleziona Colonna Data", df.columns, key="colonna_data")
colonna_target = st.sidebar.selectbox("Seleziona Colonna Target", df.columns, key="colonna_target")
periods = st.sidebar.number_input("Mesi da prevedere", min_value=1, max_value=60, value=12, key="forecast_periods")
freq = st.sidebar.selectbox("Frequenza", ["D", "W", "M"], index=2, key="forecast_freq")


if st.sidebar.button(" Crea Forecast"):
    try:
        # Validazione colonne
        if not pd.api.types.is_datetime64_any_dtype(df[colonna_data]):
            df[colonna_data] = pd.to_datetime(df[colonna_data], errors='coerce')
            if df[colonna_data].isnull().all():
                st.error(f"La colonna '{colonna_data}' non contiene dati validi come date.")
                st.stop()

        if not pd.api.types.is_numeric_dtype(df[colonna_target]):
            st.error(f"La colonna '{colonna_target}' deve essere numerica per usare Prophet.")
            st.stop()

        # Forecast
        forecast = forecast_prophet(df, colonna_data, colonna_target, periods, freq)
        st.success("Previsione completata")
        st.dataframe(forecast)

        # Plot
        fig2, ax2 = plt.subplots()
        ax2.plot(forecast['ds'], forecast['yhat'], label='Previsione')
        ax2.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='lightblue', alpha=0.5)
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Valore previsto')
        ax2.set_title('Forecast con Prophet')
        ax2.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Errore nella previsione: {e}")


# === CHAT ===
st.header("💬 Chat con Jarvis")
user_input = st.chat_input("Scrivi una domanda...")

if user_input:
    if st.session_state["modalita_modello"] == "LM Studio Locale":
        if check_lmstudio_alive():
            risposta = chiedi_a_jarvis_locale(user_input)
        else:
            risposta = "❌ LM Studio non è attivo. Avvialo e riprova."
    else:
        if not st.session_state.get("openai_api_key"):
            risposta = "❌ Inserisci la tua API Key di OpenAI nella sidebar."
        else:
            risposta = chiedi_a_openai(user_input)

    risposta = pulisci_testo(risposta)
    with st.chat_message("Utente"):
        st.write(user_input)
    with st.chat_message("Jarvis"):
        st.markdown(risposta)
    salva_memoria(user_input, risposta)
    def estrai_codice_python(testo):
        pattern = r"```python(.*?)```"
        match = re.search(pattern, testo, re.DOTALL)
        return match.group(1).strip() if match else ""

    codice_estratto = estrai_codice_python(risposta)

    if codice_estratto:
        st.divider()
        st.subheader("🧪 Codice Python suggerito dal modello")
        codice_python = st.text_area("📜 Modifica il codice prima dell'esecuzione, se necessario:", codice_estratto, height=200, key="codice_suggerito")

        if st.button("▶️ Esegui Codice Suggerito"):
            with st.expander("🔍 Output del Codice", expanded=True):
                f = io.StringIO()
                try:
                    with contextlib.redirect_stdout(f):
                        exec(codice_python, {"df": df, "pd": pd, "plt": plt, "st": st})
                        st.session_state["codici_eseguiti_log"].append(codice_python)
                    st.success(" Codice eseguito correttamente.")
                    output = f.getvalue()
                    if output:
                        st.code(output)
                except Exception as e:
                    st.error(f" Errore nell'esecuzione del codice:\n{e}")
    if st.session_state["codici_eseguiti_log"]:
      st.divider()
      st.subheader("📚 Log dei Codici Eseguiti (sessione attuale)")
    for i, codice in enumerate(st.session_state["codici_eseguiti_log"], 1):
        with st.expander(f"🔢 Codice #{i}", expanded=False):
            st.code(codice, language="python")
    
if st.sidebar.button("🔄 Reset Conversazione"):
    st.session_state["chat_history"] = [
        {"role": "system", "content": "Agisci come un analista dati esperto. Rispondi sempre in italiano."}
    ]
    st.success("Memoria della conversazione azzerata.")


st.sidebar.divider()
st.sidebar.info("Sviluppato da Piero 🤖")
