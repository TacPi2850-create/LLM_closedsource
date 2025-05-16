````markdown
🤖 Jarvis Dual Mode - Closed Source

**Jarvis** è un'applicazione interattiva sviluppata in **Streamlit** che integra due modalità operative:

- 🧠 **LLM locali** via [LM Studio](https://lmstudio.ai/)
- ☁️ **LLM cloud-based** via OpenAI API (`gpt-3.5-turbo` e `gpt-4-turbo`)

Il progetto è pensato per analisti, studenti e professionisti che desiderano un assistente AI in grado di:
- comprendere domande in linguaggio naturale,
- analizzare dataset CSV,
- generare ed eseguire codice Python dinamico,
- produrre forecast tramite **Facebook Prophet**,
- interagire tramite una chat intelligente con memoria persistente (SQLite).

---

## ⚙️ Caratteristiche principali

| Funzione                    | Descrizione Tecnica |
|-----------------------------|---------------------|
| **Selezione modello**       | Supporta 4 modelli: 2 locali (Qwen, DeepSeek) e 2 OpenAI (GPT-3.5, GPT-4) |
| **LM Studio Check**         | Controllo automatico dello stato di LM Studio sulla rete locale |
| **Upload CSV**              | Caricamento di file `.csv` per analisi automatica |
| **Preview Statistiche**     | Calcolo righe, colonne, nulli, colonne numeriche/categoriali |
| **Analisi descrittiva**     | Statistiche: media, mediana, varianza, skewness, kurtosis |
| **Forecast con Prophet**    | Previsioni su serie temporali con intervallo di confidenza |
| **Generazione codice Python** | Estrazione e possibilità di eseguire codice prodotto dal modello |
| **Log codice eseguito**     | Storico dei codici Python eseguiti durante la sessione |
| **Memoria chat persistente** | Conversazioni archiviate in `jarvis_brain.sqlite` |
| **Reset conversazione**     | Pulizia della memoria della sessione corrente |

---

## 🧠 Modelli supportati

| Modalità | Modello | Descrizione |
|---------|---------|-------------|
| **Locale (LM Studio)** | `Qwen2.5 7B Instruct 1M` | Spiegazioni aziendali, analisi concettuali |
| | `DeepSeek Math 7B` | Calcolo statistico, regressioni, forecasting |
| **Cloud (OpenAI)** | `gpt-3.5-turbo` | Economico, performante per task leggeri |
| | `gpt-4-turbo` | Avanzato, ideale per analisi complesse e codice |

---

## 🧪 Esecuzione locale

### 🔁 Requisiti

- Python 3.10+
- LM Studio attivo in locale (default IP `192.168.1.111:1234`)
- Una chiave API OpenAI valida (opzionale)
- Dipendenze Python installate (vedi sotto)

### 🛠️ Setup ambiente

```bash
git clone https://github.com/TUO_USERNAME/jarvis-dual-mode.git
cd jarvis-dual-mode

# Ambiente virtuale (opzionale ma consigliato)
python -m venv venv
source venv/bin/activate    # Su Windows: venv\Scripts\activate

# Installazione pacchetti
pip install -r requirements.txt

# Avvio app Streamlit
streamlit run app.py
````

---

## 🔐 Configurazione API

Crea un file `.env` nella root del progetto:

```env
OPENAI_API_KEY=sk-INSERISCI-LA-TUA-CHIAVE
```

> ⚠️ Questo è necessario solo se usi i modelli `gpt-3.5-turbo` o `gpt-4-turbo`.

---

## 🧬 Esempio d'uso: Forecast con Prophet

1. Carica un file `.csv` con una colonna temporale e una colonna numerica.
2. Vai nella sezione **🔮 Previsione con Prophet**.
3. Seleziona le colonne corrette e il numero di mesi da prevedere.
4. Ottieni:

   * Tabella con i valori `yhat`, `yhat_lower`, `yhat_upper`
   * Grafico a linea con bande di confidenza

---

## 🧾 Database interno (SQLite)

Le interazioni sono salvate in `jarvis_brain.sqlite` con questa struttura:

```sql
CREATE TABLE memoria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domanda TEXT,
    risposta TEXT,
    timestamp TEXT
);
```

Puoi accedere alla cronologia anche in modo esterno con strumenti SQL.

---

## 🧠 Esecuzione del codice Python generato

Quando il modello suggerisce codice Python, l'utente può:

* **visualizzarlo**
* **modificarlo**
* **eseguirlo dinamicamente** all'interno di `Streamlit` con output diretto
* **salvare il codice nel log della sessione**

---

## 📚 Esempi di prompt utili

| Scopo       | Prompt                                                  |
| ----------- | ------------------------------------------------------- |
| Analisi KPI | `"Spiegami i KPI di questo report."`                    |
| Forecast    | `"Prevedi i ricavi con Prophet"`                        |
| Statistica  | `"Calcola la skewness del dataset"`                     |
| Codice      | `"Scrivi un codice Python per una regressione lineare"` |

---

## 📦 Struttura del progetto

```
jarvis-dual-mode/
│
├── app.py                # Codice principale
├── .env.example          # Esempio di configurazione
├── requirements.txt      # Librerie Python necessarie
├── .gitignore            # File ignorati da Git
├── README.md             # Documentazione tecnica
├── jarvis_brain.sqlite   # (autogenerato)
└── chat_logs/            # (autogenerato)
```

---

## 🧠 Autore

**Piero Crispin Tacunan Eulogio**
Analista dati, sviluppatore e appassionato di intelligenza artificiale distribuita e open-source.

---

## 📜 Licenza

Questo progetto è open source e distribuito sotto licenza **MIT**.
