"""
Applicazione web Streamlit per la classificazione di recensioni alberghiere.

Fornisce due modalità di utilizzo:
1. Predizione singola: Inserimento manuale di una recensione
2. Predizione batch: Upload di file CSV con recensioni multiple

Per ogni recensione predice:
- Reparto (Housekeeping, Reception, F&B)
- Sentiment (positive, negative)
"""
import streamlit as st
import pandas as pd
import datetime as dt
from joblib import load
import io

# Configurazione della pagina Streamlit
st.set_page_config(page_title="Hotel Review Classifier", layout="centered")
st.title("Hotel Review Classifier + Sentiment Analysis")

@st.cache_resource
def load_models():
    """
    Carica i modelli di classificazione pre-addestrati.
    
    Usa @st.cache_resource per caricare i modelli una sola volta
    e condividerli tra tutte le sessioni utente, ottimizzando le performance.
    
    Returns:
        tuple: (department_classifier, sentiment_classifier) - I due modelli caricati
    """
    return load("models/department_classifier.joblib"), load("models/sentiment_classifier.joblib")

# Carica i modelli all'avvio dell'applicazione (con caching)
DEPARTMENT, SENTIMENT = load_models()

def clean(s):
    """
    Preprocessa il testo di una recensione per la predizione.
    
    Operazioni:
    - Conversione in minuscolo
    - Rimozione punteggiatura (mantiene lettere accentate italiane)
    - Normalizzazione spazi multipli
    
    Args:
        s (str): Testo da preprocessare
    
    Returns:
        str: Testo pulito e normalizzato
    """
    import re
    # Converte in minuscolo per uniformità
    s = s.lower()
    # Rimuove punteggiatura mantenendo parole, spazi e accenti
    s = re.sub(r"[^\w\sàèéìòóù]", " ", s)
    # Normalizza spazi multipli e rimuove spazi iniziali/finali
    return " ".join(s.split())

# Interfaccia Utente: Due modalità in tab separate
tab1, tab2 = st.tabs(["Single Review Prediction", "Batch CSV"])

# TAB 1: Predizione di una singola recensione
with tab1:
    # Input per il titolo della recensione
    title = st.text_input("Review Title", "")
    
    # Input per il corpo della recensione (area di testo multilinea)
    body = st.text_area("Review Body", "", height=160, placeholder="Write the review text here...")
    
    # Bottone per attivare la predizione
    if st.button("Predict"):
        # Combina title e body, poi applica preprocessing
        text = clean(f"{title} {body}")
        
        # Esegue le predizioni con entrambi i modelli
        department = DEPARTMENT.predict([text])[0]
        sentiment = SENTIMENT.predict([text])[0]
        
        # Mostra i risultati in un messaggio di successo
        st.success(f"Predicted Department: **{department}** | Predicted Sentiment: **{sentiment}**")
        
# TAB 2: Predizione batch da file CSV
with tab2:
    # Widget per l'upload del file CSV
    uploaded_file = st.file_uploader("Upload CSV file with 'id','title' and 'body' columns", type=["csv"])
    
    if uploaded_file is not None:
        # Carica il CSV in un DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Prepara i testi combinando title e body con preprocessing
        texts = (df["title"].fillna("") + " " + df["body"].fillna("")).map(clean)
        
        # Esegue predizioni batch per entrambi i modelli
        df["predicted_department"] = DEPARTMENT.predict(texts)
        df["predicted_sentiment"] = SENTIMENT.predict(texts)
        
        # Aggiunge timestamp della predizione
        df["timestamp"] = pd.Timestamp.now().isoformat()
        
        # Mostra un'anteprima delle prime 20 righe con le predizioni
        st.dataframe(df.head(20))
        
        # Prepara il CSV per il download
        buf = io.StringIO(); df.to_csv(buf, index=False)
        
        # Bottone per scaricare il CSV completo con le predizioni
        st.download_button("Download Predictions CSV", buf.getvalue(), file_name=f"predictions_batch_{dt.datetime.now():%Y-%m-%d_%H-%M-%S}.csv", mime="text/csv")