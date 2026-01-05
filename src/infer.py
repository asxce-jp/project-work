"""
Modulo per l'inferenza dei modelli di classificazione.

Carica i modelli pre-addestrati e fornisce funzioni per predire
il reparto e il sentiment di recensioni singole o batch.
"""
import pandas as pd
from pathlib import Path
from joblib import load
from utils import basic_clean

# Carica il modello di classificazione del reparto (Housekeeping, Reception, F&B)
DEPARTMENT = load("models/department_classifier.joblib")

# Carica il modello di classificazione del sentiment (positive, negative)
SENTIMENT = load("models/sentiment_classifier.joblib")

def predict_one(title: str, body: str):
    """
    Funzione per predire il reparto e il sentiment per una singola recensione.
    
    Args:
        title (str): Titolo della recensione (può essere None o vuoto)
        body (str): Corpo della recensione (può essere None o vuoto)
    
    Returns:
        tuple: (department, sentiment) - Reparto e sentiment predetti
    """
    # Concatena title e body gestendo valori None, poi applica il preprocessing
    text = basic_clean((title or "") + " " + (body or ""))
    
    # Predice il reparto usando il primo modello
    dept = DEPARTMENT.predict([text])[0]
    
    # Predice il sentiment usando il secondo modello
    sent = SENTIMENT.predict([text])[0]
    
    return dept, sent

def predict_csv(input_csv: str, output_csv: str = "outputs/predictions_batch.csv"):
    """
    Funzione per predire reparto e sentiment per un batch di recensioni da file CSV.
    
    Args:
        input_csv (str): Percorso del file CSV di input con colonne 'title' e 'body'
        output_csv (str): Percorso del file CSV di output (default: outputs/predictions_batch.csv)
    
    Output:
        Salva un CSV contenente le colonne originali più predicted_department,
        predicted_sentiment e timestamp della predizione.
    """
    # Carica il CSV di input
    df = pd.read_csv(input_csv)
    
    # Prepara i testi combinando title e body, gestisce valori NaN
    texts = (df["title"].fillna("") + " " + df["body"].fillna("")).map(basic_clean)
    
    # Esegue predizioni in batch per il reparto
    df["predicted_department"] = DEPARTMENT.predict(texts)
    
    # Esegue predizioni in batch per il sentiment
    df["predicted_sentiment"] = SENTIMENT.predict(texts)
    
    # Aggiunge timestamp ISO 8601 per tracciare quando è stata fatta la predizione
    df["timestamp"] = pd.Timestamp.now().isoformat()
    
    # Crea la directory outputs se non esiste
    Path("outputs").mkdir(exist_ok=True)
    
    # Salva il dataframe arricchito con le predizioni
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    
if __name__ == "__main__":
    # Entry point per l'esecuzione da linea di comando
    import sys
    
    # Legge il percorso del CSV da riga di comando, altrimenti usa il default
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "data/samples_to_predict.csv"