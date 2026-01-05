"""
Modulo per l'addestramento dei modelli di classificazione.

Addestra due pipeline di machine learning:
- Classificatore di reparto (LinearSVC)
- Classificatore di sentiment (Logistic Regression)

Entrambi i modelli usano TF-IDF come feature extraction.
"""
import pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from utils import basic_clean, make_train_test

# Percorso del dataset di training
DATA = "data/synthetic_reviews.csv"

# Directory dove salvare i modelli addestrati
MODEL_DIRECTORY = Path("models"); MODEL_DIRECTORY.mkdir(exist_ok=True)

def build_text(df):
    """
    Costruisce le feature testuali combinando title e body delle recensioni.
    
    Args:
        df (pd.DataFrame): DataFrame con colonne 'title' e 'body'
    
    Returns:
        pd.Series: Serie di stringhe preprocessate pronte per la vectorization
    """
    # Concatena title e body gestendo valori NaN, poi applica il preprocessing
    return (df["title"].fillna("") + " " + df["body"].fillna("")).map(basic_clean)

def make_pipeline(task: str) -> Pipeline:
    """
    Crea una pipeline scikit-learn per il task specificato.
    
    Args:
        task (str): Tipo di task ('department' o 'sentiment')
    
    Returns:
        Pipeline: Pipeline con TfidfVectorizer e classificatore appropriato
    
    Raises:
        ValueError: Se il task non è riconosciuto
    """
    # Vectorizer TF-IDF con unigrammi e bigrammi, ignora termini troppo rari
    vec = TfidfVectorizer(ngram_range=(1,2), analyzer='word', min_df=2)
    
    # Scelta del classificatore in base al task
    if task == "department":
        # LinearSVC per classificazione multiclasse (3 reparti)
        clf = LinearSVC()
    elif task == "sentiment":
        # Logistic Regression per classificazione binaria (positive/negative)
        clf = LogisticRegression(max_iter=200)
    else:
        raise ValueError("Unknown task")
    
    # Costruisce la pipeline: vectorization -> classification
    return Pipeline([("vectorizer", vec), ("classifier", clf)])

def train_model(df, y_col: str, model_name: str):
    """
    Addestra e valuta un modello di classificazione, poi lo salva su disco.
    
    Args:
        df (pd.DataFrame): Dataset completo delle recensioni
        y_col (str): Nome della colonna target ('department' o 'sentiment')
        model_name (str): Nome del file del modello (senza estensione)
    
    Output:
        Stampa il classification report e salva il modello in models/
    """
    # Divide il dataset in train e test set
    train_df, test_df = make_train_test(df, y_col=y_col)
    
    # Crea la pipeline appropriata per il task
    pipe = make_pipeline(task=y_col)
    
    # Addestra la pipeline sul training set
    pipe.fit(build_text(train_df), train_df[y_col])
    
    # Genera predizioni sul test set per la valutazione
    y_pred = pipe.predict(build_text(test_df))
    
    # Stampa metriche di performance (precision, recall, f1-score)
    print (f"\n=== {model_name} ===")
    print (classification_report(test_df[y_col], y_pred, digits=3))
    
    # Serializza la pipeline addestrata su disco
    dump(pipe, MODEL_DIRECTORY / f"{model_name}.joblib")

def main():
    """
    Entry point principale: carica il dataset e addestra entrambi i modelli.
    """
    # Carica il dataset sintetico di recensioni
    df = pd.read_csv(DATA)
    
    # Addestra il classificatore di reparto (3 classi: Housekeeping, Reception, F&B)
    train_model(df, y_col="department", model_name="department_classifier")
    
    # Addestra il classificatore di sentiment (2 classi: positive, negative)
    train_model(df, y_col="sentiment", model_name="sentiment_classifier")
    
if __name__ == "__main__":
    main()