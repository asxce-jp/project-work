"""
Modulo per la valutazione delle performance dei modelli di classificazione.

Valuta i modelli addestrati sul test set e genera:
- Report di classificazione (precision, recall, f1-score)
- Confusion matrix visualizzate e salvate come immagini PNG
"""
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import basic_clean, make_train_test

# Percorso del dataset per la valutazione
DATA = "data/synthetic_reviews.csv"

# Directory dove salvare gli output della valutazione (confusion matrix)
OUTPUT_DIRECTORY = Path("outputs"); OUTPUT_DIRECTORY.mkdir(exist_ok=True)

def build_text(df):
    """
    Costruisce le feature testuali combinando title e body delle recensioni.
    
    Args:
        df (pd.DataFrame): DataFrame con colonne 'title' e 'body'
    
    Returns:
        pd.Series: Serie di stringhe preprocessate pronte per la predizione
    """
    # Concatena title e body gestendo valori NaN, poi applica il preprocessing
    return (df["title"].fillna("") + " " + df["body"].fillna("")).map(basic_clean)

def evaluate_task(model_path, df, y_col, out_png):
    """
    Valuta le performance di un modello di classificazione su un dataset di test.
    
    Carica il modello, genera predizioni sul test set, stampa le metriche
    e salva la confusion matrix come immagine PNG.
    
    Args:
        model_path (str): Percorso del modello serializzato (.joblib)
        df (pd.DataFrame): Dataset completo con le recensioni
        y_col (str): Nome della colonna target (es. 'department', 'sentiment')
        out_png (str): Nome del file PNG per salvare la confusion matrix
    """
    # Carica il modello pre-addestrato dal file
    model = load(model_path)
    
    # Crea il split train/test (scarta il train, usa solo il test)
    _, test = make_train_test(df, y_col=y_col)
    
    # Prepara le feature testuali concatenando title e body
    x_test = build_text(test)
    
    # Estrae le etichette vere dal dataset di test
    y_true = test[y_col]
    
    # Genera le predizioni del modello sul test set
    y_pred = model.predict(x_test)
    
    # Stampa il report di classificazione (precision, recall, f1-score)
    print(classification_report(y_true, y_pred, digits=3))
    
    # Calcola la confusion matrix con le classi ordinate
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    
    # Crea il display della confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_true.unique()))
    
    # Visualizza la matrice con etichette ruotate per leggibilità
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    
    # Salva la figura in formato PNG ad alta risoluzione
    plt.savefig(OUTPUT_DIRECTORY / out_png, dpi=150)
    
    # Chiude la figura per liberare memoria
    plt.close()
    
def main():
    """
    Entry point principale: carica il dataset e valuta entrambi i modelli.
    
    Genera report di classificazione e confusion matrix per:
    - Classificatore di reparto (3 classi)
    - Classificatore di sentiment (2 classi)
    """
    # Carica il dataset sintetico
    df = pd.read_csv(DATA)
    
    # Valutazione classificatore di reparto
    print("\n=== Evaluating Department Classifier ===")
    evaluate_task("models/department_classifier.joblib", df, "department", "confusion_matrix_department.png")
    
    # Valutazione classificatore di sentiment
    print("\n=== Evaluating Sentiment Classifier ===")
    evaluate_task("models/sentiment_classifier.joblib", df, "sentiment", "confusion_matrix_sentiment.png")
    
if __name__ == "__main__":
    main()