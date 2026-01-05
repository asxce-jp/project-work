"""
Modulo di utility per il preprocessing e la gestione dei dati.

Fornisce funzioni helper per:
- Pulizia e normalizzazione del testo
- Divisione stratificata del dataset in train/test
"""
import re
from sklearn.model_selection import train_test_split

def basic_clean(s: str) -> str:
    """
    Preprocessa una stringa di testo per la feature extraction.
    
    Operazioni applicate:
    1. Conversione in minuscolo
    2. Rimozione punteggiatura (mantiene caratteri accentati italiani)
    3. Normalizzazione spazi multipli
    4. Rimozione spazi iniziali/finali
    
    Args:
        s (str): Testo da preprocessare
    
    Returns:
        str: Testo pulito e normalizzato
    """
    # Converte tutto in minuscolo per uniformità
    s = s.lower()
    
    # Rimuove punteggiatura mantenendo lettere, numeri, spazi e accenti italiani
    s = re.sub(r"[^\w\sàèéìòóù]", " ", s)
    
    # Normalizza spazi multipli in uno singolo e rimuove spazi iniziali/finali
    s = re.sub(r"\s+", " ", s).strip()
    
    return s

def make_train_test(df, y_col, test_size=0.2, random_state=42, stratify=True):
    """
    Suddivide un DataFrame in set di train e test con opzione di stratificazione.
    
    Args:
        df (pd.DataFrame): Dataset completo da dividere
        y_col (str): Nome della colonna target per la stratificazione
        test_size (float): Proporzione del test set (default: 0.2 = 20%)
        random_state (int): Seed per riproducibilità (default: 42)
        stratify (bool): Se True, mantiene la distribuzione delle classi (default: True)
    
    Returns:
        tuple: (train_df, test_df) - DataFrame di training e test
    """
    # Usa la colonna target per stratificare, oppure None per split casuale
    stratify_column = df[y_col] if stratify else None
    
    # Divide il dataset mantenendo la proporzione delle classi nel target
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_column)
    
    return train_df, test_df