import re
from sklearn.model_selection import train_test_split

# Pulizia testo: converte in minuscolo, rimuove punteggiatura e normalizza spazi
def basic_clean(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\sàèéìòóù]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Suddivide il DataFrame in train e test, con opzione di stratificazione
def make_train_test(df, y_col, test_size=0.2, random_state=42, stratify=True):
    stratify_column = df[y_col] if stratify else None
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_column)
    return train_df, test_df