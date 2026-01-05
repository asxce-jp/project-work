"""
Modulo per la generazione di un dataset sintetico di recensioni alberghiere.

Genera recensioni sintetiche bilanciate per addestrare modelli di classificazione:
- 3 reparti: Housekeeping, Reception, F&B
- 2 sentiment: positive, negative

Il dataset è completamente bilanciato con distribuzione uniforme tra tutte le classi.
"""
import random, uuid, csv
from pathlib import Path

# Fissa il seed per riproducibilità del dataset
random.seed(42)

# Definizione delle classi target per i due task di classificazione
DEPARTMENTS = ["Housekeeping", "Reception", "F&B"]  # 3 reparti dell'hotel
SENTIMENTS  = ["positive", "negative"]  # 2 polarità di sentiment

# === Lessici sintetici per generare recensioni realistiche ===

# Lessico per Housekeeping (pulizia, camere, comfort)
LEX_HK_POS  = ["camera pulita e profumata", "staff cordiale e disponibile", "letto comodo", "colazione abbondante"]
LEX_HK_NEG  = ["camera sporca", "staff scortese", "letto scomodo", "lenzuola macchiate"]

# Lessico per Reception (accoglienza, check-in, pagamenti)
LEX_RC_POS  = ["check-in veloce", "personale accogliente", "pagamento senza problemi", "accoglienza calorosa"]
LEX_RC_NEG  = ["lunghe attese al check-in", "personale scortese", "errori nel pagamento", "servizio lento"]

# Lessico per F&B (ristorazione, colazione, servizio)
LEX_FB_POS  = ["colazione ricca e varia", "ristorante eccellente", "cameriere attento", "porzioni abbondanti"]
LEX_FB_NEG  = ["colazione scarsa", "servizio lento", "menu limitato", "porzioni scarse"]

# Titoli generici per le recensioni (non specifici per reparto)
TITLES_POS  = ["Eccellente soggiorno!", "Servizio impeccabile", "Esperienza fantastica", "Consigliatissimo!"]
TITLES_NEG  = ["Deludente esperienza", "Servizio pessimo", "Non tornerò mai più", "Molto insoddisfatto"]

def synthesize_review(department, sentiment):
    """
    Genera una recensione sintetica combinando un titolo e un corpo appropriati.
    
    Seleziona casualmente frasi dai lessici specifici per reparto e sentiment,
    combinandole con titoli generici per creare recensioni realistiche.
    
    Args:
        department (str): Reparto dell'hotel ('Housekeeping', 'Reception', 'F&B')
        sentiment (str): Sentiment della recensione ('positive', 'negative')
    
    Returns:
        tuple: (title, phrase) - Titolo e corpo della recensione generata
    """
    # Seleziona una frase specifica dal lessico del reparto con il sentiment appropriato
    if department == "Housekeeping":
        phrase = random.choice(LEX_HK_POS if sentiment == "positive" else LEX_HK_NEG)
    elif department == "Reception":
        phrase = random.choice(LEX_RC_POS if sentiment == "positive" else LEX_RC_NEG)
    else:  # F&B department
        phrase = random.choice(LEX_FB_POS if sentiment == "positive" else LEX_FB_NEG)

    # Seleziona un titolo generico basato solo sul sentiment (indipendente dal reparto)
    title = random.choice(TITLES_POS if sentiment == "positive" else TITLES_NEG)
    
    return title, phrase

def main(n=360, output_path="data/synthetic_reviews.csv"):
    """
    Genera un dataset sintetico bilanciato di recensioni alberghiere.
    
    Il dataset contiene recensioni distribuite uniformemente tra:
    - 3 reparti (Housekeeping, Reception, F&B)
    - 2 sentiment (positive, negative)
    - Totale: 6 combinazioni con n/6 esempi ciascuna
    
    Args:
        n (int): Numero totale di recensioni da generare (default: 360)
        output_path (str): Percorso del file CSV di output (default: data/synthetic_reviews.csv)
    
    Output:
        Salva un CSV con colonne: id, title, body, department, sentiment
    """
    # Crea la directory data se non esiste
    Path("data").mkdir(exist_ok=True)
    
    # Lista per accumulare tutte le righe del dataset
    rows = []
    
    # Calcola il numero di recensioni per ogni combinazione reparto-sentiment
    # Esempio: 360 / (3 reparti × 2 sentiment) = 60 recensioni per combinazione
    reviews = n // (len(DEPARTMENTS) * len(SENTIMENTS))
    
    # Genera recensioni per ogni combinazione di reparto e sentiment
    for department in DEPARTMENTS:
        for sentiment in SENTIMENTS:
            for _ in range(reviews):
                # Genera un ID univoco esadecimale di 12 caratteri
                id = str(uuid.uuid4().hex[:12])
                
                # Sintetizza title e body della recensione
                title, body = synthesize_review(department, sentiment)
                
                # Aggiungi la recensione con i suoi metadati
                rows.append([id, title, body, department, sentiment])
                
    # Mescola casualmente le recensioni per evitare pattern ordinati
    random.shuffle(rows)
    
    # Scrive il dataset completo in formato CSV
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header del CSV
        writer.writerow(["id", "title", "body", "department", "sentiment"])
        # Tutte le recensioni
        writer.writerows(rows)
    
    print(f"Synthetic dataset with {n} reviews saved to {output_path}")

if __name__ == "__main__":
    main()