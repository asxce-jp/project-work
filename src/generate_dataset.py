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
LEX_HK_POS = [
    "camera pulita e profumata", "letto comodo e ben fatto",
    "bagno impeccabile", "biancheria fresca cambiata ogni giorno",
    "stanza silenziosa e ben areata", "asciugamani morbidi e puliti",
    "pulizia accurata nei dettagli", "ambiente curato e ordinato",
    "personale delle pulizie discreto e professionale",
    "moquette pulita e senza odori", "frigobar rifornito puntualmente",
    "tutto funzionava perfettamente in camera",
    "stanza pronta prima del previsto", "piumone caldo e confortevole",
    "nessun problema durante tutto il soggiorno",
    "camera silenziosa nonostante il piano alto",
    "servizio di turndown serale molto gradito",
    "pulizia eseguita anche nei giorni festivi",
    "personale sorridente e disponibile",
    "camera rifornita di tutto il necessario"
]
LEX_HK_NEG = [
    "camera sporca all'arrivo", "letto scomodo e rumoroso",
    "lenzuola macchiate e non cambiate", "bagno con muffa e calcare",
    "odore sgradevole difficile da ignorare",
    "polvere visibile su mobili e superfici",
    "pavimento non lavato", "asciugamani umidi e consumati",
    "climatizzatore rotto e non riparato",
    "pulizia superficiale e frettolosa",
    "biancheria non sostituita per più giorni",
    "stanza rumorosa con pareti sottili",
    "frigobar non funzionante", "specchio del bagno rotto",
    "personale delle pulizie invadente e poco discreto",
    "stanza non pronta all'orario di check-in",
    "macchie sul copriletto", "zanzare in camera",
    "nessuno ha risposto alla richiesta di asciugamani extra",
    "camera che non veniva pulita se non su richiesta esplicita"
]

LEX_RC_POS = [
    "check-in rapido e senza attese", "personale accogliente e sorridente",
    "pagamento gestito senza intoppi", "accoglienza calorosa all'arrivo",
    "receptionist disponibile a qualsiasi ora",
    "informazioni utili sulla città fornite spontaneamente",
    "check-out rapido e senza sorprese in fattura",
    "staff multilingue e competente",
    "prenotazione taxi organizzata senza problemi",
    "consigli personalizzati su ristoranti e attrazioni",
    "upgrade di camera offerto gentilmente",
    "richiesta speciale gestita con professionalità",
    "orario di check-in anticipato accordato senza problemi",
    "portiere sempre presente e reattivo",
    "risposta rapida a ogni richiesta durante il soggiorno",
    "deposito bagagli gestito con cura",
    "rimborso processato velocemente",
    "staff che ricordava le preferenze del cliente",
    "area lounge della reception confortevole",
    "comunicazioni pre-arrivo chiare e precise"
]
LEX_RC_NEG = [
    "attesa lunghissima al check-in", "personale scortese e sbrigativo",
    "errori nella fattura difficili da correggere",
    "nessuna informazione fornita all'arrivo",
    "receptionist distratto e poco professionale",
    "prenotazione non trovata nel sistema",
    "camera assegnata diversa da quella prenotata",
    "nessuna risposta alle chiamate alla reception",
    "check-out caotico con lunghe code",
    "deposito bagagli smarrito",
    "staff che non parlava nessuna lingua straniera",
    "richiesta di late check-out ignorata",
    "addebiti non autorizzati sulla carta",
    "portiere mai presente quando serviva",
    "informazioni errate sugli orari dei servizi",
    "nessuno disponibile per assistenza notturna",
    "upgrade promesso e mai consegnato",
    "tono scostante e poco disponibile allo sportello",
    "attesa di oltre un'ora per avere le chiavi",
    "problemi con la prenotazione scaricati sul cliente"
]

LEX_FB_POS = [
    "colazione ricca e con prodotti freschi",
    "ristorante di ottimo livello", "cameriere attento e premuroso",
    "porzioni generose e ben presentate",
    "menu vario con opzioni per ogni esigenza",
    "cibo preparato al momento e servito caldo",
    "carta dei vini curata e ben assortita",
    "dessert artigianali di alta qualità",
    "servizio al tavolo rapido e professionale",
    "prodotti locali valorizzati nel menu",
    "colazione servita puntualmente e senza attese",
    "chef disponibile per richieste dietetiche speciali",
    "ambiente del ristorante elegante e tranquillo",
    "frutta fresca sempre disponibile a colazione",
    "caffè eccellente servito a qualsiasi ora",
    "personale di sala cortese e competente",
    "menù fisso conveniente e di qualità",
    "intolleranze alimentari gestite con attenzione",
    "cocktail bar ben fornito con ottimi bartender",
    "servizio in camera disponibile fino a tarda sera"
]
LEX_FB_NEG = [
    "colazione povera e sempre uguale",
    "cibo freddo servito come se fosse caldo",
    "menù troppo limitato e senza varietà",
    "porzioni insufficienti per il prezzo pagato",
    "cameriere assente per lunghi periodi",
    "qualità degli ingredienti deludente",
    "nessuna opzione vegetariana o vegana disponibile",
    "ristorante chiuso senza preavviso",
    "conto errato e difficile da far correggere",
    "attesa eccessiva prima di essere serviti",
    "rumore insopportabile in sala durante i pasti",
    "caffè annacquato e di scarsa qualità",
    "prodotti confezionati spacciati come freschi",
    "personale di sala frettoloso e poco curato",
    "orari del ristorante incompatibili con le esigenze dei clienti",
    "servizio in camera in ritardo di oltre un'ora",
    "intolleranze alimentari ignorate con rischio per la salute",
    "ambiente del ristorante rumoroso e caotico",
    "lista vini esaurita già a metà serata",
    "cibo chiaramente riscaldato e non preparato fresco"
]

TITLES_POS = [
    "Eccellente soggiorno!", "Servizio impeccabile", "Esperienza fantastica",
    "Consigliatissimo!", "Soggiorno da ricordare", "Tutto perfetto",
    "Tornerò sicuramente", "Hotel di qualità", "Personale straordinario",
    "Soddisfatto al 100%", "Superate le aspettative", "Esperienza top",
    "Hotel eccezionale", "Vale ogni centesimo", "Ci siamo trovati benissimo",
    "Struttura ottima", "Personale disponibile e gentile",
    "Soggiorno piacevolissimo", "Niente da dire, tutto perfetto",
    "Lo consiglio a tutti"
]
TITLES_NEG = [
    "Deludente esperienza", "Servizio pessimo", "Non tornerò mai più",
    "Molto insoddisfatto", "Esperienza negativa", "Soldi buttati",
    "Sotto ogni aspettativa", "Hotel da evitare", "Qualità inaccettabile",
    "Personale scortese e impreparato", "Esperienza da dimenticare",
    "Non lo consiglio a nessuno", "Struttura fatiscente",
    "Rapporto qualità-prezzo pessimo", "Rimpiango di aver prenotato",
    "Problemi irrisolti durante tutto il soggiorno",
    "Zero stelle se potessi", "Gestione disastrosa",
    "Aspettative completamente disattese", "Una delusione su tutta la linea"
]

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
