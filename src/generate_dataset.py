import random, uuid, csv
from pathlib import Path
random.seed(42)

DEPARTMENTS = ["Housekeeping", "Reception", "F&B"]
SENTIMENTS  = ["positive", "negative"]

LEX_HK_POS  = ["camera pulita e profumata", "staff cordiale e disponibile", "letto comodo", "colazione abbondante"]
LEX_HK_NEG  = ["camera sporca", "staff scortese", "letto scomodo", "lenzuola macchiate"]

LEX_RC_POS  = ["check-in veloce", "personale accogliente", "pagamento senza problemi", "accoglienza calorosa"]
LEX_RC_NEG  = ["lunghe attese al check-in", "personale scortese", "errori nel pagamento", "servizio lento"]

LEX_FB_POS  = ["colazione ricca e varia", "ristorante eccellente", "cameriere attento", "porzioni abbondanti"]
LEX_FB_NEG  = ["colazione scarsa", "servizio lento", "menu limitato", "porzioni scarse"]

TITLES_POS  = ["Eccellente soggiorno!", "Servizio impeccabile", "Esperienza fantastica", "Consigliatissimo!"]
TITLES_NEG  = ["Deludente esperienza", "Servizio pessimo", "Non tornerò mai più", "Molto insoddisfatto"]

def synthesize_review(department, sentiment):
    if department == "Housekeeping":
        phrase = random.choice(LEX_HK_POS if sentiment == "positive" else LEX_HK_NEG)
    elif department == "Reception":
        phrase = random.choice(LEX_RC_POS if sentiment == "positive" else LEX_RC_NEG)
    else:
        phrase = random.choice(LEX_FB_POS if sentiment == "positive" else LEX_FB_NEG)

    title = random.choice(TITLES_POS if sentiment == "positive" else TITLES_NEG)
    return title, phrase

def main(n=360, output_path="data/synthetic_reviews.csv"):
    Path("data").mkdir(exist_ok=True)
    rows = []
    
    # 3 reparti x 2 sentimenti x 60 recensioni ciascuno = 360 recensioni totali
    per_bucket = n // (len(DEPARTMENTS) * len(SENTIMENTS))
    for department in DEPARTMENTS:
        for sentiment in SENTIMENTS:
            for _ in range(per_bucket):
                id = str(uuid.uuid4().hex[:12])
                title, body = synthesize_review(department, sentiment)
                rows.append([id, title, body, department, sentiment])
                
    random.shuffle(rows)
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "body", "department", "sentiment"])
        writer.writerows(rows)
    print(f"Synthetic dataset with {n} reviews saved to {output_path}")

if __name__ == "__main__":
    main()