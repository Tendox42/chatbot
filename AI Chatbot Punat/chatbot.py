import re
from flask import Flask, render_template, request, jsonify
from rag import RagStore, answer_llm_grounded

app = Flask(__name__)

AI_NOTICE = {
    "title": "Eksperimentalni AI asistent (prototip)",
    "body": (
        "Razgovarate s AI sustavom. Odgovori se generiraju na temelju javno dostupnog sadržaja web stranice Općine Punat "
        "i mogu biti netočni ili nepotpuni. Ovaj alat ne predstavlja službene informacije Općine Punat. "
        "Molimo ne unosite osobne podatke (OIB, adresa stanovanja, brojevi dokumenata, IBAN itd.)."
    ),
    "data_sources": "Izvor podataka: indeksirani HTML sadržaj s https://www.punat.hr/ (PDF-ovi se preskaču).",
}

# Učitavanje FAISS indeksa i dokumenata
store = RagStore()


@app.get("/")
def home():
    return render_template("index.html", notice=AI_NOTICE)


@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return jsonify({"answer": "Napiši pitanje pa ću pokušati pronaći relevantne informacije.", "sources": []})

    # ✅ 0) Small-talk / pozdrav (bez RAG-a)
    msg = user_msg.strip().lower()
    if re.fullmatch(r"(bok+|pozdrav+|hej+|hello+|hi+|dobar\s+dan|dobro\s+jutro|dobra\s+vecer)", msg):
        return jsonify({
            "answer": "Bok! 😊 Reci što te zanima o Općini Punat (npr. radno vrijeme, kontakti, obrasci, natječaji).",
            "sources": []
        })

    # Minimalna zaštita od osobnih podataka (prototip)
    lowered = msg
    if any(x in lowered for x in ["oib", "iban", "broj osobne", "broj iskaznice", "adresa stanovanja"]):
        return jsonify({
            "answer": (
                "Molim ne upisuj osobne podatke. Postavi pitanje općenito (npr. 'gdje mogu pronaći obrazac za ...' "
                "ili 'koji su uvjeti za ...')."
            ),
            "sources": []
        })

    try:
        hits = store.search(user_msg, k=6)
        answer, sources = answer_llm_grounded(user_msg, hits)
        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        print("ERROR:", repr(e))
        return jsonify({
            "answer": f"Dogodila se greška: {type(e).__name__}: {e}",
            "sources": []
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
