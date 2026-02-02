AI Chatbot Općina Punat – upute za pokretanje projekta

--------------------------------------------------
1. Preduvjeti
--------------------------------------------------
- Python verzija: 3.11.x (provjereno radi s 3.11.9)
Provjera Python verzije:
python –version

--------------------------------------------------
3. Virtualno okruženje
--------------------------------------------------
U folderu projekta:
python -m venv .venv

Aktivacija (PowerShell):
.venv\Scripts\activate

Ako PowerShell blokira skripte treba upalit powershell od windowsa i zaljepit sljedeću komndu:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned





--------------------------------------------------
4. Instalacija paketa
--------------------------------------------------
U aktiviranom virtualnom okruženju:

python -m pip install --upgrade pip
pip install flask requests beautifulsoup4 lxml tqdm
pip install sentence-transformers
pip install faiss-cpu
pip install openai
NA NETU INSTALIRAT OLLAMA, nakon toga:
Omllama pull mistral/llama3 (ako je jači komp) – nekad treba cijeli path od tamo di je instalirano jer ne more ga nać
Upiši: Pozdrav!
Izađi: /bye












--------------------------------------------------
5. Izrada vektorske baze podataka
--------------------------------------------------
Ovaj korak se izvodi samo prvi put ili pri osvježavanju podataka:

python build_index.py

U folderu data/ moraju se pojaviti:
- docs.jsonl
- index.faiss
- meta.json

--------------------------------------------------
6. Pokretanje aplikacije
--------------------------------------------------
python chatbot.py

U web pregledniku otvoriti:
http://localhost:5000


