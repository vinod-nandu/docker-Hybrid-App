# Hybrid RAG — FAISS + Neo4j AuraDB + Guardrails

PDF → Extraction (PyMuPDF) → Chunking → Vector Pipeline (OpenAI + FAISS) + Knowledge Graph
(OpenAI-extracted entities/relations → Neo4j AuraDB) → Hybrid Retrieval → Guardrails → LLM → Answer

**Stack**
- Frontend: Streamlit
- Backend: FastAPI
- Vector store: FAISS (local, on disk)
- Knowledge graph: Neo4j AuraDB (online)
- LLM/embeddings: OpenAI
- PDF extraction: PyMuPDF
- Guardrails: deterministic, in `backend/guardrails/` (no extra LLM calls)
  - `input_guard.py`: blocks empty/too-long queries, prompt-injection phrases, disallowed
    topics; redacts PII (emails, SSNs, card numbers) before retrieval.
  - `output_guard.py`: lexical grounding check (answer must overlap enough with retrieved
    context) + disallowed-content filter, so the model can't answer from outside the document.

```
hybrid-rag/
├── backend/
│   ├── main.py            FastAPI app (/upload, /query, /health)
│   ├── config.py          env var loading
│   ├── extraction.py      PyMuPDF PDF -> text
│   ├── chunking.py        page-aware chunking with overlap
│   ├── embeddings.py      OpenAI embeddings + entity/relation extraction
│   ├── vector_store.py    FAISS index + metadata sidecar
│   ├── knowledge_graph.py Neo4j AuraDB driver (documents/chunks/entities/relations)
│   ├── retrieval.py       hybrid retrieval (vector + graph, merged & deduped)
│   ├── llm.py             grounded answer generation
│   ├── guardrails/
│   │   ├── input_guard.py
│   │   └── output_guard.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py              Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore
```

---

## 1. Run locally (no Docker)

### Prerequisites
- Python 3.11+
- An OpenAI API key
- A free Neo4j AuraDB instance: https://console.neo4j.io → "New Instance" (Free tier) →
  download the generated credentials file (URI, username, password)

### Steps

```bash
# 1. Clone / unzip the project, then cd into it
cd hybrid-rag

# 2. Create a virtualenv
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install both backend and frontend dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# 4. Configure environment variables
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# 5. Run the backend (from the project root, so `backend.*` imports resolve)
uvicorn backend.main:app --reload --port 8000

# 6. In a second terminal (same venv), run the frontend
export BACKEND_URL=http://localhost:8000   # Windows (PowerShell): $env:BACKEND_URL="http://localhost:8000"
streamlit run frontend/app.py
```

Open http://localhost:8501, upload a PDF, wait for indexing, then ask questions.
API docs (Swagger) are at http://localhost:8000/docs.

---

## 2. Convert to Docker

Everything is already Dockerized (`backend/Dockerfile`, `frontend/Dockerfile`,
`docker-compose.yml`). Neo4j AuraDB is a managed cloud service, so it does **not** run in a
container — only your `.env` needs to point to it.

```bash
cd hybrid-rag
cp .env.example .env     # fill in your real keys/credentials

# Build and start both services
docker compose up --build -d

# Check logs
docker compose logs -f backend
docker compose logs -f frontend

# Stop
docker compose down
```

- Backend: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:8501
- FAISS index persists in the `faiss_storage` named volume, so it survives container restarts.

---

## 3. Upload to GitHub

```bash
cd hybrid-rag
git init
git add .
git commit -m "Initial commit: Hybrid RAG (FAISS + Neo4j AuraDB) with guardrails"

# Create an empty repo on GitHub first (via github.com or `gh repo create`), then:
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

`.env` is in `.gitignore` — **never commit real API keys or Neo4j credentials**. Only
`.env.example` (with placeholders) should be pushed.

---

## 4. Pull from GitHub to a VPS (e.g. Hostinger VPS) and publish

These steps assume a Hostinger VPS running Ubuntu, accessed via SSH.

### a) Install Docker on the VPS (one-time)

```bash
ssh root@your-vps-ip

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
sudo apt-get install -y docker-compose-plugin
```

### b) Pull the repo and configure secrets

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

cp .env.example .env
nano .env    # fill in OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
```

### c) Run it

```bash
docker compose up --build -d
```

Check it's up: `docker compose ps` and `curl http://localhost:8000/health`.

### d) Expose it publicly (choose one)

**Option 1 — quick/simple:** open the ports in your VPS firewall / Hostinger panel:
- `8501` (Streamlit) and `8000` (FastAPI), then visit `http://your-vps-ip:8501`.

**Option 2 (recommended) — Nginx reverse proxy + domain + HTTPS:**

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/hybrid-rag`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/hybrid-rag /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d your-domain.com     # free HTTPS via Let's Encrypt
```

Point your domain's DNS `A` record (in Hostinger's DNS panel) to the VPS's IP address first,
then run certbot.

### e) Redeploying after future code changes

```bash
cd <your-repo>
git pull origin main
docker compose up --build -d
```

### Optional: keep it running with a CI hook
Add a small deploy script on the VPS (`deploy.sh`) that does `git pull && docker compose up
--build -d`, and trigger it via a GitHub Actions SSH step or a webhook if you want push-to-deploy.

---

## Notes & tips
- **Costs**: every PDF upload calls OpenAI embeddings once per chunk, plus one chat completion
  per chunk for entity/relation extraction (used to populate Neo4j). For very large PDFs this
  can add up — consider batching or raising `CHUNK_SIZE` to reduce the number of chunks.
- **Neo4j Aura free tier** sleeps after inactivity; the first query after idle time may be slow.
- **Guardrails are deterministic** (regex + lexical overlap), so they run with zero extra LLM
  calls and fail closed (block) rather than fail open.
- To reset the vector index, stop the containers and delete the `faiss_storage` volume
  (`docker volume rm hybrid-rag_faiss_storage`) — this does not touch Neo4j data, which you'd
  clear separately in the Aura console or via Cypher (`MATCH (n) DETACH DELETE n`).
