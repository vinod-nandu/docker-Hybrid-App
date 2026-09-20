# Hybrid RAG — FAISS + Neo4j AuraDB + Guardrails

PDF → Extraction (PyMuPDF) → Chunking → Vector Pipeline (OpenAI + FAISS) + Knowledge Graph
(OpenAI-extracted entities/relations → Neo4j AuraDB) → Hybrid Retrieval → Guardrails → LLM → Answer

**Stack**
- Frontend: Streamlit
- Backend: FastAPI
- Vector store: FAISS (local, on disk)
- Knowledge graph: Neo4j — either AuraDB (online) **or** a local self-hosted Neo4j
  Community container (included in `docker-compose.yml`, always on, no auto-pause).
  Which one is used is decided purely by `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD`
  in `.env` — the backend code doesn't change.
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
- A Neo4j database to connect to — either:
  - **Local (recommended for dev)**: run just the Neo4j container standalone, even
    though the rest of the app runs outside Docker:
    ```bash
    docker run -d --name hybrid_rag_neo4j \
      -p 7474:7474 -p 7687:7687 \
      -e NEO4J_AUTH=neo4j/changeme12345 \
      -v neo4j_data:/data \
      neo4j:5.24-community
    ```
    Then in `.env` use `NEO4J_URI=bolt://localhost:7687` and matching password.
    Always on, no pausing — good for iterating locally.
  - **AuraDB (online)**: https://console.neo4j.io → "New Instance" (Free tier) →
    download the generated credentials file (URI, username, password). Note: Aura
    Free auto-pauses after a few days idle; you'll need to "Resume" it in the
    console before reconnecting.

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
`docker-compose.yml`), and `docker-compose.yml` includes a **local Neo4j Community**
service by default — no AuraDB pausing to worry about.

```bash
cd hybrid-rag
cp .env.example .env     # fill in OPENAI_API_KEY; Neo4j vars already default to local

# Build and start all three services (neo4j, backend, frontend)
docker compose up --build -d

# Check logs
docker compose logs -f neo4j
docker compose logs -f backend
docker compose logs -f frontend

# Stop
docker compose down
```

- Backend: http://localhost:8000 (docs at `/docs`)
- Frontend: http://localhost:8501
- Neo4j Browser (optional, to inspect the graph): http://localhost:7474 — log in with
  username `neo4j` and the `NEO4J_PASSWORD` from your `.env`.
- FAISS index persists in the `faiss_storage` volume; graph data persists in `neo4j_data`
  — both survive container restarts. `docker compose down -v` wipes everything.

### Switching to AuraDB instead of local Neo4j
1. In `.env`, comment out the local `NEO4J_URI=bolt://neo4j:7687` line and uncomment/fill
   in the `neo4j+s://...` AuraDB block instead.
2. In `docker-compose.yml`, you can leave the `neo4j` service defined (it'll just sit
   unused) or delete it along with `depends_on: neo4j` under `backend` and the
   `neo4j_data`/`neo4j_logs` volumes — either way works, since only the env vars decide
   what the backend connects to.

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
