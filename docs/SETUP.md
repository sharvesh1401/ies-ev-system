# Setup Guide

## Requirements

- Docker Desktop (or Engine + Compose)
- Git
- Make (Optional, commands also provided)

## Installation Steps

1. **Clone the Repository**

   ```bash
   git clone <repo-url>
   cd ies-ev-system
   ```

2. **Environment Configuration**
   Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

   Open `.env` and set your `DEEPSEEK_API_KEY`. If you don't have one, the AI features will return errors, but the rest of the system will work.

3. **Build and Run**
   Using Make:

   ```bash
   make up
   ```

   Or using Docker Compose directly:

   ```bash
   docker-compose up -d --build
   ```

4. **Verify Deployment**
   Check if services are running:
   ```bash
   docker-compose ps
   ```
   You should see 7 services: `backend`, `frontend`, `db`, `redis`, `prometheus`, `grafana`, `adminer`.

## Troubleshooting

- **Ports already in use**: If port 8000 or 3000 is taken, modify `.env` or `docker-compose.yml`.
- **Database connection failed**: Ensure the `db` container is healthy (`docker-compose ps`).
- **AI Error**: Check your API key in `.env` and ensure you have internet access.

## Development

- **Backend**: Python/FastAPI code in `backend/`
- **Frontend**: React/Vite code in `frontend/`

Changes in `backend` require a restart (or use volume mounts for hot reload, currently configured for production-like build in Dockerfile, but local development usually runs via `uvicorn` directly).

To run Backend locally (without Docker):

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```
