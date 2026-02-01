# IES_EV System - Phase 0

Intelligent Energy System for Electric Vehicles (IES_EV) - Phase 0 Implementation.

![Status](https://img.shields.io/badge/Status-Phase_0_Complete-success)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![React](https://img.shields.io/badge/Frontend-React-blue)

## 🎯 Overview

This repository contains the Phase 0 implementation of the IES_EV project. It establishes the core infrastructure, including:

- **Backend**: FastAPI with PostgreSQL, Redis, and DeepSeek AI integration.
- **Frontend**: React (Vite) with a modern dark-mode dashboard.
- **Infrastructure**: Docker Compose orchestration for 7 services.
- **Monitoring**: Prometheus and Grafana setup.

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- DeepSeek API Key (Optional for basic run, required for AI features)

### One-Command Setup

```bash
# 1. Clone (if not already) and Enter directory
cd ies-ev-system

# 2. Setup Environment
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY

# 3. Launch System
make up
```

Wait about 30 seconds for services to initialize.

### 🌐 Access Points

| Service                | URL                        | Description                                                                                         |
| ---------------------- | -------------------------- | --------------------------------------------------------------------------------------------------- |
| **Frontend Dashboard** | http://localhost:3000      | Main User Interface                                                                                 |
| **Backend API Docs**   | http://localhost:8000/docs | Swagger UI                                                                                          |
| **Grafana**            | http://localhost:3001      | Monitoring Dashboards (Login: admin/admin)                                                          |
| **Prometheus**         | http://localhost:9090      | Metrics Browser                                                                                     |
| **Adminer**            | http://localhost:8080      | Database Management (System: PostgreSQL, Server: db, User: postgres, Pass: postgres, db: ies_ev_db) |

## 🧪 Verification

Run the verification script to ensure everything is correctly set up:

```bash
make verify
```

To run backend tests:

```bash
make test
```

## 📂 Project Structure

- `backend/` - Python FastAPI Application
- `frontend/` - React TypeScript Application
- `monitoring/` - Prometheus & Grafana Configuration
- `scripts/` - Utility scripts
- `docs/` - Detailed documentation

## 📄 Documentation

- [Setup Guide](docs/SETUP.md)
- [API Documentation](docs/API.md)
- [Contributing](CONTRIBUTING.md)

## 📜 License

MIT License
