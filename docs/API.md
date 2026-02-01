# API Documentation

The IES_EV Backend API is built with FastAPI. Complete interactive documentation is available at `http://localhost:8000/docs`.

## Base URL

`http://localhost:8000/api`

## Endpoints

### Health Check

**GET** `/health`

Returns the connectivity status of the backend and its dependencies.

**Response:**

```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected"
}
```

### AI Chat

**POST** `/ai/chat`

Send a message to the DeepSeek AI agent.

**Request:**

```json
{
  "message": "How can I optimize EV charging?"
}
```

**Response:**

```json
{
  "response": "To optimize EV charging, you can...",
  "raw": { ... }
}
```
