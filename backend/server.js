const express = require("express");
const cors = require("cors");
const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });

const app = express();
const PORT = Number(process.env.PORT || 5000);
const PYTHON_AGENT_URL = (process.env.PYTHON_AGENT_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const CORS_ORIGINS = (process.env.CORS_ORIGINS || "").split(",").map((origin) => origin.trim()).filter(Boolean);

app.disable("x-powered-by");
app.use(express.json({ limit: "1mb" }));

if (CORS_ORIGINS.length > 0) {
  app.use(cors({ origin: CORS_ORIGINS }));
} else {
  app.use(cors());
}

function normalizeStatusPayload(ok, extra = {}) {
  return {
    ok,
    service: "backend",
    port: PORT,
    python_agent_url: PYTHON_AGENT_URL,
    ...extra,
  };
}

async function fetchJson(url, options = {}, timeoutMs = 5000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "content-type": "application/json",
        ...(options.headers || {}),
      },
    });

    const text = await response.text();
    let body = text;

    try {
      body = text ? JSON.parse(text) : {};
    } catch (_) {
      body = text;
    }

    return {
      ok: response.ok,
      status: response.status,
      body,
    };
  } finally {
    clearTimeout(timeoutId);
  }
}

async function probePythonService(path) {
  try {
    return await fetchJson(`${PYTHON_AGENT_URL}${path}`, {}, 4000);
  } catch (error) {
    return {
      ok: false,
      status: 503,
      body: {
        error: error.name === "AbortError" ? "Python agent request timed out" : error.message,
      },
    };
  }
}

app.get("/", async (req, res) => {
  const pythonHealth = await probePythonService("/health");
  res.json(
    normalizeStatusPayload(true, {
      status: "running",
      routes: ["/health", "/ready", "/api/test", "/api/chat"],
      python_agent: pythonHealth.body,
    })
  );
});

app.get("/health", async (req, res) => {
  const pythonHealth = await probePythonService("/health");

  res.status(200).json(
    normalizeStatusPayload(true, {
      status: "healthy",
      python_agent: pythonHealth.body,
      dependencies: {
        python_agent_reachable: pythonHealth.ok,
      },
    })
  );
});

app.get("/ready", async (req, res) => {
  const pythonReady = await probePythonService("/ready");
  const ready = pythonReady.ok;

  res.status(ready ? 200 : 503).json(
    normalizeStatusPayload(ready, {
      status: ready ? "ready" : "not_ready",
      python_agent: pythonReady.body,
      dependencies: {
        python_agent_ready: pythonReady.ok,
      },
    })
  );
});

app.get("/api/test", async (req, res) => {
  const pythonHealth = await probePythonService("/health");

  res.json(
    normalizeStatusPayload(true, {
      message: "Backend is reachable.",
      python_agent_reachable: pythonHealth.ok,
    })
  );
});

app.get("/api/profile/:user_id", async (req, res) => {
  const userId = String(req.params.user_id || "").trim().toLowerCase();

  if (!userId) {
    return res.status(400).json({
      ok: false,
      error: "user_id cannot be empty",
    });
  }

  const pythonResponse = await fetchJson(
    `${PYTHON_AGENT_URL}/api/profile/${encodeURIComponent(userId)}`,
    {},
    30000
  );

  res.status(pythonResponse.status).json(pythonResponse.body);
});

app.post("/api/profile", async (req, res) => {
  const pythonResponse = await fetchJson(`${PYTHON_AGENT_URL}/api/profile`, {
    method: "POST",
    body: JSON.stringify(req.body || {}),
  }, 30000);
  res.status(pythonResponse.status).json(pythonResponse.body);
});

app.post("/api/feedback", async (req, res) => {
  const pythonResponse = await fetchJson(`${PYTHON_AGENT_URL}/api/feedback`, {
    method: "POST",
    body: JSON.stringify(req.body || {}),
  }, 30000);

  res.status(pythonResponse.status).json(pythonResponse.body);
});

app.post("/api/bookings/:pending_id/confirm", async (req, res) => {
  const pendingId = encodeURIComponent(String(req.params.pending_id || "").trim());
  const pythonResponse = await fetchJson(
    `${PYTHON_AGENT_URL}/api/bookings/${pendingId}/confirm`,
    { method: "POST" },
    30000
  );
  res.status(pythonResponse.status).json(pythonResponse.body);
});

app.post("/api/bookings/:pending_id/cancel", async (req, res) => {
  const pendingId = encodeURIComponent(String(req.params.pending_id || "").trim());
  const pythonResponse = await fetchJson(
    `${PYTHON_AGENT_URL}/api/bookings/${pendingId}/cancel`,
    { method: "POST" },
    30000
  );
  res.status(pythonResponse.status).json(pythonResponse.body);
});

app.get("/api/venue/:venue_name", async (req, res) => {
  const venueName = encodeURIComponent(String(req.params.venue_name || "").trim());
  const pythonResponse = await fetchJson(
    `${PYTHON_AGENT_URL}/api/venue/${venueName}`,
    {},
    15000
  );
  res.status(pythonResponse.status).json(pythonResponse.body);
});

app.post("/api/chat", async (req, res) => {
  const { user_id, user_message } = req.body || {};

  if (!user_id || !user_message) {
    return res.status(400).json({
      ok: false,
      error: "Missing required fields: user_id and user_message",
    });
  }

  const pythonResponse = await fetchJson(`${PYTHON_AGENT_URL}/api/chat`, {
    method: "POST",
    body: JSON.stringify({ user_id, user_message }),
  }, 120000);

  res.status(pythonResponse.status).json(pythonResponse.body);
});

app.use((req, res) => {
  res.status(404).json({
    ok: false,
    error: "Route not found",
    path: req.originalUrl,
  });
});

app.listen(PORT, () => {
  console.log(`Backend gateway running on http://localhost:${PORT}`);
  console.log(`Python agent target: ${PYTHON_AGENT_URL}`);
});
