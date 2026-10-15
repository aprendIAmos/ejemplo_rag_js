// app.js — servidor Express: recibe preguntas, responde con RAG

const path = require("path");
const express = require("express");
const { answer } = require("./rag");

require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, "public")));
app.use(express.json({ limit: "1mb" }));

// Health check
app.get("/api/health", (_req, res) => res.json({ ok: true }));

// Endpoint principal de chat
app.post("/api/chat", async (req, res) => {
  const { message, history } = req.body || {};

  if (!message || typeof message !== "string" || !message.trim()) {
    return res.status(400).json({ error: "Falta el campo 'message'." });
  }

  try {
    const result = await answer(message.trim(), Array.isArray(history) ? history : []);
    res.json(result);
  } catch (err) {
    console.error("Error en /api/chat:", err);
    res.status(500).json({ error: "Ocurrió un error al procesar la pregunta." });
  }
});

app.listen(PORT, () => {
  console.log(`Chat RAG corriendo en http://localhost:${PORT}`);
});
