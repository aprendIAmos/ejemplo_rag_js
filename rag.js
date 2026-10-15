// rag.js — módulo compartido: indexado y recuperación (retrieve)

const path = require("path");
const fs = require("fs/promises");
const OpenAI = require("openai");
const lancedb = require("@lancedb/lancedb");

require("dotenv").config();

// ---------- Config ----------
const DATA_DIR = path.join(__dirname, "data");
const DOCS_DIR = path.join(__dirname, "docs");
const TABLE = "docs";

const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const EMBED_DIMS = EMBED_MODEL === "text-embedding-3-large" ? 3072 : 1536;

const TOP_K = 5;
const CHUNK_SIZE = 1200;
const CHUNK_OVERLAP = 200;

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------- Chunking ----------
// Parte el texto en trozos de ~CHUNK_SIZE caracteres con overlap,
// cortando en el límite de párrafo más cercano.
function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const clean = text.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  if (!clean) return [];

  const chunks = [];
  let start = 0;

  while (start < clean.length) {
    let end = Math.min(start + size, clean.length);

    if (end < clean.length) {
      // busco el último corte "natural" dentro de la ventana
      const window = clean.slice(start, end);
      const cut = Math.max(
        window.lastIndexOf("\n\n"),
        window.lastIndexOf(". "),
        window.lastIndexOf("\n")
      );
      if (cut > size * 0.5) end = start + cut + 1;
    }

    const piece = clean.slice(start, end).trim();
    if (piece) chunks.push(piece);

    if (end >= clean.length) break;
    start = Math.max(end - overlap, start + 1);
  }

  return chunks;
}

// ---------- Embeddings ----------
async function embed(texts) {
  const res = await openai.embeddings.create({ model: EMBED_MODEL, input: texts });
  return res.data.map((d) => d.embedding);
}

// ---------- Lectura de documentos ----------
async function readDocument(filePath) {
  const ext = path.extname(filePath).toLowerCase();

  if (ext === ".pdf") {
    const pdfParse = require("pdf-parse");
    const buffer = await fs.readFile(filePath);
    const data = await pdfParse(buffer);
    return data.text || "";
  }

  if (ext === ".txt" || ext === ".md") {
    return await fs.readFile(filePath, "utf8");
  }

  return null; // formato no soportado
}

// ---------- Indexado ----------
async function indexDocs() {
  let files = [];
  try {
    files = (await fs.readdir(DOCS_DIR)).filter((f) =>
      [".pdf", ".txt", ".md"].includes(path.extname(f).toLowerCase())
    );
  } catch {
    throw new Error(`No existe la carpeta ${DOCS_DIR}. Creala y poné tus documentos ahí.`);
  }

  if (files.length === 0) {
    throw new Error(`No hay archivos .pdf/.txt/.md en ${DOCS_DIR}`);
  }

  const records = [];

  for (const file of files) {
    const full = path.join(DOCS_DIR, file);
    const text = await readDocument(full);
    if (!text || !text.trim()) {
      console.log(`  ! "${file}" no dio texto, se saltea.`);
      continue;
    }

    const chunks = chunkText(text);
    for (const c of chunks) {
      records.push({ source: file, text: c, chars: c.length });
    }
    console.log(`  ✓ ${file}: ${chunks.length} chunks`);
  }

  if (records.length === 0) throw new Error("Ningún documento produjo texto indexable.");

  console.log(`Embebiendo ${records.length} chunks con ${EMBED_MODEL}...`);
  for (let i = 0; i < records.length; i += 50) {
    const batch = records.slice(i, i + 50);
    const vectors = await embed(batch.map((r) => r.text));
    batch.forEach((r, j) => (r.vector = vectors[j]));
    console.log(`  embebidos ${Math.min(i + 50, records.length)}/${records.length}`);
  }

  // guardo un hash simple por chunk para poder referenciarlo
  records.forEach((r, i) => (r.id = `${r.source}#${i}`));

  // Reconstruyo la tabla: así re-indexar no duplica nada.
  const db = await lancedb.connect(DATA_DIR);
  const existing = await db.tableNames();
  if (existing.includes(TABLE)) await db.dropTable(TABLE);
  await db.createTable(TABLE, records);

  console.log(`\nListo. Tabla "${TABLE}" creada en ${DATA_DIR} con ${records.length} chunks.`);
}

// ---------- Recuperación (retrieve) ----------
async function retrieve(question, topK = TOP_K) {
  const db = await lancedb.connect(DATA_DIR);
  const names = await db.tableNames();
  if (!names.includes(TABLE)) return [];

  const table = await db.openTable(TABLE);
  const [queryVector] = await embed([question]);

  const rows = await table.search(queryVector).limit(topK).toArray();
  return rows.map((r) => ({ text: r.text, source: r.source, score: r._distance }));
}

// ---------- Generación ----------
const SYSTEM_PROMPT = `Sos un asistente útil. Respondé SIEMPRE en español.

Reglas:
- Respondé únicamente usando la información del contexto que se te brinda abajo.
- Si el contexto no contiene la respuesta, decí claramente que no encontraste esa información en los documentos. No inventes.
- Si la respuesta aparece en varios fragmentos, sintetizalas.
- Sé conciso y concreto.`;

async function answer(question, history = []) {
  const contexts = await retrieve(question);

  if (contexts.length === 0) {
    return {
      answer:
        "Todavía no hay documentos indexados. Poné tus archivos en la carpeta docs/ y ejecutá `node index.js`.",
      sources: [],
    };
  }

  const contextText = contexts
    .map((c, i) => `[Fragmento ${i + 1} — ${c.source}]\n${c.text}`)
    .join("\n\n---\n\n");

  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    ...(Array.isArray(history) ? history.slice(-8) : []),
    {
      role: "user",
      content: `Contexto de los documentos:\n\n${contextText}\n\n---\n\nPregunta: ${question}`,
    },
  ];

  const res = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    messages,
    temperature: 0.2,
  });

  return {
    answer: res.choices[0].message.content,
    sources: contexts.map((c) => c.source),
  };
}

module.exports = { indexDocs, retrieve, answer, chunkText, EMBED_DIMS, OPENAI_MODEL };
