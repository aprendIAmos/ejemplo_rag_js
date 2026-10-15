// index.js — script de indexado. Se corre a mano:
//   node index.js
// Lee docs/, parte los documentos en chunks, los embebe con OpenAI
// y los guarda en LanceDB (carpeta ./data).

const { indexDocs } = require("./rag");

indexDocs().catch((err) => {
  console.error("\nError al indexar:", err.message);
  process.exit(1);
});
