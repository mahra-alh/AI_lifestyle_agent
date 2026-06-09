// PURPOSE: Upload venue data (unified_venue_pool + faiss_corpus) to Firestore.
//
// Both files are uploaded in row-order, and every document receives an `index`
// field so the two collections can be joined back by position (row 0 in
// unified_venue_pool corresponds to index=0 in faiss_corpus, and vice-versa).
//
// Run:  node import_venues_to_firestore.js

const fs  = require("fs");
const csv = require("csv-parser");
const db  = require("./firebaseAdmin");
// Add this helper function near the top
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// ─── CONFIGURATION ────────────────────────────────────────────────────────────

const UPLOADS = [
  // {
  //   csvPath:        "../ai_agent/data/unified_venue_pool.csv",
  //   collectionName: "venues",
  // },
  {
    csvPath:        "../ai_agent/data/faiss_corpus.csv",
    collectionName: "faiss_corpus",
  },
];

const BATCH_SIZE = 500;

// ─── HELPERS ──────────────────────────────────────────────────────────────────

function cleanValue(value) {
  if (value === "" || value === "NaN" || value === "nan" || value === undefined) {
    return null;
  }
  const num = Number(value);
  if (!isNaN(num) && String(value).trim() !== "") {
    return num;
  }
  return value;
}

// Read a CSV file and return a Promise that resolves to an array of cleaned rows.
function readCSV(csvPath) {
  return new Promise((resolve, reject) => {
    const rows = [];
    console.log(`\nREADING: ${csvPath}`);

    fs.createReadStream(csvPath)
      .pipe(csv())
      .on("data", (row) => {
        const cleanedRow = {};
        for (const key in row) {
          cleanedRow[key] = cleanValue(row[key]);
        }
        rows.push(cleanedRow);
      })
      .on("end",  () => resolve(rows))
      .on("error", reject);
  });
}

// Upload rows to a Firestore collection in batches.
// Each document gets an `index` field equal to its 0-based row position so the
// two collections stay aligned even after independent queries.
async function uploadRows(rows, collectionName) {
  const totalBatches = Math.ceil(rows.length / BATCH_SIZE);
  console.log(`\nUPLOADING → ${collectionName}`);
  console.log(`Total rows : ${rows.length}`);
  console.log(`Total batches: ${totalBatches}`);

  let uploaded = 0;
  console.time(`upload:${collectionName}`);

  for (let i = 0; i < rows.length; i += BATCH_SIZE) {
    const batchNumber = Math.floor(i / BATCH_SIZE) + 1;
    const chunk = rows.slice(i, i + BATCH_SIZE);
    const batch = db.batch();

    chunk.forEach((row, offsetInChunk) => {
      const globalIndex = i + offsetInChunk;

      const docId = row.venue_id != null ? String(row.venue_id) : String(globalIndex);

      if (row.venue_id == null && collectionName === "venues") {
        console.warn(`  ⚠ Row ${globalIndex} missing venue_id — using index as doc ID`);
      }

      const ref = db.collection(collectionName).doc(docId);
      batch.set(ref, { ...row, index: globalIndex });
    });

    console.log(`  Batch ${batchNumber}/${totalBatches} — uploading ${chunk.length} docs...`);
    await batch.commit();

    uploaded += chunk.length;
    const pct = ((uploaded / rows.length) * 100).toFixed(1);
    console.log(`  Progress: ${uploaded}/${rows.length} (${pct}%)`);

    await sleep(500); // wait 500ms between batches to avoid quota limits
  }

  console.timeEnd(`upload:${collectionName}`);
  console.log(`✓ ${collectionName} upload complete\n`);
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────

async function main() {
  for (const { csvPath, collectionName } of UPLOADS) {
    const rows = await readCSV(csvPath);
    await uploadRows(rows, collectionName);
  }

  console.log("ALL COLLECTIONS SUCCESSFULLY UPLOADED");
  process.exit(0);
}

main().catch((err) => {
  console.error("Upload failed:", err);
  process.exit(1);
});