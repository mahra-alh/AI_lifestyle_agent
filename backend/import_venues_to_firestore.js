// PURPOSE: Upload venue data from CSV into Firebase

// Built-in Node.js module for reading files
const fs = require("fs");

// CSV parser package, reads CSV row-by-row
const csv = require("csv-parser");
// Firebase Firestore connection
const db = require("./firebaseAdmin");

// CONFIGURATION
// Path to CSV file
const csvPath = "./data/unified_venue_pool.csv";

// Firestore collection name (think of this like a SQL table)
const collectionName = "venues";

// Firestore maximum batch size (upload in chunks)
const BATCH_SIZE = 500;

// DATA CLEANING FUNCTION (safety net to clean CSV values before upload)
//

function cleanValue(value) {
  // Convert invalid values into null (Firestore handles null cleanly)
  if (
    value === "" ||
    value === "NaN" ||
    value === "nan" ||
    value === undefined
  ) {
    return null;
  }

  // Try convert into a number ("25.4" → 25.4)
  const num = Number(value);

  // If value is a valid number, store it as numeric type
  if (
    !isNaN(num) &&
    String(value).trim() !== ""
  ) {
    return num;
  }
  // Otherwise keep original string/text
  return value;
}

// CSV UPLOAD FUNCTION
async function uploadCSV() {
  // Temporary array storing all rows
  const rows = [];
  // READ CSV FILE
  console.log("READING CSV FILE...\n");
  fs.createReadStream(csvPath)
    // Send CSV stream into parser
    .pipe(csv())

    // Runs ONCE for every row in CSV
    .on("data", (row) => {
      // Object holding cleaned row values
      const cleanedRow = {};

      // Loop through every column
      for (const key in row) {

        // Clean each value
        cleanedRow[key] = cleanValue(
          row[key]
        );

      }

      // Store cleaned row
      rows.push(cleanedRow);

    })



    // CSV FINISHED LOADING
    .on("end", async () => {

      console.log("CSV FINISHED LOADING");
      console.log(
        `Total rows found: ${rows.length}`
      );

      // Calculate number of upload batches
      const totalBatches = Math.ceil(
        rows.length / BATCH_SIZE
      );

      console.log(
        `Total batches required: ${totalBatches}`
      );

      console.log("STARTING FIRESTORE UPLOAD\n");

      // Upload progress counter
      let uploaded = 0;

      // Start upload timer
      console.time("TOTAL UPLOAD TIME");

      // LOOP THROUGH BATCHES
      for (
        let i = 0;
        i < rows.length;
        i += BATCH_SIZE
      ) {

        // Current batch number
        const batchNumber =
          Math.floor(i / BATCH_SIZE) + 1;



        console.log(
          `Starting Batch ${batchNumber}/${totalBatches}`
        );

        // Create Firestore batch object
        const batch = db.batch();

        // Current chunk of rows
        const chunk = rows.slice(
          i,
          i + BATCH_SIZE
        );

        console.log(
          `Chunk size: ${chunk.length}`
        );

        // PREPARE DOCUMENTS INSIDE BATCH
        chunk.forEach((row) => {

          // Unique venue identifier
          const venueId = row.venue_id;

          // Skip invalid rows
          if (!venueId) {

            console.log(
              `Skipped row because venue_id missing`
            );

            return;
          }

          const ref = db
            .collection(collectionName)
            .doc(String(venueId));



          // Add document to batch
          // Nothing uploads yet. We are only PREPARING operations for batch upload
          batch.set(ref, row);

        });

        // EXECUTE BATCH UPLOAD
        console.log(
          `Uploading Batch ${batchNumber}...`
        );

        await batch.commit();
        uploaded += chunk.length;
        // PROGRESS MONITORING
        console.log(
          `Finished Batch ${batchNumber}`
        );

        console.log(
          `Progress: ${uploaded}/${rows.length}`
        );

        // Upload completion percentage
        const percent = (
          (uploaded / rows.length) * 100
        ).toFixed(2);

        console.log(
          `Completion: ${percent}%`
        );



        console.log(
          "---------------------------------\n"
        );

      }

      // FINISHED
      console.timeEnd("TOTAL UPLOAD TIME");



      console.log("ALL VENUES SUCCESSFULLY UPLOADED\n");

      // Safely terminate Node process
      process.exit();

    });

}

// START PROGRAM
uploadCSV();