const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });

const admin = require("firebase-admin");

const serviceAccountPath = path.resolve(
  __dirname,
  "..",
  process.env.FIREBASE_SERVICE_ACCOUNT_PATH || "config/serviceAccountKey.json"
);
const serviceAccount = require(serviceAccountPath);

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const db = admin.firestore();

module.exports = db;