const admin = require("firebase-admin");
const path = require("path");
const fs = require("fs");

const serviceAccountPath =
  process.env.FIREBASE_SERVICE_ACCOUNT_PATH ||
  path.join(__dirname, "config", "serviceAccountKey.json");
const serviceAccountJson = process.env.FIREBASE_SERVICE_ACCOUNT_JSON;

const hasServiceAccountFile = fs.existsSync(serviceAccountPath);

if (serviceAccountJson) {
  let serviceAccount;

  try {
    serviceAccount = JSON.parse(serviceAccountJson);
  } catch (error) {
    throw new Error("FIREBASE_SERVICE_ACCOUNT_JSON must contain valid JSON.");
  }

  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
  });
} else if (hasServiceAccountFile) {
  const serviceAccount = require(serviceAccountPath);

  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
  });
} else {
  admin.initializeApp({
    credential: admin.credential.applicationDefault(),
  });
}

const db = admin.firestore();

module.exports = db;
