const admin = require("firebase-admin");

if (!admin.apps.length) {
  const serviceAccountPath = process.env.FIREBASE_SERVICE_ACCOUNT_PATH;

  if (serviceAccountPath) {
    // Local dev fallback: explicit service account file
    const serviceAccount = require(require("path").resolve(serviceAccountPath));
    admin.initializeApp({
      credential: admin.credential.cert(serviceAccount),
    });
    console.log("Firebase Admin: initialised with service account file.");
  } else {
    // Cloud Run / GCE: use Application Default Credentials automatically
    admin.initializeApp({
      credential: admin.credential.applicationDefault(),
    });
    console.log("Firebase Admin: initialised with Application Default Credentials.");
  }
}

const db = admin.firestore();

module.exports = db;