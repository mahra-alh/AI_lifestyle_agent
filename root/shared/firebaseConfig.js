const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY || "REMOVED_FIREBASE_KEY",
  authDomain:
    process.env.REACT_APP_FIREBASE_AUTH_DOMAIN || "ai-lifestyle-agent.firebaseapp.com",
  projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID || "ai-lifestyle-agent",
  storageBucket:
    process.env.REACT_APP_FIREBASE_STORAGE_BUCKET || "ai-lifestyle-agent.firebasestorage.app",
  messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID || "269365133725",
  appId: process.env.REACT_APP_FIREBASE_APP_ID || "1:269365133725:web:7b41f00d6268d417842f74",
  measurementId: process.env.REACT_APP_FIREBASE_MEASUREMENT_ID || "G-WR884VVBYH",
};

export default firebaseConfig;
