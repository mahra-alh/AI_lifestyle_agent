// src/firebase.js
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getAuth } from 'firebase/auth';

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "REMOVED_FIREBASE_KEY",
    authDomain: "ai-lifestyle-agent.firebaseapp.com",
    projectId: "ai-lifestyle-agent",
    storageBucket: "ai-lifestyle-agent.firebasestorage.app",
    messagingSenderId: "269365133725",
    appId: "1:269365133725:web:7b41f00d6268d417842f74",
    measurementId: "G-WR884VVBYH"
  };

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);

export { db, auth };