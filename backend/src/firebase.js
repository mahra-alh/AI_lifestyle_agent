// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
  authDomain: "ai-lifestyle-agent.firebaseapp.com",
  projectId: "ai-lifestyle-agent",
  storageBucket: "ai-lifestyle-agent.firebasestorage.app",
  messagingSenderId: "269365133725",
  appId: "1:269365133725:web:6055f341fc57b44d842f74",
  measurementId: "G-JJE7PQ2B2G"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);