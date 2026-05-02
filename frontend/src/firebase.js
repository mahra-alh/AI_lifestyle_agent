// src/firebase.js
import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';
import { getAuth } from 'firebase/auth';
import firebaseConfig from '../../root/shared/firebaseConfig';

const app = initializeApp({
  ...firebaseConfig,
  apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
});
const db = getFirestore(app);
const auth = getAuth(app);

export { db, auth };