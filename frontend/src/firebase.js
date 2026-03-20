import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";

const firebaseConfig = {
  apiKey: "AIzaSyAptUpERqBekqVLRTAMsmWSOXHTk_FS8GA",
  authDomain: "cardioretina.firebaseapp.com",
  projectId: "cardioretina",
  storageBucket: "cardioretina.firebasestorage.app",
  messagingSenderId: "9660802982",
  appId: "1:9660802982:web:8010e66549ea82e708d167",
  measurementId: "G-RJCZG3Y495"
};

const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);