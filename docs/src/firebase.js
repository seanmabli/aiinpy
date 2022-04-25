import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getFirestore } from "@firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyAeVOKLeGPek386fDsyZ7lC9zQ_9JlnaIc",
  authDomain: "aiinpy.firebaseapp.com",
  projectId: "aiinpy",
  storageBucket: "aiinpy.appspot.com",
  messagingSenderId: "612510022256",
  appId: "1:612510022256:web:65bfc5368b9ab91ef392b5",
  measurementId: "G-W0TZJ4R740"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
export const db = getFirestore(app);