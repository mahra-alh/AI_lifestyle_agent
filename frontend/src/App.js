import { useState } from "react";
import EmailEntry from "./components/EmailEntry";
import ChatWindow from "./components/ChatWindow";

const API_BASE = (process.env.REACT_APP_API_URL || "").replace(/\/$/, "");

function App() {
  const [userEmail, setUserEmail] = useState(null);
  const [profile, setProfile] = useState(null);

  async function handleEmailSubmit(email) {
    setUserEmail(email);

    const res = await fetch(
      `${API_BASE}/api/profile/${encodeURIComponent(email)}`
    );

    const data = await res.json();
    setProfile(data);
  }

  if (!userEmail) {
    return <EmailEntry onEmailSubmit={handleEmailSubmit} />;
  }

  return <ChatWindow userEmail={userEmail} />;
}

export default App;
