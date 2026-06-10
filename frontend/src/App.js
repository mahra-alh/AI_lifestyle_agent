import { useState } from "react";
import EmailEntry from "./components/EmailEntry";
import ProfileSetupPage from "./components/ProfileSetupPage";
import ChatWindow from "./components/ChatWindow";

const API_BASE = (process.env.REACT_APP_API_URL || "").replace(/\/$/, "");

function App() {
  // page: "email" | "setup" | "chat"
  const [page, setPage] = useState("email");
  const [userEmail, setUserEmail] = useState(null);

  async function handleEmailSubmit(email) {
    setUserEmail(email);

    try {
      const res = await fetch(`${API_BASE}/api/profile/${encodeURIComponent(email)}`);
      const data = await res.json();

      if (data.profile_complete) {
        setPage("chat");
      } else {
        setPage("setup");
      }
    } catch {
      // If profile check fails, still allow setup
      setPage("setup");
    }
  }

  function handleSetupComplete() {
    setPage("chat");
  }

  if (page === "email") {
    return <EmailEntry onEmailSubmit={handleEmailSubmit} />;
  }

  if (page === "setup") {
    return <ProfileSetupPage userEmail={userEmail} onComplete={handleSetupComplete} />;
  }

  return <ChatWindow userEmail={userEmail} />;
}

export default App;
