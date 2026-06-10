import { useState } from "react";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";

const API_BASE = (process.env.REACT_APP_API_URL || "").replace(/\/$/, "");

function ChatWindow({ userEmail }) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  async function sendMessage(text) {
    setMessages((prev) => [...prev, { role: "user", text }]);
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userEmail, user_message: text }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Request failed");
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          text: data.response,
          pendingBooking: data.pending_booking || null,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "agent",
          text: "Something went wrong. Please check your connection and try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <div style={styles.avatar}>
            <div>
              <div style={styles.agentName}>Dubai Lifestyle Agent</div>
              <div style={styles.onlineRow}>
                <span style={styles.onlineDot} />
                <span style={styles.onlineText}>Online</span>
              </div>
            </div>
          </div>
          <span style={styles.headerEmail}>{userEmail}</span>
        </div>
      </div>
      <MessageList messages={messages} loading={loading} userEmail={userEmail} />
      <MessageInput onSend={sendMessage} disabled={loading} />
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    backgroundColor: "#0f1117",
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "14px 20px",
    backgroundColor: "#151821",
    borderBottom: "1px solid #1e2130",
    flexShrink: 0,
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  avatar: {
    width: "40px",
    height: "40px",
    borderRadius: "50%",
    backgroundColor: "#1e2130",
    border: "1px solid #2a2d3e",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "18px",
    flexShrink: 0,
  },
  agentName: {
    fontSize: "15px",
    fontWeight: "700",
    color: "#e2e8f0",
    lineHeight: 1.2,
  },
  onlineRow: {
    display: "flex",
    alignItems: "center",
    gap: "5px",
    marginTop: "2px",
  },
  onlineDot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    backgroundColor: "#22c55e",
    flexShrink: 0,
  },
  onlineText: {
    fontSize: "12px",
    color: "#64748b",
  },
  headerEmail: {
    fontSize: "12px",
    color: "#4a5568",
  },
};

export default ChatWindow;
