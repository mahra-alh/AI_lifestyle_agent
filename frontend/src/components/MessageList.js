import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";

function MessageList({ messages, loading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div style={styles.container}>
      {messages.length === 0 && (
        <p style={styles.empty}>
          Ask me anything — I can suggest activities, check the weather, or plan your free time in Dubai.
        </p>
      )}

      {messages.map((msg, i) => (
        <MessageBubble key={i} role={msg.role} text={msg.text} />
      ))}

      {loading && (
        <div style={styles.typingRow}>
          <div style={styles.typingBubble}>
            <span style={{ ...styles.dot, animationDelay: "0s" }} />
            <span style={{ ...styles.dot, animationDelay: "0.2s" }} />
            <span style={{ ...styles.dot, animationDelay: "0.4s" }} />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}

const styles = {
  container: {
    flex: 1,
    overflowY: "auto",
    padding: "24px 20px 8px",
    display: "flex",
    flexDirection: "column",
  },
  empty: {
    textAlign: "center",
    color: "#4a5568",
    fontSize: "14px",
    marginTop: "auto",
    marginBottom: "auto",
    padding: "0 32px",
    lineHeight: "1.7",
  },
  typingRow: {
    display: "flex",
    justifyContent: "flex-start",
    marginBottom: "16px",
  },
  typingBubble: {
    display: "flex",
    alignItems: "center",
    gap: "5px",
    backgroundColor: "#1a1b26",
    border: "1px solid #2a2d3e",
    borderRadius: "16px",
    borderBottomLeftRadius: "4px",
    padding: "10px 14px",
  },
  dot: {
    display: "inline-block",
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    backgroundColor: "#4a5568",
    animation: "bounce 1.2s infinite ease-in-out",
  },
};

export default MessageList;
