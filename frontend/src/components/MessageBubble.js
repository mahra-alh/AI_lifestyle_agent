import { parseAgentResponse } from "../utils/parseAgentResponse";
import RecommendationView from "./RecommendationView";
import BookingConfirmCard from "./BookingConfirmCard";

function MessageBubble({ role, text, userEmail, pendingBooking }) {
  const isUser = role === "user";

  if (!isUser) {
    const parsed = parseAgentResponse(text);

    if (parsed.type === "recommendations") {
      return (
        <div style={styles.agentRow}>
          <div style={styles.agentColumn}>
            <RecommendationView parsed={parsed} userEmail={userEmail} />
            {pendingBooking && (
              <BookingConfirmCard pendingBooking={pendingBooking} />
            )}
          </div>
        </div>
      );
    }

    return (
      <div style={styles.agentRow}>
        <div style={styles.agentColumn}>
          <p style={styles.agentText}>
            {text.split("\n").map((line, i, arr) => (
              <span key={i}>
                {line}
                {i < arr.length - 1 && <br />}
              </span>
            ))}
          </p>
          {pendingBooking && (
            <BookingConfirmCard pendingBooking={pendingBooking} />
          )}
        </div>
      </div>
    );
  }

  return (
    <div style={styles.userRow}>
      <div style={styles.userBubble}>
        {text.split("\n").map((line, i, arr) => (
          <span key={i}>
            {line}
            {i < arr.length - 1 && <br />}
          </span>
        ))}
      </div>
    </div>
  );
}

const styles = {
  userRow: {
    display: "flex",
    justifyContent: "flex-end",
    marginBottom: "16px",
  },
  userBubble: {
    maxWidth: "65%",
    padding: "10px 16px",
    borderRadius: "18px",
    borderBottomRightRadius: "4px",
    backgroundColor: "#2563eb",
    color: "#ffffff",
    fontSize: "14px",
    lineHeight: "1.6",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  agentRow: {
    display: "flex",
    justifyContent: "flex-start",
    marginBottom: "20px",
  },
  agentColumn: {
    display: "flex",
    flexDirection: "column",
    maxWidth: "80%",
  },
  agentText: {
    maxWidth: "80%",
    margin: 0,
    fontSize: "14px",
    color: "#e2e8f0",
    lineHeight: "1.7",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
};

export default MessageBubble;
