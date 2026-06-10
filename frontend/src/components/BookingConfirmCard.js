import { useState } from "react";

const API_BASE = (
  process.env.REACT_APP_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

function formatTime(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleString([], {
      weekday: "short",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function BookingConfirmCard({ pendingBooking }) {
  const { pending_id, activity_name, start_time, end_time, location } =
    pendingBooking;

  // idle | working | confirmed | cancelled | error
  const [state, setState] = useState("idle");
  const [eventLink, setEventLink] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  async function act(action) {
    setState("working");
    setErrorMessage("");

    try {
      const res = await fetch(
        `${API_BASE}/api/bookings/${encodeURIComponent(pending_id)}/${action}`,
        { method: "POST" }
      );
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || data.message || "Request failed");
      }

      if (action === "confirm") {
        setEventLink(data.event_link || null);
        setState("confirmed");
      } else {
        setState("cancelled");
      }
    } catch (err) {
      setErrorMessage(err.message || "Something went wrong.");
      setState("error");
    }
  }

  return (
    <div style={styles.card}>
      <div style={styles.header}>📅 Booking confirmation</div>
      <div style={styles.name}>{activity_name}</div>
      <div style={styles.detail}>
        {formatTime(start_time)} – {formatTime(end_time)}
      </div>
      {location && <div style={styles.detail}>📍 {location}</div>}

      {state === "confirmed" && (
        <div style={styles.confirmedNote}>
          ✅ Booked in your Google Calendar.{" "}
          {eventLink && (
            <a href={eventLink} target="_blank" rel="noreferrer" style={styles.link}>
              View event
            </a>
          )}
        </div>
      )}

      {state === "cancelled" && (
        <div style={styles.cancelledNote}>✖ Booking cancelled.</div>
      )}

      {state === "error" && (
        <div style={styles.errorNote}>⚠ {errorMessage}</div>
      )}

      {(state === "idle" || state === "working" || state === "error") && (
        <div style={styles.buttonRow}>
          <button
            style={styles.confirmButton}
            onClick={() => act("confirm")}
            disabled={state === "working"}
          >
            {state === "working" ? "Working…" : "✅ Confirm booking"}
          </button>
          <button
            style={styles.cancelButton}
            onClick={() => act("cancel")}
            disabled={state === "working"}
          >
            ✖ Cancel
          </button>
        </div>
      )}
    </div>
  );
}

const styles = {
  card: {
    maxWidth: "420px",
    backgroundColor: "#1a1b26",
    border: "1px solid #2563eb",
    borderRadius: "12px",
    padding: "14px 16px",
    marginTop: "10px",
  },
  header: {
    fontSize: "11px",
    fontWeight: "700",
    color: "#60a5fa",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    marginBottom: "8px",
  },
  name: {
    fontSize: "15px",
    fontWeight: "600",
    color: "#e2e8f0",
    marginBottom: "4px",
  },
  detail: {
    fontSize: "13px",
    color: "#94a3b8",
    marginBottom: "4px",
  },
  buttonRow: {
    display: "flex",
    gap: "8px",
    marginTop: "12px",
  },
  confirmButton: {
    flex: 1,
    padding: "8px 14px",
    borderRadius: "8px",
    border: "none",
    backgroundColor: "#2563eb",
    color: "#ffffff",
    fontSize: "13px",
    fontWeight: "600",
    cursor: "pointer",
  },
  cancelButton: {
    padding: "8px 14px",
    borderRadius: "8px",
    border: "1px solid #2a2d3e",
    backgroundColor: "#0f1117",
    color: "#94a3b8",
    fontSize: "13px",
    cursor: "pointer",
  },
  confirmedNote: {
    marginTop: "10px",
    fontSize: "13px",
    color: "#4ade80",
  },
  cancelledNote: {
    marginTop: "10px",
    fontSize: "13px",
    color: "#94a3b8",
  },
  errorNote: {
    marginTop: "10px",
    fontSize: "13px",
    color: "#fb923c",
  },
  link: {
    color: "#60a5fa",
  },
};

export default BookingConfirmCard;
