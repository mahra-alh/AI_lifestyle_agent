import { useState } from "react";

function EmailEntry({ onEmailSubmit }) {
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");

  function handleSubmit(e) {
    e.preventDefault();

    const trimmed = email.trim().toLowerCase();

    if (!trimmed) {
      setError("Please enter your email address.");
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmed)) {
      setError("Please enter a valid email address.");
      return;
    }

    setError("");
    onEmailSubmit(trimmed);
  }

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <h1 style={styles.title}>Dubai Lifestyle Agent</h1>
        <p style={styles.subtitle}>
          Your free time called. We checked the weather and made plans, please enter your email.
        </p>
        <form onSubmit={handleSubmit} style={styles.form}>
          <input
            type="email"
            placeholder="your@email.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={styles.input}
            autoFocus
          />
          {error && <p style={styles.error}>{error}</p>}
          <button type="submit" style={styles.button}>
            Get Started
          </button>
        </form>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100vh",
    backgroundColor: "#494948",
  },
  card: {
    backgroundColor: "#dee85c",
    borderRadius: "12px",
    padding: "48px 40px",
    width: "100%",
    maxWidth: "420px",
    boxShadow: "0 4px 24px rgba(0,0,0,0.08)",
    textAlign: "center",
  },
  title: {
    fontSize: "24px",
    fontWeight: "700",
    color: "#1a1a2e",
    margin: "0 0 8px",
  },
  subtitle: {
    fontSize: "14px",
    color: "#666",
    margin: "0 0 32px",
    lineHeight: "1.5",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  input: {
    padding: "12px 16px",
    fontSize: "15px",
    border: "1px solid #787575",
    borderRadius: "8px",
    outline: "none",
    width: "100%",
    boxSizing: "border-box",
  },
  error: {
    fontSize: "13px",
    color: "#e53e3e",
    margin: "0",
    textAlign: "left",
  },
  button: {
    padding: "12px",
    fontSize: "15px",
    fontWeight: "600",
    color: "#fffefe",
    backgroundColor: "#516deb",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    marginTop: "4px",
  },
};

export default EmailEntry;
