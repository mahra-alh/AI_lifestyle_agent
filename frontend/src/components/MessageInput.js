import { useState } from "react";

function MessageInput({ onSend, disabled }) {
  const [text, setText] = useState("");

  function handleSubmit() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  const isEmpty = !text.trim();

  return (
    <div style={styles.container}>
      <textarea
        rows={1}
        placeholder="Message your lifestyle agent..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        style={{
          ...styles.textarea,
          opacity: disabled ? 0.45 : 1,
          cursor: disabled ? "not-allowed" : "text",
        }}
      />
      <button
        onClick={handleSubmit}
        disabled={disabled || isEmpty}
        style={{
          ...styles.button,
          opacity: disabled || isEmpty ? 0.35 : 1,
          cursor: disabled || isEmpty ? "not-allowed" : "pointer",
        }}
      >
        ↑
      </button>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    alignItems: "flex-end",
    gap: "8px",
    padding: "12px 16px 16px",
    borderTop: "1px solid #1e2130",
    backgroundColor: "#0f1117",
  },
  textarea: {
    flex: 1,
    resize: "none",
    padding: "10px 14px",
    fontSize: "14px",
    lineHeight: "1.5",
    border: "1px solid #2a2d3e",
    borderRadius: "10px",
    outline: "none",
    fontFamily: "inherit",
    overflowY: "auto",
    maxHeight: "120px",
    backgroundColor: "#1a1b26",
    color: "#e2e8f0",
    caretColor: "#e2e8f0",
  },
  button: {
    width: "36px",
    height: "36px",
    fontSize: "18px",
    fontWeight: "700",
    color: "#ffffff",
    backgroundColor: "#2563eb",
    border: "none",
    borderRadius: "8px",
    flexShrink: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};

export default MessageInput;
