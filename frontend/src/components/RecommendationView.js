import ActivityCard from "./ActivityCard";
import { weatherIcon } from "../utils/parseAgentResponse";

function RecommendationView({ parsed }) {
  const { summary, weather, calendar, items } = parsed;

  const hasPills = weather || calendar;

  return (
    <div style={styles.container}>
      {summary && <p style={styles.summary}>{summary}</p>}

      {hasPills && (
        <div style={styles.pills}>
          {weather && (
            <span style={styles.pill}>
              {weatherIcon(weather.condition)}{" "}
              {weather.temp !== null ? `${weather.temp}°C` : ""}{" "}
              {weather.condition || ""}
            </span>
          )}
          {calendar && (
            <span style={styles.pill}>
              📅 Free {calendar.start}–{calendar.end}
            </span>
          )}
        </div>
      )}

      <div style={styles.cards}>
        {items.map((item, i) => (
          <ActivityCard key={i} item={item} />
        ))}
      </div>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "560px",
    width: "100%",
    padding: "4px 0 8px",
  },
  summary: {
    fontSize: "14px",
    color: "#e2e8f0",
    lineHeight: "1.6",
    margin: "0 0 12px",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  pills: {
    display: "flex",
    gap: "8px",
    flexWrap: "wrap",
    marginBottom: "14px",
  },
  pill: {
    display: "inline-flex",
    alignItems: "center",
    gap: "4px",
    fontSize: "12px",
    fontWeight: "500",
    color: "#cbd5e1",
    backgroundColor: "#1e2130",
    border: "1px solid #2a2d3e",
    borderRadius: "20px",
    padding: "4px 12px",
  },
  cards: {
    display: "flex",
    flexDirection: "column",
  },
};

export default RecommendationView;
