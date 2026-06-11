import { useState } from "react";

const API_BASE = (
  process.env.REACT_APP_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const ICON_MAP = [
  [["bike", "cycling", "cycle"], "🚴"],
  [["hike", "trail", "trek", "walk"], "🥾"],
  [["cafe", "coffee"], "☕"],
  [["restaurant", "dining", "food", "eat"], "🍽️"],
  [["beach", "swim", "pool"], "🏖️"],
  [["gym", "fitness", "workout"], "🏋️"],
  [["yoga", "pilates", "meditation"], "🧘"],
  [["museum", "gallery", "art", "exhibit"], "🎨"],
  [["park", "garden", "nature"], "🌿"],
  [["mall", "shop", "market", "souk"], "🛍️"],
  [["movie", "cinema", "film"], "🎬"],
  [["kayak", "boat", "sail", "yacht"], "🚣"],
  [["run", "jog", "sprint"], "🏃"],
  [["desert", "dune", "safari"], "🏜️"],
  [["golf"], "⛳"],
  [["tennis", "squash", "padel"], "🎾"],
  [["spa", "massage", "relax"], "💆"],
  [["frame", "tower", "burj", "view", "rooftop"], "🏙️"],
  [["creek", "marina", "harbour"], "⚓"],
];

function getIcon(name) {
  const lower = name.toLowerCase();
  for (const [keywords, icon] of ICON_MAP) {
    if (keywords.some((k) => lower.includes(k))) return icon;
  }
  return "⭐";
}

function getMatchStyle(score) {
  if (score === null) return null;
  if (score >= 80) return { bg: "#14532d", text: "#4ade80" };
  if (score >= 60) return { bg: "#422006", text: "#fbbf24" };
  return { bg: "#431407", text: "#fb923c" };
}

// Venue fields shown in the expanded details view, in display order.
// [key in the venue pool row, label, formatter]
const DETAIL_FIELDS = [
  ["area", "📍 Area", null],
  ["location_area", "📍 Area", null],
  ["category", "🏷 Category", null],
  ["primary_category", "🏷 Category", null],
  ["budget_level", "💰 Budget level", null],
  ["meal_cost_for_one", "💵 Cost for one", (v) => `AED ${Number(v).toFixed(0)}`],
  ["has_outdoor_seating", "🌤 Outdoor seating", yesNo],
  ["serves_alcohol", "🍷 Serves alcohol", yesNo],
  ["has_shisha", "💨 Shisha", yesNo],
];

function yesNo(v) {
  return v === true || v === 1 || v === "1" ? "Yes" : "No";
}

function buildDetailRows(venue) {
  const rows = [];
  const usedLabels = new Set();
  for (const [key, label, format] of DETAIL_FIELDS) {
    const value = venue[key];
    if (value === null || value === undefined || value === "") continue;
    if (usedLabels.has(label)) continue; // skip duplicate-label fallbacks
    usedLabels.add(label);
    rows.push([label, format ? format(value) : String(value)]);
  }
  return rows;
}

function ActivityCard({ item, userEmail, rank }) {
  const { name, description, score, indoorOutdoor, duration } = item;
  const matchStyle = getMatchStyle(score);
  const cleanDescription = description
    ? description.replace(/\*\*([^*]+)\*\*/g, "$1").replace(/\*([^*]+)\*/g, "$1").trim()
    : "";

  // null = no feedback yet, "liked"/"disliked" = sent, "sending" = in flight
  const [feedbackState, setFeedbackState] = useState(null);

  // Expanded venue details
  const [expanded, setExpanded] = useState(false);
  const [venue, setVenue] = useState(null);
  // idle | loading | loaded | error
  const [detailsState, setDetailsState] = useState("idle");

  async function toggleDetails() {
    const next = !expanded;
    setExpanded(next);

    // Fetch venue details from the backend the first time the card is opened.
    if (next && detailsState === "idle") {
      setDetailsState("loading");
      try {
        const res = await fetch(
          `${API_BASE}/api/venue/${encodeURIComponent(name)}`
        );
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Venue not found");
        setVenue(data.venue);
        setDetailsState("loaded");
      } catch (err) {
        setDetailsState("error");
      }
    }
  }

  async function sendFeedback(feedback) {
    if (feedbackState) return; // already sent or sending
    setFeedbackState("sending");

    // The cards are parsed from chat text, so the real recommendation/venue
    // IDs are not available here — use a slug of the name as a stable stand-in.
    const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

    try {
      const res = await fetch(`${API_BASE}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userEmail,
          recommendation_id: `ui_${slug}`,
          activity_id: slug,
          activity_name: name,
          feedback,
          recommendation_rank: rank,
          session_id: "ui_card_feedback",
        }),
      });
      if (!res.ok) throw new Error("Request failed");
      setFeedbackState(feedback);
    } catch (err) {
      setFeedbackState(null); // allow retry on failure
    }
  }

  const detailRows = venue ? buildDetailRows(venue) : [];
  const venueDescription =
    venue && venue.description && String(venue.description).trim();
  const mapsLink =
    venue && venue.latitude && venue.longitude
      ? `https://www.google.com/maps?q=${venue.latitude},${venue.longitude}`
      : null;

  return (
    <div style={styles.card} onClick={toggleDetails} title="Click for venue details">
      <div style={styles.topRow}>
        <div style={styles.left}>
          <span style={styles.icon}>{getIcon(name)}</span>
          <div style={styles.body}>
            <div style={styles.name}>{name}</div>
            {cleanDescription && <div style={styles.description}>{cleanDescription}</div>}
            {(duration || indoorOutdoor) && (
              <div style={styles.tags}>
                {duration && (
                  <span style={styles.tag}>⏱ {duration}</span>
                )}
                {indoorOutdoor && (
                  <span style={styles.tag}>{indoorOutdoor}</span>
                )}
              </div>
            )}
          </div>
        </div>
        <div style={styles.right}>
          {matchStyle && (
            <div
              style={{
                ...styles.badge,
                backgroundColor: matchStyle.bg,
                color: matchStyle.text,
              }}
            >
              {score}% match
            </div>
          )}
          <div style={styles.feedbackRow}>
            <button
              style={{
                ...styles.feedbackButton,
                ...(feedbackState === "liked" ? styles.feedbackLiked : {}),
              }}
              onClick={(e) => {
                e.stopPropagation(); // don't toggle details when voting
                sendFeedback("liked");
              }}
              disabled={!!feedbackState}
              title="I like this suggestion"
            >
              👍
            </button>
            <button
              style={{
                ...styles.feedbackButton,
                ...(feedbackState === "disliked" ? styles.feedbackDisliked : {}),
              }}
              onClick={(e) => {
                e.stopPropagation();
                sendFeedback("disliked");
              }}
              disabled={!!feedbackState}
              title="Not for me"
            >
              👎
            </button>
          </div>
          <span style={styles.expandHint}>{expanded ? "▲ less" : "▼ details"}</span>
        </div>
      </div>

      {expanded && (
        <div style={styles.details} onClick={(e) => e.stopPropagation()}>
          {detailsState === "loading" && (
            <div style={styles.detailsNote}>Loading venue details…</div>
          )}
          {detailsState === "error" && (
            <div style={styles.detailsNote}>
              Could not load details for this venue.
            </div>
          )}
          {detailsState === "loaded" && (
            <>
              {venueDescription && (
                <div style={styles.detailsDescription}>{venueDescription}</div>
              )}
              <div style={styles.detailsGrid}>
                {detailRows.map(([label, value]) => (
                  <div key={label} style={styles.detailRow}>
                    <span style={styles.detailLabel}>{label}</span>
                    <span style={styles.detailValue}>{value}</span>
                  </div>
                ))}
              </div>
              {mapsLink && (
                <a
                  href={mapsLink}
                  target="_blank"
                  rel="noreferrer"
                  style={styles.mapsLink}
                >
                  🗺 Open in Google Maps
                </a>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

const styles = {
  card: {
    display: "flex",
    flexDirection: "column",
    backgroundColor: "#1a1b26",
    border: "1px solid #2a2d3e",
    borderRadius: "10px",
    padding: "12px 14px",
    marginBottom: "8px",
    cursor: "pointer",
  },
  topRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: "12px",
  },
  left: {
    display: "flex",
    alignItems: "flex-start",
    gap: "10px",
    flex: 1,
    minWidth: 0,
  },
  icon: {
    fontSize: "20px",
    flexShrink: 0,
    marginTop: "1px",
  },
  body: {
    flex: 1,
    minWidth: 0,
  },
  name: {
    fontSize: "14px",
    fontWeight: "600",
    color: "#e2e8f0",
    marginBottom: "2px",
  },
  description: {
    fontSize: "12px",
    color: "#94a3b8",
    marginBottom: "6px",
    lineHeight: "1.4",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  tags: {
    display: "flex",
    gap: "6px",
    flexWrap: "wrap",
  },
  tag: {
    fontSize: "11px",
    color: "#64748b",
    backgroundColor: "#0f1117",
    padding: "2px 8px",
    borderRadius: "12px",
    border: "1px solid #2a2d3e",
  },
  badge: {
    fontSize: "11px",
    fontWeight: "700",
    padding: "4px 10px",
    borderRadius: "20px",
    whiteSpace: "nowrap",
    flexShrink: 0,
  },
  right: {
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-end",
    gap: "6px",
    flexShrink: 0,
  },
  feedbackRow: {
    display: "flex",
    gap: "6px",
  },
  feedbackButton: {
    fontSize: "13px",
    padding: "3px 9px",
    borderRadius: "12px",
    border: "1px solid #2a2d3e",
    backgroundColor: "#0f1117",
    cursor: "pointer",
    lineHeight: 1.4,
  },
  feedbackLiked: {
    backgroundColor: "#14532d",
    borderColor: "#4ade80",
  },
  feedbackDisliked: {
    backgroundColor: "#431407",
    borderColor: "#fb923c",
  },
  expandHint: {
    fontSize: "10px",
    color: "#64748b",
  },
  details: {
    marginTop: "10px",
    paddingTop: "10px",
    borderTop: "1px solid #2a2d3e",
    cursor: "default",
  },
  detailsNote: {
    fontSize: "12px",
    color: "#64748b",
  },
  detailsDescription: {
    fontSize: "12px",
    color: "#cbd5e1",
    lineHeight: "1.5",
    marginBottom: "8px",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  detailsGrid: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  detailRow: {
    display: "flex",
    justifyContent: "space-between",
    gap: "12px",
    fontSize: "12px",
  },
  detailLabel: {
    color: "#64748b",
    flexShrink: 0,
  },
  detailValue: {
    color: "#e2e8f0",
    textAlign: "right",
    wordBreak: "break-word",
  },
  mapsLink: {
    display: "inline-block",
    marginTop: "8px",
    fontSize: "12px",
    color: "#60a5fa",
  },
};

export default ActivityCard;
