const WEATHER_CONDITIONS = [
  "partly cloudy", "sunny", "clear", "cloudy", "overcast",
  "rainy", "drizzle", "stormy", "foggy", "hazy", "windy",
];

function extractWeather(text) {
  // Preferred: the agent's standard first line, e.g.
  // "Weather for Friday, 2026-06-12: 34°C, Sunny"
  const dated = text.match(
    /Weather for ([^:\n]+):\s*(-?\d+(?:\.\d+)?)\s*°C[,\s]*([^\n.]*)/i
  );
  if (dated) {
    return {
      dateLabel: dated[1].trim(),
      temp: Math.round(parseFloat(dated[2])),
      condition: dated[3].trim() || null,
    };
  }

  // Fallback: any temperature / condition words found in the text
  const tempMatch = text.match(/(\d+)\s*°C/);
  const lower = text.toLowerCase();
  const condition = WEATHER_CONDITIONS.find((c) => lower.includes(c));
  if (!tempMatch && !condition) return null;
  return {
    dateLabel: null,
    temp: tempMatch ? parseInt(tempMatch[1]) : null,
    condition: condition
      ? condition.split(" ").map((w) => w[0].toUpperCase() + w.slice(1)).join(" ")
      : null,
  };
}

function extractCalendar(text) {
  // "free from 2:00 PM to 5:00 PM" or "free slot 14:00–17:00"
  const freeMatch = text.match(
    /free\D{0,25}?(\d{1,2}:\d{2}\s*(?:AM|PM)?)\s*(?:to|[-–])\s*(\d{1,2}:\d{2}\s*(?:AM|PM)?)/i
  );
  if (freeMatch) return { start: freeMatch[1].trim(), end: freeMatch[2].trim() };

  // generic time range as fallback
  const rangeMatch = text.match(
    /(\d{1,2}:\d{2}\s*(?:AM|PM))\s*[-–to]+\s*(\d{1,2}:\d{2}\s*(?:AM|PM))/i
  );
  return rangeMatch ? { start: rangeMatch[1].trim(), end: rangeMatch[2].trim() } : null;
}

function extractScore(block) {
  const pct = block.match(/\b(\d{2,3})%/);
  if (pct) return parseInt(pct[1]);
  const decimal = block.match(/[Ss]core[:\s]+([\d.]+)/);
  if (decimal) {
    const v = parseFloat(decimal[1]);
    return v <= 1 ? Math.round(v * 100) : Math.round(v);
  }
  return null;
}

function parseItems(text) {
  // split text on lines that begin a new numbered entry
  const chunks = text.split(/\n(?=\d+[\.\)])/);

  return chunks
    .filter((c) => /^\d+[\.\)]/.test(c.trim()))
    .map((block) => {
      const clean = block.trim();

      // prefer bold-marked name
      const boldMatch = clean.match(/\*\*([^*]{2,80})\*\*/);
      const name = boldMatch
        ? boldMatch[1].trim()
        : clean
            .replace(/^\d+[\.\)]\s*/, "")
            .split(/\s{2,}|[-–—:|]/)[0]
            .trim()
            .slice(0, 80);

      // description is what follows the name up to the first newline
      let desc = "";
      if (boldMatch) {
        desc = clean
          .slice(clean.indexOf(boldMatch[0]) + boldMatch[0].length)
          .replace(/^[\s\-–—:]+/, "")
          .split("\n")[0];
      } else {
        const parts = clean.replace(/^\d+[\.\)]\s*/, "").split(/[-–—]/);
        desc = parts.slice(1).join("-").split("\n")[0];
      }
      // strip trailing score/badge fragments
      desc = desc
        .replace(/[\.\s]*\(?\d{2,3}%\)?[\s\S]*$/, "")
        .replace(/\|.*$/, "")
        .trim();

      const indoorOutdoor = /\boutdoor\b/i.test(clean)
        ? "Outdoor"
        : /\bindoor\b/i.test(clean)
        ? "Indoor"
        : null;

      const durMatch = clean.match(/~?(\d+)\s*min/i) || clean.match(/(\d+)\s*hour/i);
      const duration = durMatch
        ? /hour/i.test(clean)
          ? `${durMatch[1]}h`
          : `${durMatch[1]} min`
        : null;

      return { name, description: desc, score: extractScore(clean), indoorOutdoor, duration };
    })
    .filter((item) => item.name && item.name.length > 1);
}

export function parseAgentResponse(text) {
  if (!text) return { type: "text", text: "" };

  const hasNumberedItems = /^\d+[\.\)]/m.test(text);
  const hasIntent = /\b(here are|recommend|suggest|options|activities|ideas|consider|picks)\b/i.test(text);

  if (!hasNumberedItems || !hasIntent) return { type: "text", text };

  const items = parseItems(text);
  if (items.length < 2) return { type: "text", text };

  const summaryMatch = text.match(/^([\s\S]*?)(?=\n*\d+[\.\)])/);
  const summary = summaryMatch ? summaryMatch[1].replace(/\n+$/, "").trim() : "";

  return {
    type: "recommendations",
    summary,
    weather: extractWeather(text),
    calendar: extractCalendar(text),
    items,
    fullText: text,
  };
}

export function weatherIcon(condition) {
  if (!condition) return "🌤️";
  const c = condition.toLowerCase();
  if (c.includes("sunny") || c.includes("clear")) return "☀️";
  if (c.includes("partly")) return "⛅";
  if (c.includes("rain") || c.includes("drizzle")) return "🌧️";
  if (c.includes("storm")) return "⛈️";
  if (c.includes("fog") || c.includes("haze")) return "🌫️";
  if (c.includes("cloud") || c.includes("overcast")) return "☁️";
  return "🌤️";
}
