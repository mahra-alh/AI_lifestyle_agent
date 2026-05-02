const express = require("express");
const cors = require("cors");

const app = express();   // ← THIS is what was missing

app.use(cors());
app.use(express.json());

// Test route
app.get("/", (req, res) => {
  res.send("Backend running 🚀");
});

// ML endpoint
app.post("/predict", (req, res) => {
  const input = req.body;

  let recommendation = "Default activity";

  if (input.weather === "hot") {
    recommendation = "Go to the beach 🏖️";
  } else if (input.weather === "cold") {
    recommendation = "Visit a museum 🏛️";
  }

  res.json({ recommendation });
});

const PORT = 5000;

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});