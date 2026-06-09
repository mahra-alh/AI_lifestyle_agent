import { useState } from "react";

const API_BASE = (process.env.REACT_APP_API_URL || "").replace(/\/$/, "");

const DUBAI_AREAS = [
  "Dubai Marina", "Jumeirah Beach Residence", "Jumeirah Lake Towers",
  "Business Bay", "Downtown Dubai", "DIFC", "Deira", "Bur Dubai",
  "Al Barsha", "Barsha Heights", "Jumeirah Village Circle",
  "Jumeirah Village Triangle", "Dubai Hills", "Mirdif",
  "Dubai Silicon Oasis", "Dubai Internet City", "Dubai Media City",
  "Palm Jumeirah", "Al Quoz", "Karama", "Al Satwa", "Jumeirah",
  "Umm Suqeim", "Motor City", "Dubai Sports City", "Dubai South",
  "Discovery Gardens", "The Greens", "The Views",
];

const WORK_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

const ACTIVITY_TYPES = [
  "Food & Dining", "Outdoor & Sports", "Wellness & Relaxation",
  "Arts & Culture", "Shopping", "Entertainment", "Nightlife",
];

const HOBBIES = [
  "Restaurants", "Coffee Shops", "Beach Walks", "Hiking", "Gym",
  "Yoga", "Museums", "Shopping", "Movies", "Sports", "Cooking", "Reading",
];

const CUISINES = [
  "Arabic", "Indian", "Italian", "Japanese", "Mexican", "Chinese",
  "Lebanese", "Mediterranean", "Seafood", "American", "Cafe & Coffee",
  "Grills & BBQ", "Healthy", "Turkish", "French",
];

const ENVIRONMENTS = ["Indoor", "Outdoor", "Mixed"];
const ADVENTURE_LEVELS = ["Low", "Moderate", "High", "Very High"];

function ToggleChip({ label, selected, onToggle }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      style={{
        padding: "6px 14px",
        fontSize: "13px",
        borderRadius: "20px",
        border: `1px solid ${selected ? "#2563eb" : "#2a2d3e"}`,
        backgroundColor: selected ? "#1e3a8a" : "#1a1b26",
        color: selected ? "#93c5fd" : "#94a3b8",
        cursor: "pointer",
        transition: "all 0.15s",
      }}
    >
      {label}
    </button>
  );
}

function SectionLabel({ children }) {
  return <p style={styles.sectionLabel}>{children}</p>;
}

export default function ProfileSetupPage({ userEmail, onComplete }) {
  const [form, setForm] = useState({
    home_area: "",
    monthly_fun_budget_aed: "",
    max_per_activity_aed: "",
    work_days: [],
    work_start_time: "09:00",
    work_end_time: "18:00",
    preferred_activity_types: [],
    hobbies: [],
    interests: [],
    preferred_environment: "mixed",
    adventure_level_encoded: 1,
    social_alone: 0,
    social_partner: 0,
    social_friends: 0,
    social_family: 0,
    diet_halal: 0,
    diet_vegetarian: 0,
    diet_vegan: 0,
    diet_gluten_free: 0,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function set(field, value) {
    setForm((f) => ({ ...f, [field]: value }));
  }

  function toggleList(field, value) {
    setForm((f) => ({
      ...f,
      [field]: f[field].includes(value)
        ? f[field].filter((v) => v !== value)
        : [...f[field], value],
    }));
  }

  function toggleFlag(field) {
    setForm((f) => ({ ...f, [field]: f[field] === 1 ? 0 : 1 }));
  }

  function validate() {
    if (!form.home_area) return "Please select your home area.";
    if (!form.monthly_fun_budget_aed || Number(form.monthly_fun_budget_aed) <= 0)
      return "Please enter your monthly activity budget.";
    if (!form.max_per_activity_aed || Number(form.max_per_activity_aed) <= 0)
      return "Please enter your max budget per activity.";
    if (form.work_days.length === 0) return "Please select your work days.";
    if (form.preferred_activity_types.length === 0)
      return "Please select at least one activity type.";
    if (form.hobbies.length === 0) return "Please select at least one hobby.";
    return null;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const validationError = validate();
    if (validationError) {
      setError(validationError);
      return;
    }
    setError("");
    setLoading(true);

    try {
      const payload = {
        ...form,
        user_id: userEmail,
        monthly_fun_budget_aed: Number(form.monthly_fun_budget_aed),
        max_per_activity_aed: Number(form.max_per_activity_aed),
        work_days: form.work_days.map((d) => d.toLowerCase()),
        preferred_activity_types: form.preferred_activity_types.map((t) => t.toLowerCase()),
        hobbies: form.hobbies.map((h) => h.toLowerCase()),
        interests: form.interests.map((i) => i.toLowerCase()),
        preferred_environment: form.preferred_environment.toLowerCase(),
      };

      const res = await fetch(`${API_BASE}/api/profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Failed to save profile.");
      }

      onComplete();
    } catch (err) {
      setError(err.message || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.header}>
          <h1 style={styles.title}>Set Up Your Profile</h1>
          <p style={styles.subtitle}>
            Tell us about yourself so we can personalise your Dubai activity recommendations.
          </p>
          <p style={styles.emailBadge}>{userEmail}</p>
        </div>

        <form onSubmit={handleSubmit}>

          {/* ── Location & Budget ───────────────────────────────── */}
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Location & Budget</h2>

            <SectionLabel>Where are you based in Dubai?</SectionLabel>
            <select
              value={form.home_area}
              onChange={(e) => set("home_area", e.target.value)}
              style={styles.select}
            >
              <option value="">Select your area…</option>
              {DUBAI_AREAS.map((area) => (
                <option key={area} value={area}>{area}</option>
              ))}
            </select>

            <SectionLabel>Monthly activity budget (AED)</SectionLabel>
            <input
              type="number"
              min="0"
              placeholder="e.g. 800"
              value={form.monthly_fun_budget_aed}
              onChange={(e) => set("monthly_fun_budget_aed", e.target.value)}
              style={styles.input}
            />

            <SectionLabel>Max budget per activity (AED)</SectionLabel>
            <input
              type="number"
              min="0"
              placeholder="e.g. 150"
              value={form.max_per_activity_aed}
              onChange={(e) => set("max_per_activity_aed", e.target.value)}
              style={styles.input}
            />
          </div>

          {/* ── Work Schedule ───────────────────────────────────── */}
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Work Schedule</h2>
            <SectionLabel>Which days do you work?</SectionLabel>
            <div style={styles.chipRow}>
              {WORK_DAYS.map((day) => (
                <ToggleChip
                  key={day}
                  label={day.slice(0, 3)}
                  selected={form.work_days.includes(day)}
                  onToggle={() => toggleList("work_days", day)}
                />
              ))}
            </div>

            <div style={styles.timeRow}>
              <div style={styles.timeField}>
                <SectionLabel>Work starts</SectionLabel>
                <input
                  type="time"
                  value={form.work_start_time}
                  onChange={(e) => set("work_start_time", e.target.value)}
                  style={styles.input}
                />
              </div>
              <div style={styles.timeField}>
                <SectionLabel>Work ends</SectionLabel>
                <input
                  type="time"
                  value={form.work_end_time}
                  onChange={(e) => set("work_end_time", e.target.value)}
                  style={styles.input}
                />
              </div>
            </div>
          </div>

          {/* ── Activity Preferences ────────────────────────────── */}
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Activity Preferences</h2>

            <SectionLabel>What types of activities do you enjoy?</SectionLabel>
            <div style={styles.chipRow}>
              {ACTIVITY_TYPES.map((type) => (
                <ToggleChip
                  key={type}
                  label={type}
                  selected={form.preferred_activity_types.includes(type)}
                  onToggle={() => toggleList("preferred_activity_types", type)}
                />
              ))}
            </div>

            <SectionLabel>Pick your hobbies & interests</SectionLabel>
            <div style={styles.chipRow}>
              {HOBBIES.map((h) => (
                <ToggleChip
                  key={h}
                  label={h}
                  selected={form.hobbies.includes(h)}
                  onToggle={() => toggleList("hobbies", h)}
                />
              ))}
            </div>
          </div>

          {/* ── Favourite Cuisines ──────────────────────────────── */}
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Favourite Cuisines</h2>
            <SectionLabel>Select the cuisines you love — we'll prioritise matching restaurants.</SectionLabel>
            <div style={styles.chipRow}>
              {CUISINES.map((c) => (
                <ToggleChip
                  key={c}
                  label={c}
                  selected={form.interests.includes(c)}
                  onToggle={() => toggleList("interests", c)}
                />
              ))}
            </div>
          </div>

          {/* ── Lifestyle ───────────────────────────────────────── */}
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Your Lifestyle</h2>

            <SectionLabel>Preferred environment</SectionLabel>
            <div style={styles.chipRow}>
              {ENVIRONMENTS.map((env) => (
                <ToggleChip
                  key={env}
                  label={env}
                  selected={form.preferred_environment === env.toLowerCase()}
                  onToggle={() => set("preferred_environment", env.toLowerCase())}
                />
              ))}
            </div>

            <SectionLabel>Adventure level</SectionLabel>
            <div style={styles.chipRow}>
              {ADVENTURE_LEVELS.map((level, idx) => (
                <ToggleChip
                  key={level}
                  label={level}
                  selected={form.adventure_level_encoded === idx}
                  onToggle={() => set("adventure_level_encoded", idx)}
                />
              ))}
            </div>

            <SectionLabel>I usually go out…</SectionLabel>
            <div style={styles.chipRow}>
              {[
                ["Solo", "social_alone"],
                ["With Partner", "social_partner"],
                ["With Friends", "social_friends"],
                ["With Family", "social_family"],
              ].map(([label, field]) => (
                <ToggleChip
                  key={field}
                  label={label}
                  selected={form[field] === 1}
                  onToggle={() => toggleFlag(field)}
                />
              ))}
            </div>
          </div>

          {/* ── Dietary Needs ───────────────────────────────────── */}
          <div style={styles.section}>
            <h2 style={styles.sectionTitle}>Dietary Needs</h2>
            <SectionLabel>Select any that apply — we'll filter out incompatible venues.</SectionLabel>
            <div style={styles.chipRow}>
              {[
                ["Halal", "diet_halal"],
                ["Vegetarian", "diet_vegetarian"],
                ["Vegan", "diet_vegan"],
                ["Gluten-free", "diet_gluten_free"],
              ].map(([label, field]) => (
                <ToggleChip
                  key={field}
                  label={label}
                  selected={form[field] === 1}
                  onToggle={() => toggleFlag(field)}
                />
              ))}
            </div>
          </div>

          {error && <p style={styles.error}>{error}</p>}

          <button
            type="submit"
            disabled={loading}
            style={{ ...styles.submitButton, opacity: loading ? 0.6 : 1 }}
          >
            {loading ? "Saving…" : "Save & Start Planning"}
          </button>
        </form>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    backgroundColor: "#0f1117",
    display: "flex",
    justifyContent: "center",
    padding: "40px 16px",
  },
  container: {
    width: "100%",
    maxWidth: "600px",
  },
  header: {
    marginBottom: "32px",
  },
  title: {
    fontSize: "24px",
    fontWeight: "700",
    color: "#e2e8f0",
    margin: "0 0 8px",
  },
  subtitle: {
    fontSize: "14px",
    color: "#64748b",
    margin: "0 0 12px",
    lineHeight: "1.6",
  },
  emailBadge: {
    display: "inline-block",
    fontSize: "12px",
    color: "#94a3b8",
    backgroundColor: "#1a1b26",
    border: "1px solid #2a2d3e",
    borderRadius: "20px",
    padding: "4px 12px",
    margin: 0,
  },
  section: {
    backgroundColor: "#151821",
    border: "1px solid #1e2130",
    borderRadius: "12px",
    padding: "20px",
    marginBottom: "16px",
  },
  sectionTitle: {
    fontSize: "15px",
    fontWeight: "600",
    color: "#e2e8f0",
    margin: "0 0 16px",
  },
  sectionLabel: {
    fontSize: "13px",
    color: "#94a3b8",
    margin: "0 0 10px",
  },
  chipRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: "8px",
    marginBottom: "16px",
  },
  select: {
    width: "100%",
    padding: "10px 12px",
    fontSize: "14px",
    backgroundColor: "#1a1b26",
    color: "#e2e8f0",
    border: "1px solid #2a2d3e",
    borderRadius: "8px",
    outline: "none",
    marginBottom: "16px",
    boxSizing: "border-box",
  },
  input: {
    width: "100%",
    padding: "10px 12px",
    fontSize: "14px",
    backgroundColor: "#1a1b26",
    color: "#e2e8f0",
    border: "1px solid #2a2d3e",
    borderRadius: "8px",
    outline: "none",
    marginBottom: "16px",
    boxSizing: "border-box",
  },
  timeRow: {
    display: "flex",
    gap: "16px",
  },
  timeField: {
    flex: 1,
  },
  error: {
    fontSize: "13px",
    color: "#f87171",
    backgroundColor: "#1a1b26",
    border: "1px solid #7f1d1d",
    borderRadius: "8px",
    padding: "10px 14px",
    marginBottom: "16px",
  },
  submitButton: {
    width: "100%",
    padding: "14px",
    fontSize: "15px",
    fontWeight: "600",
    color: "#ffffff",
    backgroundColor: "#2563eb",
    border: "none",
    borderRadius: "10px",
    cursor: "pointer",
    marginBottom: "40px",
  },
};
