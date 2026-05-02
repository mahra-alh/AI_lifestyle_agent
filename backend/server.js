require('dotenv').config();
const express = require('express');
const cors = require('cors');

const app = express();

app.use(cors({
  origin: [
    'http://localhost:3000',
    'https://your-project.web.app',
    'https://your-project.firebaseapp.com'
  ]
}));

app.use(express.json());

// test route
app.get('/', (req, res) => {
  res.send('Backend running');
});

// API route
app.get('/api/test', (req, res) => {
  res.json({ message: 'Hello from backend' });
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});