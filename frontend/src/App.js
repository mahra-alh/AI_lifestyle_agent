import { useEffect } from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  const apiBaseUrl = (process.env.REACT_APP_API_URL || '').replace(/\/$/, '');
  const apiPath = apiBaseUrl ? `${apiBaseUrl}/api/test` : '/api/test';

  useEffect(() => {
    fetch(apiPath)
      .then(res => res.json())
      .then(data => console.log('Backend says:', data))
      .catch(err => console.error(err));
  }, [apiPath]);

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>Backend connection test — check console 👀</p>
      </header>
    </div>
  );
}

export default App;

console.log("API URL:", process.env.REACT_APP_API_URL || '(same-origin)');
