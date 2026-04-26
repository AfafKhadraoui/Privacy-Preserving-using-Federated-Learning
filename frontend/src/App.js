import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Register from './pages/Register';
import Recognize from './pages/Recognize';
import FLResults from './pages/FLResults';

export default function App() {
  return (
    <BrowserRouter>
      <nav style={styles.nav}>
        <Link to="/" style={styles.link}>Register</Link>
        <Link to="/recognize" style={styles.link}>Recognize</Link>
        {/* <Link to="/results" style={styles.link}>FL Results</Link> */}
      </nav>
      <Routes>
        <Route path="/" element={<Register />} />
        <Route path="/recognize" element={<Recognize />} />
        {/* <Route path="/results" element={<FLResults />} /> */}
      </Routes>
    </BrowserRouter>
  );
}

const styles = {
  nav: { background: '#1a1a2e', padding: '14px 24px', display: 'flex', gap: 24 },
  link: { color: '#e0e0e0', textDecoration: 'none', fontWeight: 600 }
};