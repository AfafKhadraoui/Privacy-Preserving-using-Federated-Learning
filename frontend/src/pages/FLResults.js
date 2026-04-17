import { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function FLResults() {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get('http://localhost:5000/api/fl-results').then(r => setData(r.data));
  }, []);

  // Loading state
  if (!data) return <p style={{ padding: 40 }}>Loading...</p>;

  // FL hasn't been run yet
  if (data.status === 'pending' || data.rounds.length === 0) return (
    <div style={styles.page}>
      <h2 style={styles.title}>📊 Federated Learning Results</h2>
      <div style={styles.pending}>
        ⏳ FL simulation not run yet. Run <code>python src/federated/run_fl.py</code> first.
      </div>
    </div>
  );

  const lastRound = data.rounds.at(-1);

  return (
    <div style={styles.page}>
      <h2 style={styles.title}>📊 Federated Learning Results</h2>

      {/* Summary cards */}
      <div style={styles.cards}>
        {[
          { label: 'Clients',        value: data.clients },
          { label: 'Rounds',         value: data.num_rounds },
          { label: 'Final Accuracy', value: (lastRound.accuracy * 100).toFixed(1) + '%' },
          { label: 'Final Loss',     value: lastRound.loss.toFixed(4) },
          { label: 'Privacy ε',      value: data.privacy_epsilon },
          { label: 'DP Enabled',     value: data.dp ? 'Yes' : 'No' },
        ].map(c => (
          <div key={c.label} style={styles.card}>
            <div style={styles.cardVal}>{c.value}</div>
            <div style={styles.cardLabel}>{c.label}</div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div style={styles.chartBox}>
        <h3 style={styles.chartTitle}>Accuracy & Loss per Round</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data.rounds}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="round" label={{ value: 'Round', position: 'insideBottom', offset: -2 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="accuracy" stroke="#4caf50" strokeWidth={2} dot />
            <Line type="monotone" dataKey="loss"     stroke="#f44336" strokeWidth={2} dot />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p style={styles.note}>
        Raw biometric data never left client devices. Only model gradients were aggregated.
        {data.dp && ` Differential Privacy applied (ε = ${data.privacy_epsilon}).`}
      </p>
    </div>
  );
}

const styles = {
  page:       { padding: '30px 40px', maxWidth: 900, margin: '0 auto' },
  title:      { color: '#1a1a2e', marginBottom: 24 },
  pending:    { background: '#fff3cd', padding: 20, borderRadius: 10, color: '#856404' },
  cards:      { display: 'flex', gap: 12, marginBottom: 30, flexWrap: 'wrap' },
  card:       { background: '#1a1a2e', color: '#fff', padding: '16px 20px', borderRadius: 12, flex: 1, textAlign: 'center', minWidth: 110 },
  cardVal:    { fontSize: 22, fontWeight: 700 },
  cardLabel:  { fontSize: 12, color: '#aaa', marginTop: 4 },
  chartBox:   { background: '#fff', padding: 24, borderRadius: 12, boxShadow: '0 2px 12px rgba(0,0,0,0.08)' },
  chartTitle: { margin: '0 0 16px', color: '#333' },
  note:       { marginTop: 20, color: '#555', fontStyle: 'italic', fontSize: 14 },
};