import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts'

const METRICS = ['accuracy', 'macro_f1', 'balanced_accuracy']

export default function ModelMetrics({ rfMetrics, xgbMetrics }) {
  if (!rfMetrics && !xgbMetrics) {
    return <div style={{ color: 'var(--text-muted)', padding: '1rem', textAlign: 'center' }}>No model results yet. Train models first.</div>
  }

  const data = METRICS.map(m => ({
    metric: m.replace(/_/g, ' '),
    'Random Forest': rfMetrics?.[m] ?? null,
    XGBoost: xgbMetrics?.[m] ?? null,
  })).filter(d => d['Random Forest'] !== null || d.XGBoost !== null)

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="metric" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <YAxis domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }} formatter={v => v?.toFixed(4)} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="Random Forest" fill="#10B981" />
        <Bar dataKey="XGBoost" fill="#00E5FF" />
      </BarChart>
    </ResponsiveContainer>
  )
}
