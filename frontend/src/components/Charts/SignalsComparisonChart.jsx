import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid, ReferenceLine, Cell,
} from 'recharts'

const SIGNAL_META = [
  { key: 't1_ndvi', key2: 't2_ndvi', delta: 'delta_ndvi', label: 'Vegetation Greenness', color: '#40916C' },
  { key: 't1_ndwi', key2: 't2_ndwi', delta: 'delta_ndwi', label: 'Water-Related Signal',  color: '#1A6EA8' },
  { key: 't1_ndbi', key2: 't2_ndbi', delta: 'delta_ndbi', label: 'Built-up Surface Signal', color: '#C07A2A' },
]

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)', padding: '10px 14px',
      boxShadow: 'var(--shadow-md)', fontSize: '0.875rem',
    }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: p.fill, flexShrink: 0 }} />
          <span style={{ color: 'var(--text-secondary)' }}>{p.name}:</span>
          <span style={{ fontWeight: 700 }}>{Number(p.value).toFixed(4)}</span>
        </div>
      ))}
    </div>
  )
}

export default function SignalsComparisonChart({ indexResult: r, mode = 'side' }) {
  if (!r) return null

  const barData = SIGNAL_META.map(m => ({
    signal: m.label,
    [r.year1]: r[m.key],
    [r.year2]: r[m.key2],
    delta:     r[m.delta],
    color:     m.color,
  }))

  const deltaData = SIGNAL_META.map(m => ({
    signal: m.label,
    Change: r[m.delta],
    color:  m.color,
  }))

  if (mode === 'slider') {
    // Delta view
    return (
      <div>
        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
          Change from {r.year1} to {r.year2}. Positive = increased. Negative = decreased.
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={deltaData} margin={{ top: 8, right: 8, left: -8, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="signal" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
            <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
            <ReferenceLine y={0} stroke="var(--text-muted)" />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="Change" radius={[3, 3, 0, 0]}>
              {deltaData.map((d, i) => (
                <Cell key={i} fill={d.Change < 0 ? '#C0392B' : '#40916C'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    )
  }

  // Side-by-side view
  return (
    <div>
      <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
        Comparing {r.year1} (left bar) and {r.year2} (right bar) for each signal.
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={barData} margin={{ top: 8, right: 8, left: -8, bottom: 24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="signal"
            tick={{ fill: 'var(--text-secondary)', fontSize: 10 }}
            interval={0}
          />
          <YAxis tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} domain={[-0.4, 0.6]} />
          <ReferenceLine y={0} stroke="var(--text-muted)" />
          <Tooltip content={<CustomTooltip />} />
          <Legend
            wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }}
          />
          <Bar dataKey={String(r.year1)} fill="#40916C" radius={[3, 3, 0, 0]} />
          <Bar dataKey={String(r.year2)} fill="#1A6EA8" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
