import React from 'react'

export default function StatsCard({ title, value, subtitle, trend, color = 'var(--primary)' }) {
  const trendColor = trend > 0 ? 'var(--negative)' : trend < 0 ? 'var(--positive)' : 'var(--text-muted)'
  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
      <span className="card-title">{title}</span>
      <span style={{ fontSize: '1.75rem', fontWeight: 700, color }}>{value ?? '—'}</span>
      {subtitle && <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{subtitle}</span>}
      {trend !== undefined && trend !== null && (
        <span style={{ fontSize: '0.8rem', color: trendColor }}>
          {trend > 0 ? '▲' : trend < 0 ? '▼' : '–'} {Math.abs(trend).toFixed(3)}
        </span>
      )}
    </div>
  )
}
