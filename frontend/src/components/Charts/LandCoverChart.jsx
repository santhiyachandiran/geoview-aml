import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts'

const CLASS_COLORS = {
  built_up:   '#FF3366',
  vegetation: '#10B981',
  water:      '#00E5FF',
  bare:       '#F59E0B',
}

export default function LandCoverChart({ indexResult }) {
  if (!indexResult) return <div style={{ color: 'var(--text-muted)', padding: '1rem', textAlign: 'center' }}>No data yet</div>

  const data = [
    {
      period: `${indexResult.year1}`,
      NDVI: indexResult.t1_ndvi?.toFixed(3),
      NDWI: indexResult.t1_ndwi?.toFixed(3),
      NDBI: indexResult.t1_ndbi?.toFixed(3),
    },
    {
      period: `${indexResult.year2}`,
      NDVI: indexResult.t2_ndvi?.toFixed(3),
      NDWI: indexResult.t2_ndwi?.toFixed(3),
      NDBI: indexResult.t2_ndbi?.toFixed(3),
    },
  ]

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="period" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
        <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="NDVI" fill="#10B981" />
        <Bar dataKey="NDWI" fill="#00E5FF" />
        <Bar dataKey="NDBI" fill="#FF3366" />
      </BarChart>
    </ResponsiveContainer>
  )
}
