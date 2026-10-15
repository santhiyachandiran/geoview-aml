import React from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid, ReferenceLine } from 'recharts'

export default function IndexTimeline({ availability }) {
  if (!availability?.years?.length) return null

  const data = availability.years.map(y => ({
    year: String(y.year),
    scenes: y.scene_count || 0,
    cloud: y.cloud_cover_pct || 0,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="year" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
        <Tooltip contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="scenes" stroke="#10B981" strokeWidth={2} dot={{ r: 3 }} name="Scenes" />
        <Line type="monotone" dataKey="cloud" stroke="#FF3366" strokeWidth={2} dot={{ r: 3 }} name="Cloud %" />
      </LineChart>
    </ResponsiveContainer>
  )
}
