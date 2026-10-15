import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid, ReferenceLine, Cell
} from 'recharts'

const CLASS_CONFIG = [
  { key: 'built_up',   label: 'Built-up / Urban',  color: '#C07A2A' },
  { key: 'vegetation', label: 'Vegetation / Forest', color: '#40916C' },
  { key: 'water',      label: 'Water Bodies',       color: '#1A6EA8' },
  { key: 'bare',       label: 'Bare Soil / Cleared', color: '#8B7355' },
]

export default function LandCoverAreaChart({ areaStats }) {
  const [chartMode, setChartMode] = useState('area') // 'area' | 'net_change'

  if (!areaStats?.t1 || !areaStats?.t2) return null

  const { year1, year2, t1, t2, net_change, total_area_ha, total_area_km2 } = areaStats

  const areaData = CLASS_CONFIG.map(c => ({
    name: c.label,
    classKey: c.key,
    [String(year1)]: t1[c.key]?.area_ha ?? 0,
    [String(year2)]: t2[c.key]?.area_ha ?? 0,
    delta_ha: net_change[c.key]?.delta_ha ?? 0,
    delta_pct: net_change[c.key]?.delta_pct ?? 0,
    color: c.color,
  }))

  const netData = CLASS_CONFIG.map(c => ({
    name: c.label,
    classKey: c.key,
    NetChangeHa: net_change[c.key]?.delta_ha ?? 0,
    NetChangeKm2: net_change[c.key]?.delta_km2 ?? 0,
    NetChangePct: net_change[c.key]?.delta_pct ?? 0,
    color: c.color,
  }))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          Sampled Area: <strong>{total_area_ha?.toLocaleString()} ha</strong> ({total_area_km2} km²) • Resolution: 10m
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className={`btn btn--sm ${chartMode === 'area' ? 'btn--primary' : 'btn--ghost'}`}
            onClick={() => setChartMode('area')}
          >
            Total Area (ha)
          </button>
          <button
            className={`btn btn--sm ${chartMode === 'net_change' ? 'btn--primary' : 'btn--ghost'}`}
            onClick={() => setChartMode('net_change')}
          >
            Net Gain / Loss (ha)
          </button>
        </div>
      </div>

      {chartMode === 'area' ? (
        <div>
          <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Comparing total class surface area in <strong>hectares (ha)</strong> between {year1} (green) and {year2} (blue).
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={areaData} margin={{ top: 8, right: 12, left: 4, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
              <YAxis
                tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                unit=" ha"
              />
              <Tooltip
                formatter={(val, name) => [`${Number(val).toLocaleString()} ha`, `${name}`]}
                contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.85rem' }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey={String(year1)} fill="#40916C" radius={[3, 3, 0, 0]} name={`${year1} (ha)`} />
              <Bar dataKey={String(year2)} fill="#1A6EA8" radius={[3, 3, 0, 0]} name={`${year2} (ha)`} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div>
          <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Net area gained (positive) or lost (negative) in <strong>hectares (ha)</strong> from {year1} to {year2}.
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={netData} margin={{ top: 8, right: 12, left: 4, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
              <YAxis
                tick={{ fill: 'var(--text-secondary)', fontSize: 11 }}
                unit=" ha"
              />
              <ReferenceLine y={0} stroke="var(--text-muted)" />
              <Tooltip
                formatter={(val) => [`${val > 0 ? '+' : ''}${val} ha`, 'Net Change']}
                contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6, fontSize: '0.85rem' }}
              />
              <Bar dataKey="NetChangeHa" radius={[3, 3, 0, 0]}>
                {netData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.NetChangeHa >= 0 ? '#40916C' : '#C0392B'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Numerical Data Table with unambiguous units */}
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', marginTop: '0.5rem' }}>
        <thead>
          <tr style={{ borderBottom: '2px solid var(--border)' }}>
            <th style={{ textAlign: 'left', padding: '6px 8px', color: 'var(--text-muted)' }}>Class</th>
            <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-muted)' }}>{year1} (ha / km²)</th>
            <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-muted)' }}>{year2} (ha / km²)</th>
            <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-muted)' }}>Net Change (ha)</th>
            <th style={{ textAlign: 'right', padding: '6px 8px', color: 'var(--text-muted)' }}>Net Change (%)</th>
          </tr>
        </thead>
        <tbody>
          {CLASS_CONFIG.map(c => {
            const rowT1 = t1[c.key] || {}
            const rowT2 = t2[c.key] || {}
            const net = net_change[c.key] || {}
            const deltaHa = net.delta_ha ?? 0
            const deltaPct = net.delta_pct ?? 0
            const isPos = deltaHa > 0
            const isNeg = deltaHa < 0
            const changeColor = isPos ? 'var(--vegetation)' : isNeg ? 'var(--loss)' : 'var(--text-muted)'
            return (
              <tr key={c.key} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={{ padding: '6px 8px', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: c.color, flexShrink: 0 }} />
                  <strong>{c.label}</strong>
                </td>
                <td style={{ textAlign: 'right', padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>
                  {rowT1.area_ha?.toFixed(1) ?? '—'} ha ({rowT1.area_km2?.toFixed(3) ?? '—'} km²)
                </td>
                <td style={{ textAlign: 'right', padding: '6px 8px', fontFamily: 'var(--font-mono)' }}>
                  {rowT2.area_ha?.toFixed(1) ?? '—'} ha ({rowT2.area_km2?.toFixed(3) ?? '—'} km²)
                </td>
                <td style={{ textAlign: 'right', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontWeight: 600, color: changeColor }}>
                  {deltaHa > 0 ? `+${deltaHa.toFixed(2)}` : deltaHa.toFixed(2)} ha
                </td>
                <td style={{ textAlign: 'right', padding: '6px 8px', fontFamily: 'var(--font-mono)', fontWeight: 600, color: changeColor }}>
                  {deltaPct > 0 ? `+${deltaPct.toFixed(2)}` : deltaPct.toFixed(2)}%
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
