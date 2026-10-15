import React from 'react'

/*
  Translates internal NDVI/NDWI/NDBI into plain English for the user.
  A "Technical details" accordion shows the actual index names and formulas.
*/

function getDeltaLabel(val) {
  if (val === null || val === undefined) return null
  const abs = Math.abs(val)
  const dir = val > 0 ? 'increase' : 'decrease'
  const size = abs > 0.1 ? 'notable' : abs > 0.04 ? 'moderate' : 'small'
  return { dir, size, abs: abs.toFixed(3) }
}

function SignalCard({ icon, label, techName, formula, delta, color, bgColor, loading, note }) {
  const info = getDeltaLabel(delta)

  let deltaText = '—'
  let deltaColor = 'var(--text-muted)'

  if (loading) {
    return (
      <div className="stat-card">
        <div className="stat-card__icon" style={{ background: bgColor }}>{icon}</div>
        <div className="stat-card__label">{label}</div>
        <div className="skeleton" style={{ height: 32, width: '60%', borderRadius: 4 }} />
      </div>
    )
  }

  if (info) {
    const isPositive = (label.includes('Greenness') && delta > 0)
                    || (label.includes('Water') && delta > 0)
                    || (label.includes('Built-up') && delta < 0)
    const isNegative = (label.includes('Greenness') && delta < 0)
                    || (label.includes('Water') && delta < 0)
                    || (label.includes('Built-up') && delta > 0)
    deltaColor  = isNegative ? 'var(--loss)' : isPositive ? 'var(--vegetation)' : 'var(--text-muted)'
    const arrow = delta > 0 ? '▲' : '▼'
    deltaText = `${arrow} ${info.size} ${info.dir} (${info.abs})`
  }

  return (
    <div className="stat-card">
      <div className="stat-card__icon" style={{ background: bgColor }}>{icon}</div>
      <div className="stat-card__label">{label}</div>
      <div
        className={delta !== null && delta !== undefined ? 'stat-card__value' : 'stat-card__value--unavailable'}
        style={{ fontSize: delta !== null && delta !== undefined ? '1.5rem' : '1rem', color }}
      >
        {delta !== null && delta !== undefined ? delta.toFixed(3) : 'Run analysis'}
      </div>
      {info && (
        <div className="stat-card__delta" style={{ color: deltaColor, background: 'transparent', padding: 0 }}>
          {deltaText}
        </div>
      )}
      {note && (
        <div className="stat-card__sub" style={{ fontSize: '0.75rem', fontStyle: 'italic' }}>{note}</div>
      )}
      <details className="tech-details" style={{ marginTop: '0.5rem' }}>
        <summary className="tech-details__summary">Technical details</summary>
        <div className="tech-details__body">
          <strong>Index:</strong> {techName}<br />
          <strong>Formula:</strong> <code>{formula}</code>
        </div>
      </details>
    </div>
  )
}

export default function SignalSummaryCards({ indexResult, changeResult, loading }) {
  const r = indexResult

  return (
    <div className="grid-3">
      <SignalCard
        icon="🌿"
        label="Vegetation Greenness"
        techName="NDVI — Normalised Difference Vegetation Index"
        formula="(NIR − Red) / (NIR + Red)"
        delta={r?.delta_ndvi ?? null}
        color="var(--vegetation)"
        bgColor="var(--vegetation-bg)"
        loading={loading}
        note="Higher = more green vegetation"
      />
      <SignalCard
        icon="💧"
        label="Water-Related Signal"
        techName="NDWI — Normalised Difference Water Index"
        formula="(Green − SWIR1) / (Green + SWIR1)"
        delta={r?.delta_ndwi ?? null}
        color="var(--water)"
        bgColor="var(--water-bg)"
        loading={loading}
        note="Higher = wetter surfaces"
      />
      <SignalCard
        icon="🏗️"
        label="Built-up Surface Signal"
        techName="NDBI — Normalised Difference Built-up Index"
        formula="(SWIR1 − NIR) / (SWIR1 + NIR)"
        delta={r?.delta_ndbi ?? null}
        color="var(--buildup)"
        bgColor="var(--buildup-bg)"
        loading={loading}
        note="Higher = more paved/built surface"
      />
    </div>
  )
}
