import React, { useState } from 'react'

const CURRENT_YEAR = new Date().getFullYear()
const YEARS = Array.from({ length: CURRENT_YEAR - 2016 }, (_, i) => 2017 + i)

export default function AnalysisControls({ studyAreas, onRun, onReset, loading }) {
  const [studyArea, setStudyArea]   = useState('coimbatore')
  const [year1, setYear1]           = useState(2018)
  const [year2, setYear2]           = useState(2024)
  const [expanded, setExpanded]     = useState(false)
  const [seasonStart, setSeasonStart] = useState(11)
  const [seasonEnd, setSeasonEnd]   = useState(3)

  const areaList = studyAreas?.length
    ? studyAreas
    : [{ slug: 'coimbatore', name: 'Coimbatore, Tamil Nadu' }]

  const valid = year1 < year2

  function handleSubmit(e) {
    e.preventDefault()
    if (!valid) return
    onRun({ studyArea, year1, year2, seasonStart, seasonEnd })
  }

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div>
        <div className="card-label">Analysis settings</div>
        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
          Choose a location and two years to compare.
        </div>
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {/* Study area */}
        <div className="form-group">
          <label className="form-label">Study Area</label>
          <select
            className="form-control"
            value={studyArea}
            onChange={e => setStudyArea(e.target.value)}
            disabled={loading}
          >
            {areaList.map(a => (
              <option key={a.slug} value={a.slug}>{a.name}</option>
            ))}
          </select>
        </div>

        {/* Years */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
          <div className="form-group">
            <label className="form-label">Earlier Year</label>
            <select
              className="form-control"
              value={year1}
              onChange={e => setYear1(Number(e.target.value))}
              disabled={loading}
            >
              {YEARS.map(y => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Later Year</label>
            <select
              className="form-control"
              value={year2}
              onChange={e => setYear2(Number(e.target.value))}
              disabled={loading}
            >
              {YEARS.map(y => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
        </div>

        {!valid && (
          <div className="alert alert--warning" style={{ padding: '0.5rem 0.75rem' }}>
            <span>⚠️</span>
            <span style={{ fontSize: '0.8125rem' }}>Earlier year must be before later year.</span>
          </div>
        )}

        {/* Advanced options */}
        <details onToggle={e => setExpanded(e.target.open)}>
          <summary className="tech-details__summary" style={{ border: 'none', background: 'none', padding: '0' }}>
            {expanded ? '▾' : '▸'} Season filter (advanced)
          </summary>
          <div style={{ paddingTop: '0.75rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div className="form-group">
              <label className="form-label">Season start month</label>
              <select className="form-control" value={seasonStart}
                onChange={e => setSeasonStart(Number(e.target.value))} disabled={loading}>
                {[10,11,12,1,2,3].map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Season end month</label>
              <select className="form-control" value={seasonEnd}
                onChange={e => setSeasonEnd(Number(e.target.value))} disabled={loading}>
                {[1,2,3,4,5].map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div style={{ gridColumn: '1/-1', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              November–February avoids monsoon cloud cover over Tamil Nadu.
            </div>
          </div>
        </details>

        {/* Actions */}
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            type="submit"
            className="btn btn--primary"
            style={{ flex: 1 }}
            disabled={loading || !valid}
          >
            {loading
              ? <><span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> Analysing…</>
              : '🔍 Analyse Changes'
            }
          </button>
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={onReset}
            disabled={loading}
            title="Clear results"
          >
            ↺
          </button>
        </div>
      </form>
    </div>
  )
}
