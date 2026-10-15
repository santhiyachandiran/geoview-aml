import React, { useState } from 'react'

export default function AnalysisPanel({ studyAreas, onRun, loading }) {
  const [form, setForm] = useState({
    studyArea: 'coimbatore',
    year1: 2018,
    year2: 2024,
    seasonStart: 11,
    seasonEnd: 3,
  })

  const handleChange = (k, v) => setForm(f => ({ ...f, [k]: Number(v) || v }))

  const handleSubmit = (e) => {
    e.preventDefault()
    onRun(form)
  }

  return (
    <div className="card">
      <div className="card-title">Run Analysis</div>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div className="form-group">
          <label className="form-label">Study Area</label>
          <select className="form-control" value={form.studyArea}
            onChange={e => handleChange('studyArea', e.target.value)}>
            {(studyAreas || [{ slug: 'coimbatore', name: 'Coimbatore' }]).map(a => (
              <option key={a.slug} value={a.slug}>{a.name}</option>
            ))}
          </select>
        </div>
        <div className="grid-2">
          <div className="form-group">
            <label className="form-label">Year 1 (Before)</label>
            <input className="form-control" type="number" min={2013} max={2030}
              value={form.year1} onChange={e => handleChange('year1', e.target.value)} />
          </div>
          <div className="form-group">
            <label className="form-label">Year 2 (After)</label>
            <input className="form-control" type="number" min={2013} max={2030}
              value={form.year2} onChange={e => handleChange('year2', e.target.value)} />
          </div>
        </div>
        <button className="btn btn--primary" type="submit" disabled={loading}>
          {loading ? 'Analysing…' : '▶ Run Change Detection'}
        </button>
      </form>
    </div>
  )
}
