import React, { useEffect, useState } from 'react'
import ModelMetrics from '../components/Charts/ModelMetrics'
import { getModels, getExperiments } from '../services/api'

export default function ModelEval() {
  const [models, setModels] = useState(null)
  const [experiments, setExperiments] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getModels(), getExperiments()])
      .then(([m, e]) => { setModels(m.data); setExperiments(e.data) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="page-body" style={{ color: 'var(--text-muted)' }}>Loading…</div>

  // Try to extract RF and XGB metrics from experiment runs
  const rfRun = experiments?.runs?.find(r => r.run_name?.includes('rf') || r.params?.model === 'random_forest')
  const xgbRun = experiments?.runs?.find(r => r.run_name?.includes('xgb') || r.params?.model === 'xgboost')

  return (
    <div className="page-body">
      <h2 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: 700 }}>Model Evaluation</h2>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
        <div className="card">
          <div className="card-title">Model Comparison</div>
          <ModelMetrics rfMetrics={rfRun?.metrics} xgbMetrics={xgbRun?.metrics} />
        </div>
        <div className="card">
          <div className="card-title">Saved Models ({models?.count ?? 0})</div>
          {models?.models?.length === 0
            ? <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No models saved yet. Run training scripts.</p>
            : (models?.models || []).map(m => (
              <div key={m.name} style={{ padding: '0.4rem 0', borderBottom: '1px solid var(--border)', fontSize: '0.85rem' }}>
                <strong>{m.name}</strong> <span style={{ color: 'var(--text-muted)' }}>{m.size_kb} KB</span>
              </div>
            ))
          }
        </div>
        <div className="card" style={{ gridColumn: '1/-1' }}>
          <div className="card-title">Experiment Runs ({experiments?.count ?? 0})</div>
          {experiments?.runs?.length === 0
            ? <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No MLflow runs recorded yet.</p>
            : (experiments?.runs || []).map(r => (
              <div key={r.run_id} style={{ padding: '0.5rem 0', borderBottom: '1px solid var(--border)', fontSize: '0.8rem' }}>
                <strong>{r.run_name || r.run_id.slice(0, 8)}</strong>
                <span style={{ color: 'var(--text-muted)', marginLeft: '0.75rem' }}>{r.status}</span>
                {Object.entries(r.metrics || {}).slice(0, 4).map(([k, v]) => (
                  <span key={k} style={{ marginLeft: '0.75rem', color: 'var(--primary)' }}>{k}: {v?.toFixed(4)}</span>
                ))}
              </div>
            ))
          }
        </div>
      </div>
    </div>
  )
}
