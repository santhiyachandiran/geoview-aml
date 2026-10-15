import React, { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, CartesianGrid,
} from 'recharts'
import { getModels, getExperiments, getModelPlots } from '../services/api'

const METRIC_LABELS = {
  accuracy:          'Overall Accuracy',
  macro_f1:          'Balanced F1 Score',
  balanced_accuracy: 'Balanced Accuracy',
  mean_iou:          'Mean Overlap (IoU)',
}

const METRIC_DESC = {
  accuracy:          'Fraction of pixels classified correctly.',
  macro_f1:          'Average precision/recall balance across all classes. Preferred for imbalanced data.',
  balanced_accuracy: 'Accuracy adjusted for class imbalance.',
  mean_iou:          'Average overlap between predicted and true class areas.',
}

function MetricBar({ label, value, max = 1 }) {
  const pct = (value / max) * 100
  const color = value >= 0.8 ? 'var(--vegetation)' : value >= 0.65 ? 'var(--buildup)' : 'var(--loss)'
  return (
    <div style={{ marginBottom: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: '0.875rem' }}>
        <span style={{ color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontWeight: 700, color }}>{value != null ? value.toFixed(4) : '—'}</span>
      </div>
      <div style={{ height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 4, transition: 'width 0.6s' }} />
      </div>
    </div>
  )
}

function NoModels() {
  return (
    <div className="empty-state">
      <div className="empty-state__icon">🤖</div>
      <div className="empty-state__title">No trained models yet</div>
      <div className="empty-state__body">
        Models will appear here after training. Run the training scripts once
        real satellite data and validated labels are available.
      </div>
      <div style={{ marginTop: '1rem' }}>
        <details className="tech-details" style={{ maxWidth: 480 }}>
          <summary className="tech-details__summary">How to train models</summary>
          <div className="tech-details__body">
            <ol style={{ paddingLeft: '1.25rem', lineHeight: 1.8 }}>
              <li>Authenticate Google Earth Engine: <code>earthengine authenticate</code></li>
              <li>Download Sentinel-2 composites for the study area</li>
              <li>Extract ESA WorldCover 2021 labels</li>
              <li>Run <code>python ml/models/train_rf.py</code></li>
              <li>Run <code>python ml/models/train_xgboost.py</code></li>
              <li>Results appear here via MLflow tracking</li>
            </ol>
          </div>
        </details>
      </div>
    </div>
  )
}

export default function ModelPerformance() {
  const [models,      setModels]      = useState(null)
  const [experiments, setExperiments] = useState(null)
  const [plots,       setPlots]       = useState([])
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState(null)

  useEffect(() => {
    Promise.all([
      getModels(),
      getExperiments(),
      getModelPlots().catch(() => ({ data: { plots: [] } })),
    ])
      .then(([m, e, p]) => {
        setModels(m.data)
        setExperiments(e.data)
        setPlots(p.data?.plots || [])
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const rfRun  = experiments?.runs?.find(r => r.run_name?.toLowerCase().includes('rf')  || r.params?.model === 'random_forest')
  const xgbRun = experiments?.runs?.find(r => r.run_name?.toLowerCase().includes('xgb') || r.params?.model === 'xgboost')

  const chartData = Object.keys(METRIC_LABELS).map(key => ({
    metric: METRIC_LABELS[key],
    'Random Forest': rfRun?.metrics?.[key] ?? null,
    XGBoost:         xgbRun?.metrics?.[key] ?? null,
  })).filter(d => d['Random Forest'] !== null || d.XGBoost !== null)

  return (
    <main className="page-body">
      <div className="section-header">
        <div>
          <h1 className="section-title">Model Performance</h1>
          <p className="section-desc">
            Comparison of the Random Forest baseline and XGBoost advanced model.
            Metrics are reported only from real training runs — never fabricated.
          </p>
        </div>
      </div>

      {loading && (
        <div className="loading-overlay"><div className="spinner" /><span>Loading experiment data…</span></div>
      )}

      {error && (
        <div className="alert alert--warning mb-6">
          <span className="alert__icon">⚠️</span>
          <div><div className="alert__title">Could not load experiments</div>{error}</div>
        </div>
      )}

      {!loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Model files */}
          <div className="grid-2">
            <div className="card">
              <div className="card-label">Saved models ({models?.count ?? 0})</div>
              {!models?.models?.length
                ? <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>No model files in artifacts/models/</div>
                : models.models.map(m => (
                  <div key={m.name} style={{ padding: '0.5rem 0', borderBottom: '1px solid var(--border)', fontSize: '0.875rem' }}>
                    <div style={{ fontWeight: 600 }}>{m.name}</div>
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>{m.size_kb} KB · {m.model_type}</div>
                  </div>
                ))
              }
            </div>

            <div className="card">
              <div className="card-label">About the models</div>
              <div className="prose" style={{ fontSize: '0.875rem' }}>
                <p><strong>Random Forest</strong> — baseline classifier. Trains on spectral band values and vegetation/built-up indices. Fast and interpretable.</p>
                <p><strong>XGBoost</strong> — advanced gradient-boosted classifier tuned with Optuna. Generally achieves better balance across land-cover classes.</p>
                <p>Both models use a spatial block split to avoid training on pixels adjacent to test pixels, which would inflate accuracy scores.</p>
              </div>
            </div>
          </div>

          {/* Metric comparison chart */}
          {chartData.length > 0 ? (
            <div className="card">
              <div className="card-label">Metric comparison — Random Forest vs XGBoost</div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                All metrics are on a 0–1 scale. Higher is better for accuracy, F1 and overlap.
                Results come from the held-out spatial test set.
              </p>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartData} margin={{ top: 8, right: 8, left: -8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="metric" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                  <YAxis domain={[0, 1]} tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius-md)' }}
                    formatter={v => v?.toFixed(4)}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Random Forest" fill="#40916C" radius={[3,3,0,0]} />
                  <Bar dataKey="XGBoost"       fill="#1A6EA8" radius={[3,3,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="card"><NoModels /></div>
          )}

          {/* Detailed metrics */}
          {(rfRun || xgbRun) && (
            <div className="grid-2">
              {rfRun && (
                <div className="card">
                  <div className="card-label">Random Forest</div>
                  {Object.entries(METRIC_LABELS).map(([key, label]) =>
                    rfRun.metrics?.[key] != null ? (
                      <MetricBar key={key} label={label} value={rfRun.metrics[key]} />
                    ) : null
                  )}
                </div>
              )}
              {xgbRun && (
                <div className="card">
                  <div className="card-label">XGBoost</div>
                  {Object.entries(METRIC_LABELS).map(([key, label]) =>
                    xgbRun.metrics?.[key] != null ? (
                      <MetricBar key={key} label={label} value={xgbRun.metrics[key]} />
                    ) : null
                  )}
                </div>
              )}
            </div>
          )}

          {/* MLflow runs */}
          <div className="card">
            <div className="card-label">
              Experiment runs ({experiments?.count ?? 0})
              {experiments?.tracking_uri && (
                <span style={{ marginLeft: 8, fontWeight: 400, textTransform: 'none', color: 'var(--text-muted)' }}>
                  tracking: {experiments.tracking_uri}
                </span>
              )}
            </div>
            {!experiments?.runs?.length
              ? <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>No MLflow runs recorded yet. Run training scripts to log experiments.</div>
              : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '2px solid var(--border)' }}>
                        {['Run', 'Status', 'Accuracy', 'F1 (macro)', 'Started'].map(h => (
                          <th key={h} style={{ textAlign: 'left', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.8rem' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {experiments.runs.map(r => (
                        <tr key={r.run_id} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td style={{ padding: '8px 12px', fontWeight: 600 }}>{r.run_name || r.run_id.slice(0,8)}</td>
                          <td style={{ padding: '8px 12px' }}>
                            <span className={`badge ${r.status === 'FINISHED' ? 'badge--success' : 'badge--info'}`}>
                              {r.status}
                            </span>
                          </td>
                          <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
                            {r.metrics?.accuracy?.toFixed(4) ?? '—'}
                          </td>
                          <td style={{ padding: '8px 12px', fontFamily: 'var(--font-mono)' }}>
                            {r.metrics?.macro_f1?.toFixed(4) ?? '—'}
                          </td>
                          <td style={{ padding: '8px 12px', color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                            {r.start_time ? new Date(Number(r.start_time)).toLocaleDateString() : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )
            }
          </div>

          {/* Saved Visualizations & Plots */}
          {plots.length > 0 && (
            <div className="card">
              <div className="card-label">Saved Visualizations & Analytical Plots ({plots.length})</div>
              <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                High-resolution evaluation plots generated from real Sentinel-2 satellite data and held-out test evaluation.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
                {plots.map(p => (
                  <div key={p.filename} style={{ border: '1px solid var(--border)', borderRadius: 'var(--radius-md)', padding: '0.75rem', background: 'var(--surface)' }}>
                    <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.5rem', wordBreak: 'break-all' }}>
                      {p.filename.replace(/_\d{8}T\d{6}\.png$/, '').replace(/_/g, ' ').toUpperCase()}
                    </div>
                    <img
                      src={p.url}
                      alt={p.filename}
                      style={{ width: '100%', height: 'auto', borderRadius: '4px', border: '1px solid var(--border)' }}
                      loading="lazy"
                    />
                    <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                      <span>{p.size_kb} KB</span>
                      <a href={p.url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)' }}>Full Resolution ↗</a>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </main>
  )
}
