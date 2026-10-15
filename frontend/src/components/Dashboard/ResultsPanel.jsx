import React from 'react'
import LandCoverChart from '../Charts/LandCoverChart'
import TransitionMatrix from '../Charts/TransitionMatrix'

export default function ResultsPanel({ indexResult, changeResult, error }) {
  if (error) {
    return <div className="card" style={{ color: 'var(--negative)' }}>Error: {error}</div>
  }
  if (!indexResult && !changeResult) return null

  const isDemo = indexResult?.demo_data || changeResult?.demo_data
  const flag = changeResult?.vegetation_to_buildup_flag

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {isDemo && (
        <div className="demo-banner">
          ⚠ DEMO DATA — Synthetic values generated for illustration. Not real satellite observations.
        </div>
      )}

      {indexResult && (
        <div className="card">
          <div className="card-title">Spectral Index Comparison</div>
          <LandCoverChart indexResult={indexResult} />
          <div style={{ marginTop: '1rem', display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <span className="pill pill--ok">ΔNDVI {indexResult.delta_ndvi?.toFixed(3)}</span>
            <span className="pill pill--ok">ΔNDWI {indexResult.delta_ndwi?.toFixed(3)}</span>
            <span className={`pill ${flag ? 'pill--error' : 'pill--ok'}`}>
              ΔNDBI {indexResult.delta_ndbi?.toFixed(3)}
            </span>
          </div>
        </div>
      )}

      {changeResult && (
        <div className="card">
          <div className="card-title">Change Detection</div>
          <div style={{ marginBottom: '0.75rem', display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <span className={`pill ${flag ? 'pill--error' : 'pill--ok'}`}>
              {flag ? '⚠ Possible Veg→Built-Up Transition' : '✓ No Transition Detected'}
            </span>
            <span className="pill pill--demo" style={{ fontSize: '0.7rem' }}>
              {changeResult.confidence_label}
            </span>
          </div>
          <TransitionMatrix changeResult={changeResult} />
          <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {changeResult.data_provenance?.notes}
          </div>
        </div>
      )}
    </div>
  )
}
