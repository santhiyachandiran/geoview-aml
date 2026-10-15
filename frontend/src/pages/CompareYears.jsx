import React, { useState, useRef, useCallback, useEffect } from 'react'
import { useAnalysis } from '../hooks/useAnalysis'
import { getStudyAreas } from '../services/api'
import AnalysisControls from '../components/Dashboard/AnalysisControls'
import SignalsComparisonChart from '../components/Charts/SignalsComparisonChart'
import LandCoverAreaChart from '../components/Charts/LandCoverAreaChart'
import TransitionMatrix from '../components/Charts/TransitionMatrix'
import BeforeAfterMap from '../components/Map/BeforeAfterMap'

export default function CompareYears() {
  const [studyAreas, setStudyAreas] = useState([])
  const [viewMode, setViewMode]     = useState('side')  // 'side' | 'slider'
  const { loading, error, indexResult, changeResult, runAnalysis, reset } = useAnalysis()

  useEffect(() => {
    getStudyAreas().then(r => setStudyAreas(r.data)).catch(() => {})
  }, [])

  const isDemo = indexResult?.demo_data || changeResult?.demo_data

  return (
    <main className="page-body">
      <div className="section-header">
        <div>
          <h1 className="section-title">Compare Years</h1>
          <p className="section-desc">
            Run change analysis between two dry seasons and compare satellite imagery, spectral signals,
            and classifier land-cover distributions side by side.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className={`btn btn--sm ${viewMode === 'side' ? 'btn--primary' : 'btn--ghost'}`}
            onClick={() => setViewMode('side')}
          >⊞ Side by side</button>
          <button
            className={`btn btn--sm ${viewMode === 'slider' ? 'btn--primary' : 'btn--ghost'}`}
            onClick={() => setViewMode('slider')}
          >⇔ Delta change view</button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px,320px) 1fr', gap: '1.5rem', alignItems: 'start' }}>
        <AnalysisControls
          studyAreas={studyAreas}
          onRun={runAnalysis}
          onReset={reset}
          loading={loading}
        />

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Demo note */}
          {isDemo && (
            <div className="alert alert--demo">
              <span className="alert__icon">📋</span>
              <div>
                <div className="alert__title">Demo mode active</div>
                These charts show synthetic illustrative values because Google Earth Engine credentials are not active.
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="alert alert--warning">
              <span className="alert__icon">⚠️</span>
              <div><div className="alert__title">Error</div>{error}</div>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="card">
              <div className="loading-overlay">
                <div className="spinner" />
                <span>Retrieving Sentinel-2 composite data and running change detection…</span>
              </div>
            </div>
          )}

          {/* Visualizations */}
          {indexResult && !loading && (
            <>
              {/* 1. Before & After Satellite Imagery */}
              <div className="card">
                <div className="card-label">1. Before & After Satellite Imagery Comparison</div>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  Direct visual comparison of {indexResult.study_area.toUpperCase()} across both observation years at 10m spatial resolution.
                </p>
                <BeforeAfterMap
                  year1={indexResult.year1}
                  year2={indexResult.year2}
                  studyArea={indexResult.study_area}
                  changeResult={changeResult}
                />
              </div>

              {/* 2. Land-Cover Area by Class & Net Change */}
              {changeResult?.area_stats && (
                <div className="card">
                  <div className="card-label">2. Land-Cover Area by Class & Net Change (Hectares / km²)</div>
                  <LandCoverAreaChart areaStats={changeResult.area_stats} />
                </div>
              )}

              {/* 3. Transition Matrix */}
              {changeResult?.transition_matrix && (
                <div className="card">
                  <div className="card-label">3. Land-Cover Transition Matrix (Class Transitions)</div>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                    Number of 10m pixels transitioning from each initial class (rows) to final class (columns).
                  </p>
                  <TransitionMatrix changeResult={changeResult} />
                  <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {changeResult.data_provenance?.notes}
                  </div>
                </div>
              )}

              {/* 4. Spectral Index Signals */}
              <div className="card">
                <div className="card-label">4. Spectral Signal Changes (NDVI, NDWI, NDBI)</div>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  <strong>NDVI</strong> measures vegetation greenness; <strong>NDWI</strong> measures water content; <strong>NDBI</strong> measures built-up and paved surfaces.
                </p>
                <SignalsComparisonChart indexResult={indexResult} mode={viewMode} />
              </div>

              {/* 5. Numerical Signals Table */}
              <div className="card">
                <div className="card-label">5. Mean Spectral Index Observations — {indexResult.year1} vs {indexResult.year2}</div>
                <AllSignalsTable indexResult={indexResult} />
              </div>

              {/* 6. Change Summary */}
              {changeResult && (
                <div className="card">
                  <div className="card-label">6. Change Detection Summary & Classification Notes</div>
                  <ChangeSummary changeResult={changeResult} />
                </div>
              )}
            </>
          )}

          {/* Empty state */}
          {!indexResult && !loading && !error && (
            <div className="card">
              <div className="empty-state">
                <div className="empty-state__icon">⇔</div>
                <div className="empty-state__title">Select two years and run analysis</div>
                <div className="empty-state__body">
                  Charts comparing satellite signals for the two selected years will
                  appear here.
                </div>
              </div>
            </div>
          )}

          <MapViewLink />
        </div>
      </div>
    </main>
  )
}

function AllSignalsTable({ indexResult: r }) {
  const rows = [
    {
      label: 'Vegetation Greenness',
      tech: 'NDVI',
      t1: r.t1_ndvi,
      t2: r.t2_ndvi,
      delta: r.delta_ndvi,
      positive_if: 'increase',
    },
    {
      label: 'Water-Related Signal',
      tech: 'NDWI',
      t1: r.t1_ndwi,
      t2: r.t2_ndwi,
      delta: r.delta_ndwi,
      positive_if: 'increase',
    },
    {
      label: 'Built-up Surface Signal',
      tech: 'NDBI',
      t1: r.t1_ndbi,
      t2: r.t2_ndbi,
      delta: r.delta_ndbi,
      positive_if: 'decrease',
    },
  ]

  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
      <thead>
        <tr style={{ borderBottom: '2px solid var(--border)' }}>
          <th style={{ textAlign: 'left',   padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.8rem' }}>Signal</th>
          <th style={{ textAlign: 'center', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.8rem' }}>{r.year1}</th>
          <th style={{ textAlign: 'center', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.8rem' }}>{r.year2}</th>
          <th style={{ textAlign: 'center', padding: '8px 12px', color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.8rem' }}>Change</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(row => {
          const d = row.delta
          const isGood = d !== null && ((row.positive_if === 'increase' && d > 0) || (row.positive_if === 'decrease' && d < 0))
          const isBad  = d !== null && ((row.positive_if === 'increase' && d < -0.04) || (row.positive_if === 'decrease' && d > 0.04))
          const color = isBad ? 'var(--loss)' : isGood ? 'var(--vegetation)' : 'var(--text)'
          return (
            <tr key={row.tech} style={{ borderBottom: '1px solid var(--border)' }}>
              <td style={{ padding: '10px 12px' }}>
                <div style={{ fontWeight: 600 }}>{row.label}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{row.tech}</div>
              </td>
              <td style={{ textAlign: 'center', padding: '10px 12px', fontFamily: 'var(--font-mono)' }}>
                {row.t1?.toFixed(4) ?? '—'}
              </td>
              <td style={{ textAlign: 'center', padding: '10px 12px', fontFamily: 'var(--font-mono)' }}>
                {row.t2?.toFixed(4) ?? '—'}
              </td>
              <td style={{ textAlign: 'center', padding: '10px 12px', fontWeight: 700, color }}>
                {d !== null && d !== undefined ? `${d > 0 ? '+' : ''}${d.toFixed(4)}` : '—'}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function ChangeSummary({ changeResult: c }) {
  const flag = c.vegetation_to_buildup_flag
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
      <div style={{
        width: 48, height: 48, borderRadius: 'var(--radius-md)',
        background: flag ? 'var(--loss-bg)' : 'var(--vegetation-bg)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '1.5rem', flexShrink: 0,
      }}>
        {flag ? '⚠️' : '✅'}
      </div>
      <div>
        <div style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--text)' }}>
          {flag
            ? 'Possible vegetation-to-built-up transition detected'
            : 'No clear vegetation-to-built-up transition'}
        </div>
        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
          Confidence level: <strong>{c.confidence_label?.replace(/_/g, ' ')}</strong>
          {c.demo_data && <span className="badge badge--demo" style={{ marginLeft: 8 }}>Demo</span>}
        </div>
        <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          {c.data_provenance?.notes}
        </div>
      </div>
    </div>
  )
}

function MapViewLink() {
  return (
    <div className="card card--compact" style={{ background: 'var(--brand-surface)', border: '1px solid var(--brand-lighter)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontWeight: 700, color: 'var(--brand)' }}>View on the map</div>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            Explore the study area interactively with layer overlays.
          </div>
        </div>
        <a href="#explore" className="btn btn--secondary btn--sm">Open Map →</a>
      </div>
    </div>
  )
}
