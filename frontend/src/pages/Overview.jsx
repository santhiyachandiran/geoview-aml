import React, { useEffect, useState } from 'react'
import { getStudyAreas, getImageryAvailability } from '../services/api'
import { useAnalysis } from '../hooks/useAnalysis'
import AnalysisControls from '../components/Dashboard/AnalysisControls'
import SignalSummaryCards from '../components/Dashboard/SignalSummaryCards'
import InsightCard from '../components/Dashboard/InsightCard'

export default function Overview({ apiStatus }) {
  const [studyAreas, setStudyAreas]   = useState([])
  const [availability, setAvailability] = useState(null)
  const { loading, error, indexResult, changeResult, runAnalysis, reset } = useAnalysis()

  const isDemo = indexResult?.demo_data || changeResult?.demo_data
  const geeOk  = apiStatus?.components?.find(c => c.name === 'google_earth_engine')?.status === 'ok'

  useEffect(() => {
    getStudyAreas().then(r => setStudyAreas(r.data)).catch(() => {})
    getImageryAvailability('coimbatore', 2017, 2025)
      .then(r => setAvailability(r.data)).catch(() => {})
  }, [])

  return (
    <main className="page-body">
      {/* ── Hero ─────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero__eyebrow">🛰 Satellite Land-Cover Change Detection</div>
        <h1 className="hero__title">See how Coimbatore is changing.</h1>
        <p className="hero__body">
          Compare satellite observations from different years to understand where
          vegetation has declined, where built-up areas are expanding, and how
          water bodies and bare land are shifting across the Coimbatore region.
        </p>

        <div className="hero__actions">
          <a href="#compare" className="hero__btn-primary">
            ⇔ Compare Two Years
          </a>
          <a href="#explore" className="hero__btn-ghost">
            🗺️ Explore the Map
          </a>
        </div>

        <div className="hero__meta">
          <div className="hero__meta-item">
            <span className="hero__meta-value">Coimbatore</span>
            <span className="hero__meta-label">Study area, Tamil Nadu</span>
          </div>
          <div className="hero__meta-item">
            <span className="hero__meta-value">~400 km²</span>
            <span className="hero__meta-label">Area analysed</span>
          </div>
          <div className="hero__meta-item">
            <span className="hero__meta-value">10 m</span>
            <span className="hero__meta-label">Sentinel-2 resolution</span>
          </div>
          <div className="hero__meta-item">
            <span className="hero__meta-value">2017–2025</span>
            <span className="hero__meta-label">Satellite archive</span>
          </div>
          {!geeOk && (
            <div className="hero__meta-item">
              <span className="hero__meta-value" style={{ fontSize: '1rem' }}>📋 Demo</span>
              <span className="hero__meta-label">Illustrative data</span>
            </div>
          )}
        </div>
      </section>

      {/* ── Demo banner ─────────────────────────────────── */}
      {!geeOk && (
        <div className="alert alert--demo mb-6">
          <span className="alert__icon">📋</span>
          <div>
            <div className="alert__title">Demo mode — illustrative data</div>
            Google Earth Engine is not connected. The numbers and charts below are
            synthetic examples that show how the application works. They are not real
            satellite measurements. Configure <code>GEE_PROJECT_ID</code> in your
            <code>.env</code> and run <code>earthengine authenticate</code> to enable
            real imagery.
          </div>
        </div>
      )}

      {/* ── Quick analysis ───────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px,340px) 1fr', gap: '1.5rem', alignItems: 'start' }}>
        <AnalysisControls
          studyAreas={studyAreas}
          onRun={runAnalysis}
          onReset={reset}
          loading={loading}
        />

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Stats cards */}
          <SignalSummaryCards indexResult={indexResult} changeResult={changeResult} loading={loading} />

          {/* Error */}
          {error && (
            <div className="alert alert--warning">
              <span className="alert__icon">⚠️</span>
              <div>
                <div className="alert__title">Analysis failed</div>
                {error}
              </div>
            </div>
          )}

          {/* Insights */}
          {(indexResult || changeResult) && (
            <InsightCard indexResult={indexResult} changeResult={changeResult} />
          )}

          {/* Empty state */}
          {!indexResult && !changeResult && !loading && !error && (
            <div className="card">
              <div className="empty-state">
                <div className="empty-state__icon">🔭</div>
                <div className="empty-state__title">No analysis run yet</div>
                <div className="empty-state__body">
                  Select a study area and two years on the left, then click
                  "Analyse Changes" to see results here.
                </div>
              </div>
            </div>
          )}

          {/* Loading */}
          {loading && (
            <div className="card">
              <div className="loading-overlay">
                <div className="spinner" />
                <span>Retrieving satellite data and computing changes…</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Availability timeline ────────────────────────── */}
      {availability && (
        <div className="card mt-6">
          <div className="card-label">Satellite imagery availability — Coimbatore</div>
          <div className="card-body" style={{ marginBottom: '1rem' }}>
            Number of usable Sentinel-2 scenes per dry season (November–February)
            after cloud filtering. {!geeOk && <em>Shown values are illustrative.</em>}
          </div>
          <AvailabilityBar availability={availability} />
        </div>
      )}
    </main>
  )
}

/* Small inline bar chart for scene availability */
function AvailabilityBar({ availability }) {
  const years = availability?.years || []
  const max   = Math.max(...years.map(y => y.scene_count || 0), 1)
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: '6px', height: '80px' }}>
      {years.map(y => {
        const h = Math.max(((y.scene_count || 0) / max) * 60, 4)
        return (
          <div key={y.year} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
            <div
              title={`${y.year}: ${y.scene_count ?? 0} scenes, ~${y.cloud_cover_pct?.toFixed(0) ?? '?'}% cloud`}
              style={{
                width: '100%',
                height: `${h}px`,
                background: y.available ? 'var(--vegetation)' : 'var(--border)',
                borderRadius: '3px 3px 0 0',
                transition: 'height 0.3s',
              }}
            />
            <span style={{ fontSize: '0.625rem', color: 'var(--text-muted)' }}>{String(y.year).slice(2)}</span>
          </div>
        )
      })}
    </div>
  )
}
