import React, { useEffect, useState } from 'react'
import { getStudyAreas } from '../services/api'
import { useAnalysis } from '../hooks/useAnalysis'
import AnalysisControls from '../components/Dashboard/AnalysisControls'
import InsightCard from '../components/Dashboard/InsightCard'
import SignalSummaryCards from '../components/Dashboard/SignalSummaryCards'

export default function ChangeInsights() {
  const [studyAreas, setStudyAreas] = useState([])
  const { loading, error, indexResult, changeResult, runAnalysis, reset } = useAnalysis()

  useEffect(() => {
    getStudyAreas().then(r => setStudyAreas(r.data)).catch(() => {})
  }, [])

  const isDemo = indexResult?.demo_data || changeResult?.demo_data

  return (
    <main className="page-body">
      <div className="section-header">
        <div>
          <h1 className="section-title">Change Insights</h1>
          <p className="section-desc">
            Plain-English summaries of what the satellite signals suggest about
            land-cover change between two years.
          </p>
        </div>
      </div>

      {isDemo && (
        <div className="alert alert--demo mb-6">
          <span className="alert__icon">📋</span>
          <div>
            <div className="alert__title">Illustrative results — not real satellite data</div>
            Run <code>earthengine authenticate</code> and set <code>GEE_PROJECT_ID</code> to see results
            based on actual Sentinel-2 imagery.
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px,320px) 1fr', gap: '1.5rem', alignItems: 'start' }}>
        <AnalysisControls
          studyAreas={studyAreas}
          onRun={runAnalysis}
          onReset={reset}
          loading={loading}
        />

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {error && (
            <div className="alert alert--warning">
              <span className="alert__icon">⚠️</span>
              <div><div className="alert__title">Analysis failed</div>{error}</div>
            </div>
          )}

          {loading && (
            <div className="card">
              <div className="loading-overlay">
                <div className="spinner" />
                <span>Computing change insights…</span>
              </div>
            </div>
          )}

          {indexResult && !loading && (
            <SignalSummaryCards indexResult={indexResult} changeResult={changeResult} loading={false} />
          )}

          {(indexResult || changeResult) && !loading && (
            <>
              <InsightCard indexResult={indexResult} changeResult={changeResult} />
              <LimitationsNote />
            </>
          )}

          {!indexResult && !changeResult && !loading && !error && (
            <div className="card">
              <div className="empty-state">
                <div className="empty-state__icon">🔍</div>
                <div className="empty-state__title">No insights yet</div>
                <div className="empty-state__body">
                  Run an analysis to see plain-English explanations of detected land-cover changes.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}

function LimitationsNote() {
  return (
    <div className="card" style={{ background: 'var(--surface-2)' }}>
      <div className="card-label">Understanding these results</div>
      <div className="prose">
        <p>
          The insights above are based on <strong>spectral index comparisons</strong> — mathematical
          ratios of satellite band reflectance values averaged over the entire study area.
          They indicate broad trends, not individual building-level changes.
        </p>
        <details className="tech-details">
          <summary className="tech-details__summary">Important limitations to understand</summary>
          <div className="tech-details__body">
            <ul>
              <li><strong>Seasonal differences:</strong> If the two images were taken in different seasons,
                apparent change may be due to crop cycles or rainfall, not land-use change.</li>
              <li><strong>Cloud contamination:</strong> Residual cloud or haze affects reflectance values
                even after cloud masking.</li>
              <li><strong>Resolution:</strong> Each pixel covers 10×10 metres. Mixed pixels near boundaries
                are misclassified.</li>
              <li><strong>Index ≠ class:</strong> A high NDBI does not guarantee buildings — bare soil and
                dry fields also produce high values. Supervised classification with validated labels
                is required for accurate land-cover mapping.</li>
              <li><strong>Demo data:</strong> If the demo flag is shown, all numbers are synthetic and
                should not be cited as measurements.</li>
            </ul>
          </div>
        </details>
      </div>
    </div>
  )
}
