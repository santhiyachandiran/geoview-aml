import React from 'react'

/*
  Plain-English interpretation of analysis results.
  Never fabricates values — only summarises what the API returned.
*/

function confidenceText(label) {
  const map = {
    high_proxy_confidence:   'Strong signal',
    medium_proxy_confidence: 'Moderate signal',
    low_proxy_confidence:    'Weak signal',
    no_transition_detected:  'No clear transition',
    unknown:                 'Uncertain',
  }
  return map[label] || label
}

function VegetationInsight({ delta, year1, year2 }) {
  if (delta === null || delta === undefined) return null
  const abs = Math.abs(delta).toFixed(3)
  const isDecline = delta < -0.04

  return (
    <div className="insight-card">
      <div className="insight-card__header">
        <div className="insight-card__icon" style={{ background: isDecline ? 'var(--loss-bg)' : 'var(--vegetation-bg)' }}>
          {isDecline ? '🍂' : '🌿'}
        </div>
        <div>
          <div className="insight-card__finding">
            {isDecline
              ? `Vegetation greenness declined between ${year1} and ${year2}`
              : Math.abs(delta) < 0.04
                ? `Vegetation greenness was broadly stable between ${year1} and ${year2}`
                : `Vegetation greenness increased between ${year1} and ${year2}`}
          </div>
          <span className={`badge ${isDecline ? 'badge--warn' : 'badge--success'}`} style={{ marginTop: 4 }}>
            Change: {delta > 0 ? '+' : ''}{delta.toFixed(3)}
          </span>
        </div>
      </div>
      <div className="insight-card__detail">
        {isDecline
          ? `The vegetation greenness signal decreased by ${abs} units. This is consistent with vegetation loss or land cleared for development, though it could also reflect seasonal differences, cloud contamination, or a dry year.`
          : `No substantial vegetation decline was detected. Areas may have remained as green cover or experienced modest regrowth.`}
      </div>
      <div className="insight-card__footer">
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Based on satellite reflectance, not field survey
        </span>
        <a href="#compare" className="btn btn--secondary btn--sm">View on map →</a>
      </div>
    </div>
  )
}

function BuiltupInsight({ delta, year1, year2 }) {
  if (delta === null || delta === undefined) return null
  const isIncrease = delta > 0.04

  return (
    <div className="insight-card">
      <div className="insight-card__header">
        <div className="insight-card__icon" style={{ background: isIncrease ? 'var(--buildup-bg)' : 'var(--vegetation-bg)' }}>
          {isIncrease ? '🏗️' : '🏘️'}
        </div>
        <div>
          <div className="insight-card__finding">
            {isIncrease
              ? `Built-up surface signal increased between ${year1} and ${year2}`
              : `Built-up surface signal showed little change between ${year1} and ${year2}`}
          </div>
          <span className={`badge ${isIncrease ? 'badge--warn' : 'badge--success'}`} style={{ marginTop: 4 }}>
            Change: {delta > 0 ? '+' : ''}{delta.toFixed(3)}
          </span>
        </div>
      </div>
      <div className="insight-card__detail">
        {isIncrease
          ? `The built-up surface signal increased by ${Math.abs(delta).toFixed(3)} units. This is consistent with new construction or paving, but a higher signal can also appear in bare, dry, or degraded soil. Field verification is needed to confirm actual building activity.`
          : `No substantial increase in the built-up surface signal was detected in the study area as a whole.`}
      </div>
      <div className="insight-card__footer">
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Proxy indicator — not a validated building count
        </span>
        <a href="#compare" className="btn btn--secondary btn--sm">View on map →</a>
      </div>
    </div>
  )
}

function TransitionInsight({ changeResult }) {
  if (!changeResult) return null
  const { vegetation_to_buildup_flag, confidence_label, year1, year2, demo_data } = changeResult

  return (
    <div className="insight-card">
      <div className="insight-card__header">
        <div className="insight-card__icon" style={{ background: vegetation_to_buildup_flag ? 'var(--loss-bg)' : 'var(--vegetation-bg)' }}>
          {vegetation_to_buildup_flag ? '⚠️' : '✅'}
        </div>
        <div>
          <div className="insight-card__finding">
            {vegetation_to_buildup_flag
              ? `Spectral signals suggest possible vegetation-to-built-up transition (${year1}–${year2})`
              : `No clear vegetation-to-built-up transition detected (${year1}–${year2})`}
          </div>
          <span className={`badge ${vegetation_to_buildup_flag ? 'badge--warn' : 'badge--success'}`} style={{ marginTop: 4 }}>
            {confidenceText(confidence_label)}
          </span>
          {demo_data && <span className="badge badge--demo" style={{ marginTop: 4, marginLeft: 4 }}>Demo</span>}
        </div>
      </div>
      <div className="insight-card__detail">
        {vegetation_to_buildup_flag
          ? `Both the vegetation signal decreased and the built-up signal increased beyond the detection threshold. This pattern is consistent with land being cleared and developed. However, this is a proxy-based estimate — it does not confirm the type of development, exact location, or area affected without pixel-level classification and field validation.`
          : `The two index thresholds for a vegetation-to-built-up transition were not both exceeded. Either change was below the threshold, or the area changed in a different way.`}
      </div>
      <div className="insight-card__footer">
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Threshold: Vegetation greenness &lt; −0.10 AND Built-up signal &gt; +0.10
        </span>
        <a href="#insights" className="btn btn--secondary btn--sm">Full insights →</a>
      </div>
    </div>
  )
}

export default function InsightCard({ indexResult, changeResult }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <VegetationInsight
        delta={indexResult?.delta_ndvi}
        year1={indexResult?.year1}
        year2={indexResult?.year2}
      />
      <BuiltupInsight
        delta={indexResult?.delta_ndbi}
        year1={indexResult?.year1}
        year2={indexResult?.year2}
      />
      <TransitionInsight changeResult={changeResult} />
    </div>
  )
}
