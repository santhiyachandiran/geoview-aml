import React, { useEffect, useState } from 'react'
import AnalysisPanel from '../components/Dashboard/AnalysisPanel'
import ResultsPanel from '../components/Dashboard/ResultsPanel'
import IndexTimeline from '../components/Charts/IndexTimeline'
import { getStudyAreas, getImageryAvailability } from '../services/api'
import { useAnalysis } from '../hooks/useAnalysis'

export default function Analysis() {
  const [studyAreas, setStudyAreas] = useState([])
  const [availability, setAvailability] = useState(null)
  const { loading, error, indexResult, changeResult, runAnalysis } = useAnalysis()

  useEffect(() => {
    getStudyAreas().then(r => setStudyAreas(r.data)).catch(() => {})
    getImageryAvailability('coimbatore', 2017, 2024).then(r => setAvailability(r.data)).catch(() => {})
  }, [])

  return (
    <div className="page-body">
      <h2 style={{ marginBottom: '1.25rem', fontSize: '1.1rem', fontWeight: 700 }}>Change Detection Analysis</h2>
      <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '1.25rem' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <AnalysisPanel studyAreas={studyAreas} onRun={runAnalysis} loading={loading} />
          {availability && (
            <div className="card">
              <div className="card-title">Imagery Availability (2017–2024)</div>
              <IndexTimeline availability={availability} />
            </div>
          )}
        </div>
        <ResultsPanel indexResult={indexResult} changeResult={changeResult} error={error} />
      </div>
    </div>
  )
}
