import React, { useEffect, useState } from 'react'
import MapView from '../components/Map/MapView'
import ChangeOverlay from '../components/Map/ChangeOverlay'
import StatsCard from '../components/Dashboard/StatsCard'
import AnalysisPanel from '../components/Dashboard/AnalysisPanel'
import ResultsPanel from '../components/Dashboard/ResultsPanel'
import { getStudyAreas } from '../services/api'
import { useAnalysis } from '../hooks/useAnalysis'

const COIMBATORE_BBOX = [76.8, 10.9, 77.2, 11.1]

export default function Dashboard() {
  const [studyAreas, setStudyAreas] = useState([])
  const { loading, error, indexResult, changeResult, runAnalysis } = useAnalysis()

  useEffect(() => {
    getStudyAreas().then(r => setStudyAreas(r.data)).catch(() => {})
  }, [])

  const handleRun = ({ studyArea, year1, year2, seasonStart, seasonEnd }) => {
    runAnalysis({ studyArea, year1, year2, seasonStart, seasonEnd })
  }

  return (
    <div className="page-body">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '1.25rem', height: '100%' }}>
        {/* Left column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="map-wrapper" style={{ height: '420px' }}>
            <MapView studyAreaBbox={COIMBATORE_BBOX}>
              {changeResult && <ChangeOverlay changeResult={changeResult} bbox={COIMBATORE_BBOX} />}
            </MapView>
          </div>

          <div className="grid-3">
            <StatsCard title="ΔNDVI" value={indexResult?.delta_ndvi?.toFixed(3)}
              subtitle="Vegetation change" trend={indexResult?.delta_ndvi} color="var(--positive)" />
            <StatsCard title="ΔNDWI" value={indexResult?.delta_ndwi?.toFixed(3)}
              subtitle="Water change" trend={indexResult?.delta_ndwi} color="var(--primary)" />
            <StatsCard title="ΔNDBI" value={indexResult?.delta_ndbi?.toFixed(3)}
              subtitle="Built-up change" trend={indexResult?.delta_ndbi} color="var(--negative)" />
          </div>

          <ResultsPanel indexResult={indexResult} changeResult={changeResult} error={error} />
        </div>

        {/* Right column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <AnalysisPanel studyAreas={studyAreas} onRun={handleRun} loading={loading} />
          {changeResult && (
            <div className="card" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              <div className="card-title">Data Provenance</div>
              <p>{changeResult.data_provenance?.notes}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
