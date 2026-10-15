import React, { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Rectangle, LayersControl, Tooltip, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { getStudyAreas } from '../services/api'

// Fix Leaflet icon
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl:       'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl:     'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

const STUDY_AREAS_MAP = {
  coimbatore: { center: [11.0168, 76.9558], zoom: 11, bbox: [76.8, 10.9, 77.2, 11.1] },
  chennai:    { center: [13.05,  80.25],    zoom: 11, bbox: [80.1, 12.9, 80.4, 13.2] },
  bengaluru:  { center: [12.95,  77.6],     zoom: 11, bbox: [77.4, 12.8, 77.8, 13.1] },
}

const LAYERS = [
  { id: 'osm',    label: 'Street Map',       desc: 'OpenStreetMap base layer' },
  { id: 'topo',   label: 'Terrain',          desc: 'Topographic hillshade' },
  { id: 'esri',   label: 'Satellite (ESRI)', desc: 'ESRI World Imagery (static background)' },
]

const OVERLAY_COLORS = {
  vegetation: { fill: '#40916C', label: 'Vegetation / Green cover' },
  water:      { fill: '#1A6EA8', label: 'Water bodies' },
  buildup:    { fill: '#C07A2A', label: 'Built-up / Urban' },
  bare:       { fill: '#8B7355', label: 'Bare soil / Cleared land' },
}

function ClickInspector({ onClick }) {
  useMapEvents({
    click(e) { onClick(e.latlng) }
  })
  return null
}

function MapResetButton({ center, zoom }) {
  // Leaflet reset — can't call useMap outside MapContainer, so we use a ref approach
  return null  // handled via key reset
}

export default function ExploreMap() {
  const [studyAreas, setStudyAreas] = useState([])
  const [selected, setSelected]     = useState('coimbatore')
  const [baseLayer, setBaseLayer]   = useState('osm')
  const [overlays, setOverlays]     = useState({ vegetation: true, water: true, buildup: true, bare: false })
  const [clickedPoint, setClickedPoint] = useState(null)
  const [mapKey, setMapKey]         = useState(0)

  useEffect(() => {
    getStudyAreas().then(r => setStudyAreas(r.data)).catch(() => {})
  }, [])

  const config = STUDY_AREAS_MAP[selected] || STUDY_AREAS_MAP.coimbatore
  const bbox   = config.bbox

  function toggleOverlay(id) {
    setOverlays(o => ({ ...o, [id]: !o[id] }))
  }

  const tileLayers = {
    osm:  { url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            attr: '&copy; OpenStreetMap' },
    topo: { url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
            attr: '&copy; OpenTopoMap' },
    esri: { url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr: '&copy; Esri' },
  }

  const tile = tileLayers[baseLayer]

  return (
    <main className="page-body page-body--full">
      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', height: 'calc(100vh - 61px)' }}>
        {/* ── Sidebar ─────────────────────────────────── */}
        <div style={{
          background: 'var(--surface)',
          borderRight: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'auto',
          padding: '1.25rem',
          gap: '1.25rem',
        }}>
          <div>
            <div className="card-label">Study Area</div>
            <select
              className="form-control"
              value={selected}
              onChange={e => { setSelected(e.target.value); setMapKey(k => k + 1) }}
            >
              {(studyAreas.length ? studyAreas : [
                { slug: 'coimbatore', name: 'Coimbatore, Tamil Nadu' }
              ]).map(a => (
                <option key={a.slug} value={a.slug}>{a.name}</option>
              ))}
            </select>
          </div>

          <div>
            <div className="card-label">Base Map</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {LAYERS.map(l => (
                <label key={l.id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.875rem' }}>
                  <input type="radio" name="baselayer" value={l.id} checked={baseLayer === l.id}
                    onChange={() => setBaseLayer(l.id)} />
                  <span>{l.label}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <div className="card-label">Illustrative Overlays</div>
            <div className="alert alert--demo" style={{ marginBottom: '0.75rem', padding: '0.5rem 0.75rem', fontSize: '0.8rem' }}>
              📋 These overlays are illustrative zone indicators, not classified satellite maps.
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {Object.entries(OVERLAY_COLORS).map(([id, { fill, label }]) => (
                <label key={id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.875rem' }}>
                  <input type="checkbox" checked={overlays[id]} onChange={() => toggleOverlay(id)} />
                  <span style={{ width: 12, height: 12, background: fill, borderRadius: 2, flexShrink: 0 }} />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <div className="card-label">Legend</div>
            <div className="map-legend" style={{ boxShadow: 'none', border: 'none', padding: 0 }}>
              <div className="map-legend__item">
                <div className="map-legend__swatch" style={{ background: 'var(--brand)', border: '2px dashed var(--brand)' }} />
                Study area boundary
              </div>
              {Object.entries(OVERLAY_COLORS).filter(([id]) => overlays[id]).map(([id, { fill, label }]) => (
                <div key={id} className="map-legend__item">
                  <div className="map-legend__swatch" style={{ background: fill, opacity: 0.5 }} />
                  {label}
                </div>
              ))}
            </div>
          </div>

          <button className="btn btn--ghost btn--sm" onClick={() => setMapKey(k => k + 1)}>
            ↺ Reset view
          </button>

          {clickedPoint && (
            <div className="card card--compact">
              <div className="card-label">Last click</div>
              <div style={{ fontSize: '0.875rem', fontFamily: 'var(--font-mono)' }}>
                {clickedPoint.lat.toFixed(5)}, {clickedPoint.lng.toFixed(5)}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                Pixel-level classification requires a trained model and real GEE data.
              </div>
            </div>
          )}
        </div>

        {/* ── Map ─────────────────────────────────────── */}
        <div style={{ position: 'relative' }}>
          <MapContainer
            key={mapKey}
            center={config.center}
            zoom={config.zoom}
            style={{ height: '100%', width: '100%' }}
            scrollWheelZoom
          >
            <TileLayer url={tile.url} attribution={tile.attr} />
            <ClickInspector onClick={setClickedPoint} />

            {/* Study area boundary */}
            <Rectangle
              bounds={[[bbox[1], bbox[0]], [bbox[3], bbox[2]]]}
              pathOptions={{ color: 'var(--brand)', weight: 2.5, fill: false, dashArray: '6 4' }}
            >
              <Tooltip sticky>Study area boundary — {selected}</Tooltip>
            </Rectangle>

            {/* Illustrative overlays — subdivide bbox into quadrants */}
            {Object.entries(OVERLAY_COLORS).map(([id, { fill }]) => {
              if (!overlays[id]) return null
              // Very rough zone positions (illustrative only)
              const zones = {
                vegetation: [[bbox[1] + 0.1, bbox[0]],            [bbox[3], bbox[0] + (bbox[2]-bbox[0])*0.5]],
                water:      [[bbox[1],         bbox[0] + (bbox[2]-bbox[0])*0.6], [(bbox[1]+bbox[3])/2, bbox[2]]],
                buildup:    [[(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2 - 0.05], [bbox[3] - 0.05, bbox[2] - 0.05]],
                bare:       [[bbox[1], bbox[0]], [(bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2]],
              }
              return (
                <Rectangle
                  key={id}
                  bounds={zones[id]}
                  pathOptions={{ color: fill, weight: 1, fillColor: fill, fillOpacity: 0.22 }}
                >
                  <Tooltip>{OVERLAY_COLORS[id].label} — Illustrative demo overlay</Tooltip>
                </Rectangle>
              )
            })}
          </MapContainer>

          {/* Demo label */}
          <div style={{
            position: 'absolute', bottom: 12, left: 12, zIndex: 500,
            background: 'rgba(255,255,255,0.92)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)',
            padding: '6px 12px',
            fontSize: '0.75rem',
            color: 'var(--text-secondary)',
            backdropFilter: 'blur(4px)',
          }}>
            📋 Overlays are illustrative. Real analysis requires GEE authentication.
          </div>
        </div>
      </div>
    </main>
  )
}
