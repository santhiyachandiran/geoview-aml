import React, { useState } from 'react'
import { MapContainer, TileLayer, Rectangle, Tooltip } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix Leaflet marker icons
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl:       'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl:     'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

const STUDY_CENTERS = {
  coimbatore: [11.0168, 76.9558],
  chennai:    [13.05, 80.25],
  bengaluru:  [12.95, 77.6],
}

const TILE_SOURCES = {
  satellite: {
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr: '&copy; Esri, Maxar, Earthstar Geographics',
    name: 'Satellite (ESRI High-Res)',
  },
  osm: {
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attr: '&copy; OpenStreetMap contributors',
    name: 'Street Map (OSM)',
  },
}

export default function BeforeAfterMap({ year1 = 2018, year2 = 2024, studyArea = 'coimbatore', bbox = [76.8, 10.9, 77.2, 11.1], changeResult }) {
  const [mapType, setMapType] = useState('satellite')
  const center = STUDY_CENTERS[studyArea] || STUDY_CENTERS.coimbatore
  const bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
  const tile = TILE_SOURCES[mapType]
  const flag = changeResult?.vegetation_to_buildup_flag

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
          Side-by-side satellite observation comparison • Study area: <strong>{studyArea.toUpperCase()}</strong> (10m Sentinel-2 dry-season composite)
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className={`btn btn--sm ${mapType === 'satellite' ? 'btn--primary' : 'btn--ghost'}`}
            onClick={() => setMapType('satellite')}
          >
            🛰 Satellite Imagery
          </button>
          <button
            className={`btn btn--sm ${mapType === 'osm' ? 'btn--primary' : 'btn--ghost'}`}
            onClick={() => setMapType('osm')}
          >
            🗺 Street Map
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
        {/* Year 1 Map */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '6px 12px', background: 'var(--surface-2)', borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border)', fontSize: '0.85rem', fontWeight: 600
          }}>
            <span style={{ color: 'var(--vegetation)' }}>◀ BEFORE: {year1}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Nov {year1}–Mar {year1 + 1}</span>
          </div>
          <div style={{ height: '300px', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)' }}>
            <MapContainer center={center} zoom={11} style={{ height: '100%', width: '100%' }} scrollWheelZoom={false}>
              <TileLayer url={tile.url} attribution={tile.attr} />
              <Rectangle bounds={bounds} pathOptions={{ color: '#40916C', weight: 2.5, fill: false, dashArray: '5 5' }}>
                <Tooltip sticky>Study Area Boundary ({studyArea}) — {year1}</Tooltip>
              </Rectangle>
            </MapContainer>
          </div>
        </div>

        {/* Year 2 Map */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '6px 12px', background: 'var(--surface-2)', borderRadius: 'var(--radius-sm)',
            border: '1px solid var(--border)', fontSize: '0.85rem', fontWeight: 600
          }}>
            <span style={{ color: 'var(--primary)' }}>▶ AFTER: {year2}</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Nov {year2}–Mar {year2 + 1}</span>
          </div>
          <div style={{ height: '300px', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border)' }}>
            <MapContainer center={center} zoom={11} style={{ height: '100%', width: '100%' }} scrollWheelZoom={false}>
              <TileLayer url={tile.url} attribution={tile.attr} />
              <Rectangle bounds={bounds} pathOptions={{ color: '#1A6EA8', weight: 2.5, fill: false, dashArray: '5 5' }}>
                <Tooltip sticky>Study Area Boundary ({studyArea}) — {year2}</Tooltip>
              </Rectangle>
              {flag && (
                <Rectangle
                  bounds={[[bounds[0][0] + 0.05, bounds[0][1] + 0.05], [bounds[1][0] - 0.05, bounds[1][1] - 0.05]]}
                  pathOptions={{ color: '#FF3366', fillColor: '#FF3366', fillOpacity: 0.15, weight: 1.5 }}
                >
                  <Tooltip sticky>Change Hotspot: Veg→Built-up transition detected</Tooltip>
                </Rectangle>
              )}
            </MapContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
