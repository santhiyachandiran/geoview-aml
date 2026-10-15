import React, { useEffect } from 'react'
import { MapContainer, TileLayer, Rectangle, useMap } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix Leaflet default icon paths broken by bundlers
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
})

// Coimbatore study area bbox [west, south, east, north]
const COIMBATORE_BBOX = [76.8, 10.9, 77.2, 11.1]

// Convert [west, south, east, north] to Leaflet [[south,west],[north,east]]
function bboxToBounds(bbox) {
  return [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
}

function StudyAreaRect({ bbox, color }) {
  return (
    <Rectangle
      bounds={bboxToBounds(bbox)}
      pathOptions={{
        color,
        weight: 2,
        fillOpacity: 0.08,
        dashArray: '6 4',
      }}
    />
  )
}

export default function MapView({ bboxOverlay, changeResult, studyAreaBbox }) {
  const bbox = studyAreaBbox || COIMBATORE_BBOX
  const center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]

  return (
    <MapContainer
      center={center}
      zoom={11}
      style={{ height: '100%', width: '100%' }}
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      <StudyAreaRect bbox={bbox} color="#00E5FF" />
      {bboxOverlay && (
        <StudyAreaRect bbox={bboxOverlay} color="#FF3366" />
      )}
    </MapContainer>
  )
}
