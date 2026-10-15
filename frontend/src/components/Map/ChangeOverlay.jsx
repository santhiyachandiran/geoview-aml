import React from 'react'
import { Rectangle, Tooltip } from 'react-leaflet'

function bboxToBounds(bbox) {
  return [[bbox[1], bbox[0]], [bbox[3], bbox[2]]]
}

export default function ChangeOverlay({ changeResult, bbox }) {
  if (!changeResult || !bbox) return null
  const flag = changeResult.vegetation_to_buildup_flag
  const color = flag ? '#FF3366' : '#10B981'
  const label = flag
    ? `Possible veg→built-up transition (${changeResult.confidence_label})`
    : 'No transition detected'

  return (
    <Rectangle
      bounds={bboxToBounds(bbox)}
      pathOptions={{ color, weight: 2, fillOpacity: 0.18 }}
    >
      <Tooltip>{label}</Tooltip>
    </Rectangle>
  )
}
