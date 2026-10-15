import React from 'react'

export default function TransitionMatrix({ changeResult }) {
  if (!changeResult?.transition_matrix) {
    return (
      <div style={{ color: 'var(--text-muted)', padding: '1rem', textAlign: 'center' }}>
        Transition matrix unavailable (requires pixel-level classification)
      </div>
    )
  }

  const { matrix, from_classes, to_classes } = changeResult.transition_matrix
  const maxVal = Math.max(...matrix.flat(), 1)

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: '0.8rem' }}>
        <thead>
          <tr>
            <th style={{ padding: '0.4rem 0.6rem', color: 'var(--text-muted)' }}>From ↓ To →</th>
            {to_classes.map(c => (
              <th key={c} style={{ padding: '0.4rem 0.6rem', color: 'var(--primary)' }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={from_classes[i]}>
              <td style={{ padding: '0.4rem 0.6rem', color: 'var(--positive)', fontWeight: 600 }}>
                {from_classes[i]}
              </td>
              {row.map((val, j) => {
                const intensity = val / maxVal
                const bg = i === j
                  ? `rgba(0,229,255,${intensity * 0.3})`
                  : `rgba(255,51,102,${intensity * 0.4})`
                return (
                  <td key={j} style={{
                    padding: '0.4rem 0.6rem', textAlign: 'center',
                    background: bg, color: 'var(--text)',
                  }}>
                    {val.toLocaleString()}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
