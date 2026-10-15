import React from 'react'

const NAV_ITEMS = [
  { hash: '#dashboard', label: 'Dashboard', icon: '📊' },
  { hash: '#analysis',  label: 'Analysis',  icon: '🔍' },
  { hash: '#modeleval', label: 'Models',     icon: '🤖' },
]

export default function Sidebar({ currentHash }) {
  return (
    <nav style={{
      width: '200px',
      background: 'var(--surface)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      padding: '1rem 0',
      flexShrink: 0,
    }}>
      {NAV_ITEMS.map(({ hash, label, icon }) => {
        const active = currentHash === hash
        return (
          <a
            key={hash}
            href={hash}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              padding: '0.75rem 1.25rem',
              color: active ? 'var(--primary)' : 'var(--text-muted)',
              background: active ? 'rgba(0,229,255,0.08)' : 'transparent',
              borderLeft: active ? '3px solid var(--primary)' : '3px solid transparent',
              textDecoration: 'none',
              fontWeight: active ? 600 : 400,
              fontSize: '0.875rem',
              transition: 'all 0.15s',
            }}
          >
            <span>{icon}</span>
            <span>{label}</span>
          </a>
        )
      })}
    </nav>
  )
}
