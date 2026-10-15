import React from 'react'

export default function Header({ geeStatus, onThemeToggle, currentTheme }) {
  const statusLabel = geeStatus?.components?.find(c => c.name === 'google_earth_engine')
  const geeOk = statusLabel?.status === 'ok'

  return (
    <header style={{
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      padding: '0.75rem 1.5rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexShrink: 0,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <span style={{ color: 'var(--primary)', fontSize: '1.25rem' }}>🛰</span>
        <span style={{ fontWeight: 700, fontSize: '1rem', letterSpacing: '0.05em' }}>
          GEOVIEW AML
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
          Land-Cover Change Detection
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {statusLabel && (
          <span className={`pill ${geeOk ? 'pill--ok' : 'pill--warn'}`}>
            {geeOk ? '● GEE Live' : '● Demo Mode'}
          </span>
        )}
        <button className="btn btn--ghost" onClick={onThemeToggle} style={{ padding: '0.375rem 0.75rem' }}>
          {currentTheme === 'dark' ? '☀ Light' : '🌙 Dark'}
        </button>
      </div>
    </header>
  )
}
