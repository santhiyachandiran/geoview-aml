import React, { useState } from 'react'

const NAV_LINKS = [
  { hash: '#overview', label: 'Overview',         icon: '🌍' },
  { hash: '#explore',  label: 'Explore Map',       icon: '🗺️' },
  { hash: '#compare',  label: 'Compare Years',     icon: '⇔' },
  { hash: '#insights', label: 'Change Insights',   icon: '🔍' },
  { hash: '#models',   label: 'Model Performance', icon: '📊' },
  { hash: '#about',    label: 'About the Project', icon: 'ℹ️' },
]

export default function TopNav({ currentHash, apiStatus, currentTheme, onThemeToggle }) {
  const [menuOpen, setMenuOpen] = useState(false)

  const geeComp = apiStatus?.components?.find(c => c.name === 'google_earth_engine')
  const isDemo  = !geeComp || geeComp.status !== 'ok'

  return (
    <nav className="topnav">
      <div className="topnav__inner">
        {/* Brand */}
        <a href="#overview" className="topnav__brand">
          <div className="topnav__logo">🛰</div>
          <div>
            <div className="topnav__title">GEOVIEW AML</div>
            <div className="topnav__subtitle">Satellite Change Detection</div>
          </div>
        </a>

        {/* Links — hidden on small screens */}
        <div className="topnav__links" style={{ display: 'flex' }}>
          {NAV_LINKS.map(({ hash, label, icon }) => (
            <a
              key={hash}
              href={hash}
              className={`topnav__link${currentHash === hash ? ' topnav__link--active' : ''}`}
            >
              <span style={{ fontSize: '0.875rem' }}>{icon}</span>
              {label}
            </a>
          ))}
        </div>

        {/* Actions */}
        <div className="topnav__actions">
          {/* Data mode indicator */}
          <span
            className={`badge ${isDemo ? 'badge--demo' : 'badge--live'}`}
            title={isDemo
              ? 'Running with synthetic demo data — no real satellite imagery'
              : 'Connected to Google Earth Engine'}
          >
            {isDemo ? '📋 Demo Mode' : '🛰 Live Data'}
          </span>

          {/* Theme toggle */}
          <button
            className="btn btn--ghost btn--sm"
            onClick={onThemeToggle}
            title="Toggle light/dark mode"
          >
            {currentTheme === 'light' ? '🌙' : '☀️'}
          </button>
        </div>
      </div>
    </nav>
  )
}
