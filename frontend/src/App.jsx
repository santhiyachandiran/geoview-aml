import React, { useState, useEffect } from 'react'
import TopNav from './components/Layout/TopNav'
import Overview from './pages/Overview'
import ExploreMap from './pages/ExploreMap'
import CompareYears from './pages/CompareYears'
import ChangeInsights from './pages/ChangeInsights'
import ModelPerformance from './pages/ModelPerformance'
import AboutProject from './pages/AboutProject'
import { getStatus } from './services/api'

const PAGES = {
  '#overview':      Overview,
  '#explore':       ExploreMap,
  '#compare':       CompareYears,
  '#insights':      ChangeInsights,
  '#models':        ModelPerformance,
  '#about':         AboutProject,
}

function getHash() {
  const h = window.location.hash || '#overview'
  return h.split('?')[0]
}

export default function App() {
  const [theme, setTheme]       = useState('light')
  const [hash, setHash]         = useState(getHash())
  const [apiStatus, setApiStatus] = useState(null)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    const handler = () => setHash(getHash())
    window.addEventListener('hashchange', handler)
    return () => window.removeEventListener('hashchange', handler)
  }, [])

  useEffect(() => {
    getStatus().then(r => setApiStatus(r.data)).catch(() => {})
  }, [])

  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light')

  const Page = PAGES[hash] || Overview

  return (
    <div className="app-shell">
      <TopNav
        currentHash={hash}
        apiStatus={apiStatus}
        currentTheme={theme}
        onThemeToggle={toggleTheme}
      />
      <Page apiStatus={apiStatus} />
    </div>
  )
}
