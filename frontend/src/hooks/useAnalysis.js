import { useState, useCallback, useRef } from 'react'
import { postAnalysisIndices, postChangeDetection, getAnalysisResults } from '../services/api'

export function useAnalysis() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [indexResult, setIndexResult] = useState(null)
  const [changeResult, setChangeResult] = useState(null)
  const pollRef = useRef(null)

  const runAnalysis = useCallback(async ({ studyArea, year1, year2, seasonStart = 11, seasonEnd = 3 }) => {
    setLoading(true)
    setError(null)
    setIndexResult(null)
    setChangeResult(null)

    const payload = {
      study_area: studyArea,
      year1,
      year2,
      season_start_month: seasonStart,
      season_end_month: seasonEnd,
    }

    try {
      const [idxRes, chgRes] = await Promise.all([
        postAnalysisIndices(payload),
        postChangeDetection(payload),
      ])
      setIndexResult(idxRes.data)
      setChangeResult(chgRes.data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback(() => {
    setLoading(false)
    setError(null)
    setIndexResult(null)
    setChangeResult(null)
    if (pollRef.current) clearInterval(pollRef.current)
  }, [])

  return { loading, error, indexResult, changeResult, runAnalysis, reset }
}
