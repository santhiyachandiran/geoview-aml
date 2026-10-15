import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// Response interceptor — normalise errors
api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.detail || err.message || 'Unknown error'
    return Promise.reject(new Error(msg))
  }
)

export const getHealth = () => api.get('/health')
export const getStatus = () => api.get('/status')
export const getStudyAreas = () => api.get('/study-areas')

export const getImageryAvailability = (studyArea, yearStart, yearEnd) =>
  api.get('/imagery/availability', {
    params: { study_area: studyArea, year_start: yearStart, year_end: yearEnd },
  })

export const postAnalysisIndices = (payload) =>
  api.post('/analysis/indices', payload)

export const postChangeDetection = (payload) =>
  api.post('/analysis/change-detection', payload)

export const getAnalysisResults = (analysisId) =>
  api.get(`/analysis/results/${analysisId}`)

export const getModels = () => api.get('/models')
export const getExperiments = () => api.get('/experiments')
export const getModelPlots = () => api.get('/models/plots')
