// Determine API base URL:
// - In production, you can set VITE_API_BASE_URL to the Render URL
//   e.g. VITE_API_BASE_URL="https://soil-satellite-system.onrender.com"
// - In development, we default to a local proxy at /api (configured in Vite)
const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.trim() ||
  '/api'

export async function analyzeImage(file, options = {}) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('grid_rows', options.gridRows || 32)
  formData.append('grid_cols', options.gridCols || 32)
  // Image type is now auto-detected by the backend based on file extension
  
  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  })
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  
  return response.json()
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`)
  return response.json()
}

