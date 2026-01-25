import React, { useState, useCallback } from 'react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import AnalysisPanel from './components/AnalysisPanel'
import { analyzeImage } from './api/analysis'

function App() {
  const [analysisResult, setAnalysisResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadedImage, setUploadedImage] = useState(null)

  const handleAnalyze = useCallback(async (file, options) => {
    setIsLoading(true)
    setError(null)
    
    try {
      // Create preview URL
      const imageUrl = URL.createObjectURL(file)
      setUploadedImage(imageUrl)
      
      const result = await analyzeImage(file, options)
      setAnalysisResult(result)
    } catch (err) {
      setError(err.message || 'Analysis failed')
      console.error('Analysis error:', err)
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleReset = useCallback(() => {
    setAnalysisResult(null)
    setUploadedImage(null)
    setError(null)
  }, [])

  return (
    <div className="min-h-screen">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-7xl">
        {!analysisResult ? (
          <ImageUpload 
            onAnalyze={handleAnalyze}
            isLoading={isLoading}
            error={error}
          />
        ) : (
          <AnalysisPanel 
            result={analysisResult}
            imageUrl={uploadedImage}
            onReset={handleReset}
          />
        )}
      </main>
      
      <footer className="text-center py-6 text-zinc-500 text-sm">
        <p>Powered by Tempes Agricultural A.I systems</p>
      </footer>
    </div>
  )
}

export default App

