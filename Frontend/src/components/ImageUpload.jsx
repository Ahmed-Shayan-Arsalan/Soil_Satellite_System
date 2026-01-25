import React, { useState, useRef, useCallback } from 'react'
import { Upload, Image, Grid3X3, Loader2, AlertCircle, FileImage, X } from 'lucide-react'

export default function ImageUpload({ onAnalyze, isLoading, error }) {
  const [dragOver, setDragOver] = useState(false)
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [options, setOptions] = useState({
    gridRows: 32,
    gridCols: 32
  })
  
  const fileInputRef = useRef(null)

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }, [])

  const handleFileSelect = (file) => {
    setSelectedFile(file)
    
    // Create preview for images
    if (file.type.startsWith('image/') || file.name.endsWith('.tif')) {
      const reader = new FileReader()
      reader.onloadend = () => setPreview(reader.result)
      reader.readAsDataURL(file)
    }
  }

  const handleSubmit = () => {
    if (selectedFile) {
      onAnalyze(selectedFile, options)
    }
  }

  const clearFile = () => {
    setSelectedFile(null)
    setPreview(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="animate-fade-in">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold mb-4 bg-gradient-to-r from-crop-400 via-crop-500 to-emerald-400 bg-clip-text text-transparent">
          Analyze Your Satellite Imagery
        </h2>
        <p className="text-zinc-400 max-w-2xl mx-auto text-lg">
          Upload multispectral GeoTIFF or RGB images for AI-powered crop classification 
          and health assessment using Tempes Agricultural A.I systems.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8 max-w-5xl mx-auto">
        {/* Upload Area */}
        <div className="gradient-border p-6">
          <div
            className={`drop-zone rounded-xl p-12 text-center transition-all ${
              dragOver ? 'drag-over' : ''
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff,.png,.jpg,.jpeg"
              className="hidden"
              onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])}
            />
            
            {selectedFile ? (
              <div className="space-y-4">
                <div className="relative inline-block">
                  {preview ? (
                    <img 
                      src={preview} 
                      alt="Preview" 
                      className="w-32 h-32 object-cover rounded-lg border border-zinc-700"
                    />
                  ) : (
                    <div className="w-32 h-32 bg-zinc-800 rounded-lg flex items-center justify-center">
                      <FileImage className="w-12 h-12 text-zinc-600" />
                    </div>
                  )}
                  <button
                    onClick={(e) => { e.stopPropagation(); clearFile() }}
                    className="absolute -top-2 -right-2 p-1 bg-red-500 rounded-full hover:bg-red-600 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div>
                  <p className="font-medium text-crop-400">{selectedFile.name}</p>
                  <p className="text-sm text-zinc-500">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="mx-auto w-16 h-16 rounded-full bg-crop-500/10 flex items-center justify-center">
                  <Upload className="w-8 h-8 text-crop-500" />
                </div>
                <div>
                  <p className="font-medium text-zinc-200">Drop your image here</p>
                  <p className="text-sm text-zinc-500 mt-1">
                    or click to browse
                  </p>
                </div>
                <div className="flex items-center justify-center gap-4 text-xs text-zinc-500">
                  <span className="px-2 py-1 bg-zinc-800 rounded">GeoTIFF</span>
                  <span className="px-2 py-1 bg-zinc-800 rounded">PNG</span>
                  <span className="px-2 py-1 bg-zinc-800 rounded">JPEG</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Options Panel */}
        <div className="gradient-border p-6 space-y-6">
          <h3 className="font-semibold text-lg flex items-center gap-2">
            <Grid3X3 className="w-5 h-5 text-crop-500" />
            Analysis Options
          </h3>
          
          {/* Grid Size */}
          <div className="space-y-3">
            <label className="block text-sm text-zinc-400">Grid Resolution</label>
            <div className="grid grid-cols-3 gap-2">
              {[32, 48, 64].map(size => (
                <button
                  key={size}
                  onClick={() => setOptions({ ...options, gridRows: size, gridCols: size })}
                  className={`py-3 px-4 rounded-lg text-sm font-medium transition-all ${
                    options.gridRows === size 
                      ? 'bg-crop-500 text-white' 
                      : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
                  }`}
                >
                  {size}×{size}
                </button>
              ))}
            </div>
            <p className="text-xs text-zinc-500">
              Higher resolution = more detail, slower processing
            </p>
          </div>

          {/* Error Display */}
          {error && (
            <div className="flex items-center gap-3 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {/* Analyze Button */}
          <button
            onClick={handleSubmit}
            disabled={!selectedFile || isLoading}
            className={`w-full py-4 rounded-xl font-semibold text-lg transition-all ${
              selectedFile && !isLoading
                ? 'bg-gradient-to-r from-crop-500 to-emerald-500 hover:from-crop-600 hover:to-emerald-600 text-white glow-green'
                : 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
            }`}
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="w-5 h-5 animate-spin" />
                Analyzing...
              </span>
            ) : (
              'Analyze Image'
            )}
          </button>
        </div>
      </div>

      {/* Supported Classes */}
      <div className="mt-12 text-center">
        <p className="text-sm text-zinc-500 mb-4">Detects 10 Land Cover Classes</p>
        <div className="flex flex-wrap justify-center gap-2 max-w-3xl mx-auto">
          {['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
            'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'].map(cls => (
            <span key={cls} className="px-3 py-1 bg-zinc-800/50 rounded-full text-xs text-zinc-400">
              {cls}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

