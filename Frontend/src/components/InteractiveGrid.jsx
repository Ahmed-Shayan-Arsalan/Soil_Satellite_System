import React, { useMemo, useRef, useEffect, useCallback, useState } from 'react'

const CLASS_COLORS = {
  'AnnualCrop': '#3b82f6',
  'Forest': '#15803d',
  'HerbaceousVegetation': '#22c55e',
  'Highway': '#6b7280',
  'Industrial': '#78716c',
  'Pasture': '#84cc16',
  'PermanentCrop': '#0ea5e9',
  'Residential': '#f59e0b',
  'River': '#06b6d4',
  'SeaLake': '#0284c7',
}

// Pre-compute color values for performance
const CLASS_COLORS_RGB = Object.fromEntries(
  Object.entries(CLASS_COLORS).map(([key, hex]) => {
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return [key, { r, g, b }]
  })
)

function getHealthColorRGB(score) {
  if (score < 0.4) {
    const t = score / 0.4
    return { r: 239, g: Math.round(68 + t * 115), b: Math.round(68 + t * 4) }
  } else if (score < 0.7) {
    const t = (score - 0.4) / 0.3
    return { r: Math.round(234 - t * 200), g: Math.round(179 + t * 17), b: Math.round(72 - t * 38) }
  } else {
    const t = (score - 0.7) / 0.3
    return { r: Math.round(34 - t * 10), g: Math.round(197 + t * 20), b: Math.round(94 + t * 20) }
  }
}

function getConfidenceColorRGB(confidence) {
  return {
    r: Math.round(139 * confidence),
    g: Math.round(92 * confidence),
    b: Math.round(confidence * 255)
  }
}

const InteractiveGrid = React.memo(function InteractiveGrid({
  predictions,
  gridSize,
  viewMode,
  enabledClasses,
  showGrid,
  selectedCell,
  onCellSelect,
  imageUrl
}) {
  const [rows, cols] = gridSize
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const [hoveredCell, setHoveredCell] = useState(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })
  
  // Organize predictions into a 2D grid - memoized
  const grid = useMemo(() => {
    const g = Array(rows).fill(null).map(() => Array(cols).fill(null))
    predictions.forEach(cell => {
      g[cell.row][cell.col] = cell
    })
    return g
  }, [predictions, rows, cols])

  // Get cell color based on view mode
  const getCellColorRGB = useCallback((cell) => {
    if (!enabledClasses.has(cell.class_name)) {
      return { r: 0, g: 0, b: 0, a: 0.2 }
    }
    
    let rgb
    let alpha = 0.85
    
    switch (viewMode) {
      case 'health':
        rgb = getHealthColorRGB(cell.health_score)
        break
      case 'confidence':
        rgb = getConfidenceColorRGB(cell.confidence)
        alpha = cell.confidence
        break
      case 'classification':
      default:
        rgb = CLASS_COLORS_RGB[cell.class_name] || { r: 107, g: 114, b: 128 }
    }
    
    return { ...rgb, a: alpha }
  }, [viewMode, enabledClasses])

  // Draw grid on canvas - high performance rendering
  useEffect(() => {
    if (!showGrid) return
    
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    const width = canvas.width
    const height = canvas.height
    const cellWidth = width / cols
    const cellHeight = height / rows
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height)
    
    // Draw all cells in a single pass
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const cell = grid[i][j]
        if (!cell) continue
        
        const x = j * cellWidth
        const y = i * cellHeight
        
        const color = getCellColorRGB(cell)
        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${color.a})`
        ctx.fillRect(x, y, cellWidth, cellHeight)
        
        // Draw cell border
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)'
        ctx.lineWidth = 1
        ctx.strokeRect(x, y, cellWidth, cellHeight)
      }
    }
    
    // Draw selected cell highlight
    if (selectedCell) {
      const x = selectedCell.col * cellWidth
      const y = selectedCell.row * cellHeight
      ctx.strokeStyle = 'white'
      ctx.lineWidth = 2
      ctx.strokeRect(x + 1, y + 1, cellWidth - 2, cellHeight - 2)
      
      // Draw pulse indicator
      ctx.fillStyle = 'white'
      ctx.beginPath()
      ctx.arc(x + cellWidth / 2, y + cellHeight / 2, 3, 0, Math.PI * 2)
      ctx.fill()
    }
    
  }, [grid, rows, cols, viewMode, enabledClasses, showGrid, selectedCell, getCellColorRGB])

  // Handle mouse events at container level for performance
  const getCellFromEvent = useCallback((e) => {
    const canvas = canvasRef.current
    if (!canvas) return null
    
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    
    const cellWidth = rect.width / cols
    const cellHeight = rect.height / rows
    
    const col = Math.floor(x / cellWidth)
    const row = Math.floor(y / cellHeight)
    
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      return grid[row][col]
    }
    return null
  }, [grid, rows, cols])

  const handleClick = useCallback((e) => {
    const cell = getCellFromEvent(e)
    if (cell) {
      const isSelected = selectedCell?.row === cell.row && selectedCell?.col === cell.col
      onCellSelect(isSelected ? null : cell)
    }
  }, [getCellFromEvent, selectedCell, onCellSelect])

  const handleMouseMove = useCallback((e) => {
    const cell = getCellFromEvent(e)
    if (cell !== hoveredCell) {
      setHoveredCell(cell)
      if (cell) {
        setTooltipPos({ x: e.clientX + 10, y: e.clientY + 10 })
      }
    } else if (cell) {
      setTooltipPos({ x: e.clientX + 10, y: e.clientY + 10 })
    }
  }, [getCellFromEvent, hoveredCell])

  const handleMouseLeave = useCallback(() => {
    setHoveredCell(null)
  }, [])

  return (
    <div 
      ref={containerRef}
      className="relative aspect-square bg-zinc-900 rounded-lg overflow-hidden"
    >
      {/* Background Image */}
      {imageUrl && (
        <img
          src={imageUrl}
          alt="Analyzed"
          className="absolute inset-0 w-full h-full object-cover opacity-30"
          loading="lazy"
        />
      )}
      
      {/* Canvas Grid Overlay - Much faster than DOM elements */}
      {showGrid && (
        <canvas
          ref={canvasRef}
          width={512}
          height={512}
          className="absolute inset-0 w-full h-full cursor-pointer"
          onClick={handleClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        />
      )}
      
      {/* Tooltip - Only render when hovering */}
      {hoveredCell && (
        <div
          className="fixed z-50 p-2 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl pointer-events-none"
          style={{ left: tooltipPos.x, top: tooltipPos.y }}
        >
          <div className="text-xs font-semibold text-white">{hoveredCell.class_name}</div>
          <div className="text-xs text-zinc-400">
            Health: {(hoveredCell.health_score * 100).toFixed(1)}%<br/>
            Conf: {(hoveredCell.confidence * 100).toFixed(1)}%
          </div>
        </div>
      )}
      
      {/* Color Legend for current view mode */}
      <div className="absolute bottom-2 left-2 right-2 flex justify-between items-end pointer-events-none">
        <div className="text-xs text-white/70 bg-black/50 px-2 py-1 rounded">
          {viewMode === 'health' && 'Health: Poor → Good'}
          {viewMode === 'confidence' && 'Confidence: Low → High'}
          {viewMode === 'classification' && 'Land Cover Classes'}
        </div>
        
        {/* Gradient legend for health/confidence */}
        {(viewMode === 'health' || viewMode === 'confidence') && (
          <div className="flex flex-col items-end text-xs text-white/70">
            <div 
              className="w-24 h-3 rounded"
              style={{
                background: viewMode === 'health'
                  ? 'linear-gradient(to right, #ef4444, #eab308, #22c55e)'
                  : 'linear-gradient(to right, #1e1b4b, #8b5cf6)'
              }}
            />
            <div className="flex justify-between w-24 mt-1">
              <span>0%</span>
              <span>100%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
})

export default InteractiveGrid
