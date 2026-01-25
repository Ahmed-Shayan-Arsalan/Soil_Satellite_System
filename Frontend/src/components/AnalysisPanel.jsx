import React, { useState, useMemo } from 'react'
import { ArrowLeft, Layers, Heart, BarChart3, Grid3X3, Eye, EyeOff, Info } from 'lucide-react'
import InteractiveGrid from './InteractiveGrid'
import HealthPanel from './HealthPanel'
import ClassDistribution from './ClassDistribution'
import GridLegend from './GridLegend'
import CellDetails from './CellDetails'

export default function AnalysisPanel({ result, imageUrl, onReset }) {
  const [selectedCell, setSelectedCell] = useState(null)
  const [viewMode, setViewMode] = useState('classification') // 'classification', 'health', 'confidence'
  const [enabledClasses, setEnabledClasses] = useState(
    new Set(result.class_distribution.map(c => c.class_name))
  )
  const [showGrid, setShowGrid] = useState(true)

  // Calculate per-class health statistics
  const classHealthStats = useMemo(() => {
    const stats = {}
    result.grid_predictions.forEach(cell => {
      const className = cell.class_name
      if (!stats[className]) {
        stats[className] = { total: 0, count: 0, min: 1, max: 0 }
      }
      stats[className].total += cell.health_score
      stats[className].count++
      stats[className].min = Math.min(stats[className].min, cell.health_score)
      stats[className].max = Math.max(stats[className].max, cell.health_score)
    })
    
    return Object.entries(stats).map(([name, data]) => ({
      name,
      avg: data.total / data.count,
      min: data.min,
      max: data.max,
      count: data.count
    })).sort((a, b) => b.avg - a.avg)
  }, [result])

  const toggleClass = (className) => {
    const newEnabled = new Set(enabledClasses)
    if (newEnabled.has(className)) {
      newEnabled.delete(className)
    } else {
      newEnabled.add(className)
    }
    setEnabledClasses(newEnabled)
  }

  return (
    <div className="animate-fade-in space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <button
          onClick={onReset}
          className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>New Analysis</span>
        </button>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowGrid(!showGrid)}
            className={`p-2 rounded-lg transition-colors ${
              showGrid ? 'bg-crop-500 text-white' : 'bg-zinc-800 text-zinc-400'
            }`}
            title={showGrid ? 'Hide grid' : 'Show grid'}
          >
            {showGrid ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <SummaryCard
          icon={<Layers className="w-5 h-5" />}
          label="Primary Class"
          value={result.overall_prediction}
          subvalue={`${(result.overall_confidence * 100).toFixed(1)}% conf`}
          color="blue"
        />
        <SummaryCard
          icon={<Heart className="w-5 h-5" />}
          label="Avg Health"
          value={result.average_health_score.toFixed(3)}
          subvalue={getHealthCategory(result.average_health_score).label}
          color={getHealthCategory(result.average_health_score).color}
        />
        <SummaryCard
          icon={<Grid3X3 className="w-5 h-5" />}
          label="Grid Size"
          value={`${result.grid_size[0]}×${result.grid_size[1]}`}
          subvalue={`${result.total_cells} cells`}
          color="zinc"
        />
        <SummaryCard
          icon={<BarChart3 className="w-5 h-5" />}
          label="Classes Found"
          value={result.class_distribution.length}
          subvalue="unique types"
          color="purple"
        />
      </div>

      {/* View Mode Tabs */}
      <div className="flex gap-2 p-1 bg-zinc-800/50 rounded-xl w-fit">
        {[
          { id: 'classification', label: 'Classification', icon: Layers },
          { id: 'health', label: 'Health', icon: Heart },
          { id: 'confidence', label: 'Confidence', icon: BarChart3 },
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setViewMode(id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              viewMode === id
                ? 'bg-crop-500 text-white'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Interactive Grid - 2 columns */}
        <div className="lg:col-span-2 space-y-4">
          <div className="gradient-border p-4">
            <InteractiveGrid
              predictions={result.grid_predictions}
              gridSize={result.grid_size}
              viewMode={viewMode}
              enabledClasses={enabledClasses}
              showGrid={showGrid}
              selectedCell={selectedCell}
              onCellSelect={setSelectedCell}
              imageUrl={imageUrl}
            />
          </div>
          
          {/* Legend */}
          <GridLegend
            classes={result.class_distribution}
            enabledClasses={enabledClasses}
            onToggleClass={toggleClass}
          />
        </div>

        {/* Right Sidebar */}
        <div className="space-y-4">
          {/* Cell Details */}
          <CellDetails
            cell={selectedCell}
            classHealthStats={classHealthStats}
          />
          
          {/* Health Panel */}
          <HealthPanel
            averageHealth={result.average_health_score}
            classHealthStats={classHealthStats}
          />
          
          {/* Class Distribution */}
          <ClassDistribution
            distribution={result.class_distribution}
            enabledClasses={enabledClasses}
            onToggleClass={toggleClass}
          />
        </div>
      </div>
    </div>
  )
}

function SummaryCard({ icon, label, value, subvalue, color }) {
  const colorClasses = {
    blue: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    green: 'bg-crop-500/10 text-crop-400 border-crop-500/20',
    yellow: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    red: 'bg-red-500/10 text-red-400 border-red-500/20',
    zinc: 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20',
    purple: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  }

  return (
    <div className={`p-4 rounded-xl border ${colorClasses[color] || colorClasses.zinc}`}>
      <div className="flex items-center gap-2 mb-2 opacity-70">
        {icon}
        <span className="text-xs uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs opacity-60">{subvalue}</div>
    </div>
  )
}

function getHealthCategory(score) {
  if (score >= 0.7) return { label: 'Good', color: 'green' }
  if (score >= 0.4) return { label: 'Moderate', color: 'yellow' }
  return { label: 'Poor', color: 'red' }
}

