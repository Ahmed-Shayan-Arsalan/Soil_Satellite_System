import React, { memo } from 'react'
import { MapPin, Layers, Heart, Target, Info } from 'lucide-react'

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

const CLASS_DESCRIPTIONS = {
  'AnnualCrop': 'Cultivated fields with seasonal crops like wheat, corn, or vegetables',
  'Forest': 'Dense tree coverage, natural or managed woodland areas',
  'HerbaceousVegetation': 'Grasslands, meadows, and non-woody vegetation',
  'Highway': 'Roads, highways, and paved transportation infrastructure',
  'Industrial': 'Factories, warehouses, and industrial facilities',
  'Pasture': 'Managed grassland for grazing livestock',
  'PermanentCrop': 'Orchards, vineyards, and perennial crop cultivation',
  'Residential': 'Housing areas, urban residential zones',
  'River': 'Waterways, streams, and rivers',
  'SeaLake': 'Large water bodies, lakes, and coastal areas',
}

const CellDetails = memo(function CellDetails({ cell, classHealthStats }) {
  if (!cell) {
    return (
      <div className="gradient-border p-4">
        <div className="flex items-center gap-2 text-zinc-500">
          <Info className="w-4 h-4" />
          <span className="text-sm">Click a grid cell to see details</span>
        </div>
      </div>
    )
  }
  
  const healthCategory = getHealthCategory(cell.health_score)
  const classStats = classHealthStats?.find(s => s.name === cell.class_name)
  
  return (
    <div className="gradient-border p-4 space-y-4 animate-scale-in">
      {/* Header */}
      <div className="flex items-start gap-3">
        <div 
          className="w-12 h-12 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: CLASS_COLORS[cell.class_name] + '30' }}
        >
          <Layers 
            className="w-6 h-6" 
            style={{ color: CLASS_COLORS[cell.class_name] }}
          />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-lg">{cell.class_name}</h3>
          <p className="text-xs text-zinc-400">{CLASS_DESCRIPTIONS[cell.class_name]}</p>
        </div>
      </div>
      
      {/* Position */}
      <div className="flex items-center gap-2 text-sm text-zinc-400">
        <MapPin className="w-4 h-4" />
        <span>Grid Position: Row {cell.row + 1}, Column {cell.col + 1}</span>
      </div>
      
      {/* Metrics */}
      <div className="grid grid-cols-2 gap-3">
        {/* Confidence */}
        <div className="p-3 bg-zinc-800/50 rounded-lg">
          <div className="flex items-center gap-2 text-xs text-zinc-400 mb-1">
            <Target className="w-3 h-3" />
            Confidence
          </div>
          <div className="text-xl font-bold font-mono text-blue-400">
            {(cell.confidence * 100).toFixed(1)}%
          </div>
          <ConfidenceBar value={cell.confidence} />
        </div>
        
        {/* Health */}
        <div className="p-3 bg-zinc-800/50 rounded-lg">
          <div className="flex items-center gap-2 text-xs text-zinc-400 mb-1">
            <Heart className="w-3 h-3" />
            Health Score
          </div>
          <div className={`text-xl font-bold font-mono ${healthCategory.textColor}`}>
            {(cell.health_score * 100).toFixed(1)}%
          </div>
          <HealthBar value={cell.health_score} />
        </div>
      </div>
      
      {/* Health Status */}
      <div className={`p-3 rounded-lg ${healthCategory.bgColor}`}>
        <div className="flex items-center gap-2">
          {healthCategory.icon}
          <div>
            <div className={`font-medium ${healthCategory.textColor}`}>
              {healthCategory.label}
            </div>
            <div className="text-xs text-zinc-400">
              {healthCategory.description}
            </div>
          </div>
        </div>
      </div>
      
      {/* Comparison to class average */}
      {classStats && (
        <div className="text-xs text-zinc-500 p-2 bg-zinc-800/30 rounded">
          <span className="text-zinc-400">vs {cell.class_name} average: </span>
          <span className={cell.health_score >= classStats.avg ? 'text-crop-400' : 'text-red-400'}>
            {cell.health_score >= classStats.avg ? '+' : ''}
            {((cell.health_score - classStats.avg) * 100).toFixed(1)}%
          </span>
        </div>
      )}
    </div>
  )
})

function ConfidenceBar({ value }) {
  return (
    <div className="h-1.5 bg-zinc-700 rounded-full mt-2 overflow-hidden">
      <div 
        className="h-full bg-gradient-to-r from-blue-600 to-blue-400 rounded-full transition-all"
        style={{ width: `${value * 100}%` }}
      />
    </div>
  )
}

function HealthBar({ value }) {
  const color = value >= 0.7 ? '#22c55e' : value >= 0.4 ? '#eab308' : '#ef4444'
  return (
    <div className="h-1.5 bg-zinc-700 rounded-full mt-2 overflow-hidden">
      <div 
        className="h-full rounded-full transition-all"
        style={{ width: `${value * 100}%`, backgroundColor: color }}
      />
    </div>
  )
}

function getHealthCategory(score) {
  if (score >= 0.7) {
    return {
      label: 'Good Health',
      description: 'Vegetation is thriving in this area',
      textColor: 'text-crop-400',
      bgColor: 'bg-crop-500/10',
      icon: <Heart className="w-4 h-4 text-crop-400" fill="currentColor" />
    }
  }
  if (score >= 0.4) {
    return {
      label: 'Moderate Health',
      description: 'Some stress indicators present',
      textColor: 'text-yellow-400',
      bgColor: 'bg-yellow-500/10',
      icon: <Heart className="w-4 h-4 text-yellow-400" />
    }
  }
  return {
    label: 'Poor Health',
    description: 'Vegetation shows signs of stress',
    textColor: 'text-red-400',
    bgColor: 'bg-red-500/10',
    icon: <Heart className="w-4 h-4 text-red-400" />
  }
}

export default CellDetails

