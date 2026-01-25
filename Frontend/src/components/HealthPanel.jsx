import React, { memo } from 'react'
import { Heart, TrendingUp, TrendingDown, Minus } from 'lucide-react'

const HealthPanel = memo(function HealthPanel({ averageHealth, classHealthStats }) {
  const category = getHealthCategory(averageHealth)
  
  return (
    <div className="gradient-border p-4 space-y-4">
      <h3 className="font-semibold flex items-center gap-2">
        <Heart className="w-4 h-4 text-crop-500" />
        Health Analysis
      </h3>
      
      {/* Overall Health Gauge */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-zinc-400">Overall Health Score</span>
          <span className={`font-mono font-bold ${category.textColor}`}>
            {(averageHealth * 100).toFixed(1)}%
          </span>
        </div>
        
        {/* Health Bar */}
        <div className="relative h-4 bg-zinc-800 rounded-full overflow-hidden">
          {/* Background gradient */}
          <div 
            className="absolute inset-0"
            style={{
              background: 'linear-gradient(to right, #ef4444 0%, #ef4444 40%, #eab308 40%, #eab308 70%, #22c55e 70%, #22c55e 100%)'
            }}
          />
          {/* Indicator */}
          <div 
            className="absolute top-0 bottom-0 w-1 bg-white shadow-lg transition-all duration-500"
            style={{ left: `${averageHealth * 100}%`, transform: 'translateX(-50%)' }}
          />
        </div>
        
        {/* Category Labels */}
        <div className="flex justify-between text-xs text-zinc-500">
          <span>Poor (0-40%)</span>
          <span>Moderate (40-70%)</span>
          <span>Good (70-100%)</span>
        </div>
      </div>
      
      {/* Health Status */}
      <div className={`flex items-center gap-3 p-3 rounded-lg ${category.bgColor}`}>
        {category.icon}
        <div>
          <div className={`font-semibold ${category.textColor}`}>{category.label}</div>
          <div className="text-xs text-zinc-400">{category.description}</div>
        </div>
      </div>
      
      {/* Per-Class Health Stats */}
      <div className="space-y-2">
        <div className="text-sm text-zinc-400 font-medium">Health by Class</div>
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {classHealthStats.map(stat => (
            <div 
              key={stat.name}
              className="flex items-center gap-2 p-2 bg-zinc-800/50 rounded-lg hover:bg-zinc-800 transition-colors"
            >
              <div className="flex-1 min-w-0">
                <div className="text-sm truncate">{stat.name}</div>
                <div className="text-xs text-zinc-500">{stat.count} cells</div>
              </div>
              <div className="text-right">
                <div className={`font-mono text-sm ${getHealthTextColor(stat.avg)}`}>
                  {(stat.avg * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-zinc-600">
                  {(stat.min * 100).toFixed(0)}-{(stat.max * 100).toFixed(0)}%
                </div>
              </div>
              {/* Mini health bar */}
              <div className="w-16 h-2 bg-zinc-700 rounded-full overflow-hidden">
                <div 
                  className="h-full rounded-full transition-all health-bar"
                  style={{
                    width: `${stat.avg * 100}%`,
                    backgroundColor: getHealthBarColor(stat.avg)
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
})

function getHealthCategory(score) {
  if (score >= 0.7) {
    return {
      label: 'Good Health',
      description: 'Vegetation is thriving',
      textColor: 'text-crop-400',
      bgColor: 'bg-crop-500/10',
      icon: <TrendingUp className="w-5 h-5 text-crop-400" />
    }
  }
  if (score >= 0.4) {
    return {
      label: 'Moderate Health',
      description: 'Some areas need attention',
      textColor: 'text-yellow-400',
      bgColor: 'bg-yellow-500/10',
      icon: <Minus className="w-5 h-5 text-yellow-400" />
    }
  }
  return {
    label: 'Poor Health',
    description: 'Vegetation is stressed',
    textColor: 'text-red-400',
    bgColor: 'bg-red-500/10',
    icon: <TrendingDown className="w-5 h-5 text-red-400" />
  }
}

function getHealthTextColor(score) {
  if (score >= 0.7) return 'text-crop-400'
  if (score >= 0.4) return 'text-yellow-400'
  return 'text-red-400'
}

function getHealthBarColor(score) {
  if (score >= 0.7) return '#22c55e'
  if (score >= 0.4) return '#eab308'
  return '#ef4444'
}

export default HealthPanel

