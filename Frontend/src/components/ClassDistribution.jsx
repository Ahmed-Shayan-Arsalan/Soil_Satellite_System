import React, { memo, useMemo } from 'react'
import { BarChart3 } from 'lucide-react'

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

const ClassDistribution = memo(function ClassDistribution({ distribution, enabledClasses, onToggleClass }) {
  const maxCount = Math.max(...distribution.map(d => d.cell_count))
  
  // Sort by count descending
  const sorted = [...distribution].sort((a, b) => b.cell_count - a.cell_count)
  
  return (
    <div className="gradient-border p-4 space-y-4">
      <h3 className="font-semibold flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-blue-400" />
        Class Distribution
      </h3>
      
      <div className="space-y-2">
        {sorted.map(item => {
          const isEnabled = enabledClasses.has(item.class_name)
          const barWidth = (item.cell_count / maxCount) * 100
          
          return (
            <button
              key={item.class_name}
              onClick={() => onToggleClass(item.class_name)}
              className={`w-full text-left p-2 rounded-lg transition-all hover:bg-zinc-800 ${
                isEnabled ? '' : 'opacity-40'
              }`}
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-sm"
                    style={{ backgroundColor: CLASS_COLORS[item.class_name] }}
                  />
                  <span className="text-sm truncate">{item.class_name}</span>
                </div>
                <span className="text-sm font-mono text-zinc-400">
                  {item.cell_count} ({item.percentage.toFixed(1)}%)
                </span>
              </div>
              
              {/* Bar */}
              <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${barWidth}%`,
                    backgroundColor: CLASS_COLORS[item.class_name],
                    opacity: isEnabled ? 1 : 0.3
                  }}
                />
              </div>
            </button>
          )
        })}
      </div>
      
      {/* Summary */}
      <div className="pt-2 border-t border-zinc-800 text-xs text-zinc-500">
        Click classes to toggle visibility on map
      </div>
    </div>
  )
})

export default ClassDistribution

