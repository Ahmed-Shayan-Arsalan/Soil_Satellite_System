import React, { memo, useCallback } from 'react'
import { Eye, EyeOff } from 'lucide-react'

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

const GridLegend = memo(function GridLegend({ classes, enabledClasses, onToggleClass }) {
  return (
    <div className="flex flex-wrap gap-2 p-3 bg-zinc-900/50 rounded-lg">
      {classes.map(item => {
        const isEnabled = enabledClasses.has(item.class_name)
        
        return (
          <button
            key={item.class_name}
            onClick={() => onToggleClass(item.class_name)}
            className={`legend-item flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all ${
              isEnabled 
                ? 'bg-zinc-800 hover:bg-zinc-700' 
                : 'bg-zinc-900 opacity-50 hover:opacity-70'
            }`}
          >
            <div 
              className="w-3 h-3 rounded-full"
              style={{ 
                backgroundColor: CLASS_COLORS[item.class_name],
                opacity: isEnabled ? 1 : 0.3
              }}
            />
            <span className={isEnabled ? 'text-zinc-200' : 'text-zinc-500'}>
              {item.class_name}
            </span>
            <span className="text-xs text-zinc-500">
              {item.percentage.toFixed(0)}%
            </span>
            {isEnabled ? (
              <Eye className="w-3 h-3 text-zinc-500" />
            ) : (
              <EyeOff className="w-3 h-3 text-zinc-600" />
            )}
          </button>
        )
      })}
    </div>
  )
})

export default GridLegend

