import React from 'react'
import { Leaf, Satellite, Activity } from 'lucide-react'

export default function Header() {
  return (
    <header className="border-b border-zinc-800/50 backdrop-blur-sm sticky top-0 z-50 bg-zinc-950/80">
      <div className="container mx-auto px-4 py-4 max-w-7xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-crop-500 blur-xl opacity-30 animate-pulse-slow"></div>
              <Leaf className="w-8 h-8 text-crop-500 relative" />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">
                Soil<span className="text-crop-500">Vision</span>
              </h1>
              <p className="text-xs text-zinc-500">Geospatial Analysis Platform</p>
            </div>
          </div>
          
          <div className="flex items-center gap-6">
            <div className="hidden md:flex items-center gap-2 text-xs text-zinc-500">
              <Satellite className="w-4 h-4" />
              <span>Sentinel-2 Compatible</span>
            </div>
            <div className="hidden md:flex items-center gap-2 text-xs text-zinc-500">
              <Activity className="w-4 h-4 text-crop-500" />
              <span>Tempes Agricultural A.I systems</span>
            </div>
            <div className="px-3 py-1 bg-zinc-800/50 rounded-full text-xs text-zinc-400">
              v1.0
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}

