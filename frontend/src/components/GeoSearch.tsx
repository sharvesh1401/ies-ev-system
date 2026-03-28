import { useState, useEffect, useRef } from 'react'

export interface GeoSearchResult {
  place_id: number
  lat: string
  lon: string
  display_name: string
  address: any
}

interface GeoSearchProps {
  placeholder?: string
  defaultValue?: string
  color?: 'blue' | 'green' | 'purple'
  variant?: 'glass' | 'solid'
  onSelect?: (result: GeoSearchResult) => void
  className?: string
}

export default function GeoSearch({
  placeholder = 'Search places, addresses...',
  defaultValue = '',
  color = 'blue',
  variant = 'solid',
  onSelect,
  className = '',
}: GeoSearchProps) {
  const [query, setQuery] = useState(defaultValue)
  const [results, setResults] = useState<GeoSearchResult[]>([])
  const [isOpen, setIsOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const wrapperRef = useRef<HTMLDivElement>(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    if (!query || query === defaultValue) {
      setResults([])
      return
    }
    setLoading(true)
    const timer = setTimeout(() => {
      fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5&addressdetails=1`)
        .then((r) => r.json())
        .then((data) => {
          setResults(data || [])
          setLoading(false)
        })
        .catch(() => setLoading(false))
    }, 500)
    
    return () => {
      clearTimeout(timer)
      setLoading(false)
    }
  }, [query])

  const glowColor =
    color === 'blue' ? 'neon-blue' : color === 'green' ? 'neon-green' : 'neon-purple'

  return (
    <div ref={wrapperRef} className={`relative w-full ${className}`}>
      {variant === 'glass' ? (
        <div className={`glass-dark p-1 flex items-center gap-2 border border-${glowColor}/20 transition-all focus-within:border-${glowColor}/50 focus-within:shadow-[0_0_15px_rgba(0,180,216,0.15)]`}>
          <div className={`w-10 h-10 rounded-lg bg-${glowColor}/10 flex items-center justify-center shrink-0 ml-1`}>
            {loading ? (
              <div className={`w-4 h-4 rounded-full border-2 border-${glowColor}/30 border-t-${glowColor} animate-spin`} />
            ) : (
              <span className={`material-symbols-outlined text-${glowColor}`}>search</span>
            )}
          </div>
          <input
            className="flex-1 bg-transparent text-surface-900 text-sm py-3 pr-4 outline-none placeholder:text-surface-800/40 rounded-lg"
            value={query}
            placeholder={placeholder}
            aria-label={placeholder}
            aria-expanded={isOpen}
            onChange={(e) => {
              setQuery(e.target.value)
              setIsOpen(true)
            }}
            onFocus={() => {
              if (results.length > 0) setIsOpen(true)
            }}
            type="text"
          />
        </div>
      ) : (
        <div className="relative">
          <span className="material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 text-surface-800/40 text-[18px]">
            {loading ? 'hourglass_empty' : 'search'}
          </span>
          <input
            className={`w-full bg-surface-200/50 border border-white/10 text-surface-900 text-sm pl-10 pr-4 py-2.5 outline-none focus:border-${glowColor}/50 focus:bg-surface-200/80 transition-all rounded-lg`}
            value={query}
            placeholder={placeholder}
            aria-label={placeholder}
            aria-expanded={isOpen}
            onChange={(e) => {
              setQuery(e.target.value)
              setIsOpen(true)
            }}
            onFocus={() => {
              if (results.length > 0) setIsOpen(true)
            }}
            type="text"
          />
        </div>
      )}

      {/* Dropdown Results */}
      {isOpen && results.length > 0 && (
        <div className="absolute top-[calc(100%+8px)] left-0 right-0 bg-surface-100/95 backdrop-blur-xl border border-white/10 shadow-2xl z-[2000] overflow-hidden rounded-xl">
          {results.map((r: GeoSearchResult) => (
            <div
              key={r.place_id}
              className="px-4 py-3 hover:bg-surface-200/80 cursor-pointer border-b border-white/5 last:border-0 transition-colors flex items-center gap-3"
              onClick={() => {
                const title = r.display_name.split(',')[0]
                setQuery(title)
                setIsOpen(false)
                if (onSelect) onSelect(r)
              }}
            >
              <span className={`material-symbols-outlined text-${glowColor}/70 text-lg shrink-0`}>location_on</span>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-surface-900 truncate">
                  {r.display_name.split(',')[0]}
                </p>
                <p className="text-[10px] text-surface-800/50 truncate mt-0.5">
                  {r.display_name}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
