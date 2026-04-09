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

  // Map color prop to Stitch token classes
  const iconColor =
    color === 'green' ? 'text-secondary-container' : color === 'purple' ? 'text-on-surface-variant' : 'text-primary'
  const iconBg =
    color === 'green' ? 'bg-secondary-container/10' : color === 'purple' ? 'bg-surface-container-highest/20' : 'bg-primary/10'
  const spinnerBorder =
    color === 'green' ? 'border-secondary-container/30 border-t-secondary-container' : color === 'purple' ? 'border-on-surface-variant/30 border-t-on-surface-variant' : 'border-primary/30 border-t-primary'
  const focusBorder =
    color === 'green' ? 'focus:border-secondary-container/50' : color === 'purple' ? 'focus:border-outline-variant/50' : 'focus:border-primary/50'

  return (
    <div ref={wrapperRef} className={`relative w-full ${className}`}>
      {variant === 'glass' ? (
        <div className={`bg-surface-container-lowest border border-outline-variant/20 p-1 flex items-center gap-2 transition-all focus-within:border-primary/50 rounded-xl`}>
          <div className={`w-10 h-10 rounded-lg ${iconBg} flex items-center justify-center shrink-0 ml-1`}>
            {loading ? (
              <div className={`w-4 h-4 rounded-full border-2 ${spinnerBorder} animate-spin`} />
            ) : (
              <span className={`material-symbols-outlined ${iconColor}`}>search</span>
            )}
          </div>
          <input
            className="flex-1 bg-transparent text-on-surface text-sm py-3 pr-4 outline-none placeholder:text-on-surface-variant/40 rounded-lg"
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
          <span className={`material-symbols-outlined absolute left-3 top-1/2 -translate-y-1/2 ${iconColor}/40 text-[18px]`}>
            {loading ? 'hourglass_empty' : 'search'}
          </span>
          <input
            className={`w-full bg-surface-container-lowest border border-outline-variant/20 text-on-surface text-sm pl-10 pr-4 py-2.5 outline-none ${focusBorder} focus:bg-surface-container transition-all rounded-lg`}
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
        <div className="absolute top-[calc(100%+8px)] left-0 right-0 bg-surface-container-high backdrop-blur-xl border border-outline-variant/20 shadow-2xl z-[2000] overflow-hidden rounded-xl">
          {results.map((r: GeoSearchResult) => (
            <div
              key={r.place_id}
              className="px-4 py-3 hover:bg-surface-container-highest cursor-pointer border-b border-outline-variant/10 last:border-0 transition-colors flex items-center gap-3"
              onClick={() => {
                const title = r.display_name.split(',')[0]
                setQuery(title)
                setIsOpen(false)
                if (onSelect) onSelect(r)
              }}
            >
              <span className={`material-symbols-outlined ${iconColor}/70 text-lg shrink-0`}>location_on</span>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-on-surface truncate">
                  {r.display_name.split(',')[0]}
                </p>
                <p className="text-[10px] text-on-surface-variant/50 truncate mt-0.5">
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
