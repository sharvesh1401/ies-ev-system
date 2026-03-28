import { useState, useEffect } from 'react'

interface WindowSize {
  width: number
  height: number
}

export default function useWindowSize(): WindowSize & {
  isMobile: boolean
  isTablet: boolean
  isDesktop: boolean
} {
  const [windowSize, setWindowSize] = useState<WindowSize>({
    width: typeof window !== 'undefined' ? window.innerWidth : 1024,
    height: typeof window !== 'undefined' ? window.innerHeight : 768,
  })

  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout>

    const handleResize = () => {
      clearTimeout(timeoutId)
      timeoutId = setTimeout(() => {
        setWindowSize({
          width: window.innerWidth,
          height: window.innerHeight,
        })
      }, 150)
    }

    window.addEventListener('resize', handleResize)

    return () => {
      clearTimeout(timeoutId)
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  return {
    ...windowSize,
    isMobile: windowSize.width <= 767,
    isTablet: windowSize.width >= 768 && windowSize.width <= 1023,
    isDesktop: windowSize.width >= 1024,
  }
}
