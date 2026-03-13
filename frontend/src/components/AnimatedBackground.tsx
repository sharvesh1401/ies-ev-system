import { useEffect, useRef } from 'react'

export default function AnimatedBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animationId: number
    let w = window.innerWidth
    let h = window.innerHeight
    canvas.width = w
    canvas.height = h

    const particles: { x: number; y: number; vx: number; vy: number; r: number }[] = []
    for (let i = 0; i < 50; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        r: Math.random() * 1.5 + 0.5,
      })
    }

    function animate() {
      ctx!.clearRect(0, 0, w, h)

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        const p1 = particles[i]
        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j]
          const dx = p1.x - p2.x
          const dy = p1.y - p2.y
          const dist = Math.sqrt(dx * dx + dy * dy)

          if (dist < 150) {
            ctx!.beginPath()
            ctx!.strokeStyle = `rgba(0,229,204,${((150 - dist) / 150) * 0.08})`
            ctx!.lineWidth = 0.5
            ctx!.moveTo(p1.x, p1.y)
            ctx!.lineTo(p2.x, p2.y)
            ctx!.stroke()
          }
        }

        // Draw particle
        ctx!.beginPath()
        ctx!.arc(p1.x, p1.y, p1.r, 0, Math.PI * 2)
        ctx!.fillStyle = 'rgba(0,229,204,0.5)'
        ctx!.fill()

        // Update position
        p1.x += p1.vx
        p1.y += p1.vy

        // Bounce off edges
        if (p1.x < 0 || p1.x > w) p1.vx *= -1
        if (p1.y < 0 || p1.y > h) p1.vy *= -1
      }

      animationId = requestAnimationFrame(animate)
    }

    animate()

    const handleResize = () => {
      w = window.innerWidth
      h = window.innerHeight
      canvas.width = w
      canvas.height = h
    }
    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        zIndex: 0,
        opacity: 0.3,
        pointerEvents: 'none',
      }}
    />
  )
}
