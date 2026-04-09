import { useRef, Suspense, useEffect } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html, Environment, ContactShadows, useGLTF } from '@react-three/drei'
import * as THREE from 'three'

/* ═══════════════════════════════════════════════
   Per-model config overrides for centering/scaling
   ═══════════════════════════════════════════════ */
const MODEL_CONFIG: Record<string, { targetSize: number; yOffset: number; rotationYOffset: number }> = {
  '/models/car.glb':      { targetSize: 4.5, yOffset: -0.5, rotationYOffset: 0 },
  '/models/commuter.glb': { targetSize: 4.5, yOffset: -0.5, rotationYOffset: 0 },
  '/models/cargo.glb':    { targetSize: 4.0, yOffset: -0.4, rotationYOffset: 0 },
}

const DEFAULT_CONFIG = { targetSize: 4.5, yOffset: -0.5, rotationYOffset: 0 }

/* ═══════════════════════════════════════════════
   Load .glb car model from /models/car.glb
   ═══════════════════════════════════════════════ */
function CarGLB({ batteryKwh, tempC, modelPath, regenActive, maxPowerKw }: { batteryKwh: number; tempC: number; modelPath: string; regenActive: boolean; maxPowerKw: number }) {
  const groupRef = useRef<THREE.Group>(null!)
  const { scene } = useGLTF(modelPath)
  const cfg = MODEL_CONFIG[modelPath] ?? DEFAULT_CONFIG

  useEffect(() => {
    // 1. Reset
    scene.scale.setScalar(1)
    scene.position.set(0, 0, 0)
    scene.rotation.set(0, 0, 0)
    scene.updateMatrixWorld(true)

    // 2. Normalizing the scale from VISIBLE MESHES ONLY to ignore lights/cameras
    const box = new THREE.Box3()
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh && child.visible) {
        box.expandByObject(child)
      }
    })
    
    let size = box.getSize(new THREE.Vector3())
    let maxDim = Math.max(size.x, size.y, size.z)
    
    if (maxDim === 0 || maxDim === -Infinity || !isFinite(maxDim)) {
      box.setFromObject(scene)
      size = box.getSize(new THREE.Vector3())
      maxDim = Math.max(size.x, size.y, size.z)
    }
    
    if (maxDim > 0 && isFinite(maxDim)) {
      const scale = cfg.targetSize / maxDim
      scene.scale.setScalar(scale)
    }

    // 3. Center the model manually and align bottom
    scene.updateMatrixWorld(true)
    const scaledBox = new THREE.Box3()
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh && child.visible) {
        scaledBox.expandByObject(child)
      }
    })
    
    if (scaledBox.isEmpty()) {
      scaledBox.setFromObject(scene)
    }

    const center = scaledBox.getCenter(new THREE.Vector3())
    
    scene.position.x -= center.x
    scene.position.z -= center.z
    scene.position.y += (cfg.yOffset - scaledBox.min.y)

    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.castShadow = true
        child.receiveShadow = true
      }
    })
  }, [scene, cfg])

  useFrame(({ clock }) => {
    if (groupRef.current) {
      // Showroom-style oscillation: ±18° (0.32 rad), smooth and premium
      groupRef.current.rotation.y = Math.sin(clock.getElapsedTime() * 0.5) * 0.32
    }
  })

  return (
    <group ref={groupRef}>
      <primitive object={scene} />

      {/* ═══ Floating Labels ═══ */}
      <Html position={[-1.2, 1.2, 0.3]} distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div className="glass-ivory px-4 py-2  whitespace-nowrap shadow-lg" style={{ animation: 'fadeIn 1s ease-out' }}>
          <div className="text-[10px] text-brand-primary font-bold uppercase tracking-wider">Front Motor</div>
          <div className="text-xs font-mono text-surface-900 font-semibold">{maxPowerKw} kW • OK</div>
        </div>
      </Html>

      <Html position={[0.5, -0.3, -0.7]} distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div className="glass-ivory px-4 py-2  whitespace-nowrap shadow-lg" style={{ animation: 'fadeIn 1.5s ease-out' }}>
          <div className="text-[10px] text-brand-secondary font-bold uppercase tracking-wider">Battery Pack</div>
          <div className="text-xs font-mono text-surface-900 font-semibold">{batteryKwh} kWh • {tempC}°C</div>
        </div>
      </Html>

      <Html position={[1.3, 0.8, 0.5]} distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div className="glass-ivory px-4 py-2  whitespace-nowrap shadow-lg" style={{ animation: 'fadeIn 2s ease-out' }}>
          <div className={`text-[10px] font-bold uppercase tracking-wider ${regenActive ? 'text-accent-success' : 'text-surface-800/40'}`}>Regen Brake</div>
          <div className={`text-xs font-mono font-semibold ${regenActive ? 'text-surface-900' : 'text-surface-800/50'}`}>{regenActive ? 'Active' : 'Inactive'}</div>
        </div>
      </Html>
    </group>
  )
}

/* ──────── Loading fallback ──────── */
function Loader() {
  return (
    <Html center>
      <div className="flex flex-col items-center gap-3">
        <div className="animate-spin w-8 h-8 border-2 border-brand-primary/20 border-t-brand-primary rounded-full" />
        <span className="text-sm text-surface-800/50 font-medium">Loading 3D Model…</span>
      </div>
    </Html>
  )
}

/* ──────── Main Exported Component ──────── */
export default function CarModel({
  batteryKwh = 75,
  tempC = 29,
  glowColor = '#00E5CC',
  modelPath = '/models/car.glb',
  regenActive = true,
  maxPowerKw = 350,
}: {
  batteryKwh?: number
  tempC?: number
  glowColor?: string
  modelPath?: string
  regenActive?: boolean
  maxPowerKw?: number
}) {
  return (
    <div className="w-full h-full relative">
      {/* Vehicle-specific glow behind car */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div
          className="w-[70%] h-[60%] rounded-full blur-[80px] transition-all duration-700"
          style={{ background: `${glowColor}0D` }}
        />
      </div>

      <Canvas
        camera={{ position: [4, 2.5, 4], fov: 40 }}
        shadows
        style={{ background: 'transparent' }}
        gl={{ alpha: true, antialias: true, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.2 }}
      >
        <ambientLight intensity={0.5} color="#f5f5ff" />
        <directionalLight
          position={[6, 10, 6]}
          intensity={1.5}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          color="#fff"
        />
        <pointLight position={[-4, 3, -3]} intensity={0.4} color="#a0d4f0" />
        <pointLight position={[4, 2, 4]} intensity={0.3} color="#e0d6c8" />
        <spotLight position={[0, 8, 0]} angle={0.4} penumbra={0.6} intensity={0.5} color="#fff" />

        <Suspense fallback={<Loader />}>
          <CarGLB batteryKwh={batteryKwh} tempC={tempC} modelPath={modelPath} regenActive={regenActive} maxPowerKw={maxPowerKw} />
        </Suspense>

        <ContactShadows
          position={[0, -0.5, 0]}
          opacity={0.3}
          scale={14}
          blur={2.5}
          far={5}
          color="#2c3e50"
        />

        <OrbitControls
          enableZoom={true}
          enablePan={false}
          minDistance={3}
          maxDistance={14}
          minPolarAngle={Math.PI / 6}
          maxPolarAngle={Math.PI / 2.1}
          autoRotate={false}
          enableDamping
          dampingFactor={0.05}
        />
        <Environment preset="studio" />
      </Canvas>
    </div>
  )
}

// Preload the model
// Preload all models
useGLTF.preload('/models/car.glb')
useGLTF.preload('/models/commuter.glb')
useGLTF.preload('/models/cargo.glb')

