import { useRef, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html, Environment, ContactShadows, useGLTF, Center } from '@react-three/drei'
import * as THREE from 'three'

/* ═══════════════════════════════════════════════
   Load .glb car model from /models/car.glb
   ═══════════════════════════════════════════════ */
function CarGLB() {
  const groupRef = useRef<THREE.Group>(null!)
  const { scene } = useGLTF('/models/car.glb')

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.12
    }
  })

  return (
    <group ref={groupRef}>
      <Center>
        <primitive
          object={scene}
          scale={1}
          castShadow
          receiveShadow
        />
      </Center>

      {/* ═══ Floating Labels ═══ */}
      <Html position={[-1.2, 1.2, 0.3]} distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div className="glass-ivory px-4 py-2  whitespace-nowrap shadow-lg" style={{ animation: 'fadeIn 1s ease-out' }}>
          <div className="text-[10px] text-brand-primary font-bold uppercase tracking-wider">Front Motor</div>
          <div className="text-xs font-mono text-surface-900 font-semibold">245 kW • OK</div>
        </div>
      </Html>

      <Html position={[0.5, -0.3, -0.7]} distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div className="glass-ivory px-4 py-2  whitespace-nowrap shadow-lg" style={{ animation: 'fadeIn 1.5s ease-out' }}>
          <div className="text-[10px] text-brand-secondary font-bold uppercase tracking-wider">Battery Pack</div>
          <div className="text-xs font-mono text-surface-900 font-semibold">75 kWh • 29°C</div>
        </div>
      </Html>

      <Html position={[1.3, 0.8, 0.5]} distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div className="glass-ivory px-4 py-2  whitespace-nowrap shadow-lg" style={{ animation: 'fadeIn 2s ease-out' }}>
          <div className="text-[10px] text-accent-success font-bold uppercase tracking-wider">Regen Brake</div>
          <div className="text-xs font-mono text-surface-900 font-semibold">Active</div>
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
export default function CarModel() {
  return (
    <div className="w-full h-full relative">
      {/* Subtle ice glow behind car */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-[70%] h-[60%] bg-brand-primary/5 rounded-full blur-[80px]" />
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
          <CarGLB />
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
useGLTF.preload('/models/car.glb')

