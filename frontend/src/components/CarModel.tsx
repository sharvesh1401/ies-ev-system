import { useRef, useMemo } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html, Environment, ContactShadows } from '@react-three/drei'
import * as THREE from 'three'

/* ═══════════════════════════════════════════════════════
   Detailed EV Crossover — procedural, matching reference
   Silver-blue metallic body, panoramic glass, LED DRLs
   ═══════════════════════════════════════════════════════ */
function EVCrossover() {
  const groupRef = useRef<THREE.Group>(null!)

  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.12
    }
  })

  // Materials
  const bodyMat = useMemo(() => new THREE.MeshPhysicalMaterial({
    color: '#b8c5d0',
    metalness: 0.85,
    roughness: 0.15,
    clearcoat: 1.0,
    clearcoatRoughness: 0.05,
    reflectivity: 0.9,
    envMapIntensity: 1.2,
  }), [])

  const glassMat = useMemo(() => new THREE.MeshPhysicalMaterial({
    color: '#0a1a2e',
    metalness: 0.1,
    roughness: 0.05,
    transmission: 0.6,
    thickness: 0.5,
    transparent: true,
    opacity: 0.75,
    ior: 1.5,
  }), [])

  const darkTrimMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#1a1a1a',
    metalness: 0.3,
    roughness: 0.6,
  }), [])

  const chromeMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#ddd',
    metalness: 0.95,
    roughness: 0.05,
  }), [])

  const tireMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#1a1a1a',
    roughness: 0.95,
    metalness: 0.0,
  }), [])

  const rimMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#333',
    metalness: 0.9,
    roughness: 0.1,
  }), [])

  const drlMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#fff',
    emissive: '#a0d4f0',
    emissiveIntensity: 3,
  }), [])

  const tailMat = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#ff2020',
    emissive: '#cc0000',
    emissiveIntensity: 2.0,
  }), [])

  return (
    <group ref={groupRef} position={[0, -0.15, 0]} scale={[0.9, 0.9, 0.9]}>

      {/* ── Lower body / chassis ──────────── */}
      <mesh position={[0, 0.32, 0]} castShadow material={bodyMat}>
        <boxGeometry args={[4.6, 0.48, 1.85]} />
      </mesh>

      {/* ── Rocker panels (sides) ──────────── */}
      <mesh position={[0, 0.12, 0.94]} castShadow material={darkTrimMat}>
        <boxGeometry args={[3.8, 0.12, 0.08]} />
      </mesh>
      <mesh position={[0, 0.12, -0.94]} castShadow material={darkTrimMat}>
        <boxGeometry args={[3.8, 0.12, 0.08]} />
      </mesh>

      {/* ── Wheel arches ──────────────────── */}
      {[
        [-1.38, 0.35, 0.93],
        [-1.38, 0.35, -0.93],
        [1.38, 0.35, 0.93],
        [1.38, 0.35, -0.93],
      ].map((pos, i) => (
        <mesh key={`arch-${i}`} position={pos as [number, number, number]} castShadow material={darkTrimMat}>
          <boxGeometry args={[0.75, 0.35, 0.08]} />
        </mesh>
      ))}

      {/* ── Upper body / cabin ────────────── */}
      <mesh position={[0.2, 0.78, 0]} castShadow material={bodyMat}>
        <boxGeometry args={[2.6, 0.5, 1.72]} />
      </mesh>

      {/* ── Roof ─────────────────────────── */}
      <mesh position={[0.15, 1.06, 0]} castShadow material={bodyMat}>
        <boxGeometry args={[2.3, 0.06, 1.6]} />
      </mesh>

      {/* ── Panoramic glass roof ─────────── */}
      <mesh position={[0.1, 1.04, 0]} material={glassMat}>
        <boxGeometry args={[1.8, 0.03, 1.3]} />
      </mesh>

      {/* ── Windshield front (angled) ────── */}
      <mesh position={[-0.95, 0.82, 0]} rotation={[0, 0, 0.42]} castShadow material={glassMat}>
        <boxGeometry args={[0.72, 0.48, 1.58]} />
      </mesh>

      {/* ── Rear window (angled) ─────────── */}
      <mesh position={[1.35, 0.82, 0]} rotation={[0, 0, -0.35]} castShadow material={glassMat}>
        <boxGeometry args={[0.55, 0.45, 1.54]} />
      </mesh>

      {/* ── Side windows left ────────────── */}
      <mesh position={[0.2, 0.82, 0.87]} material={glassMat}>
        <boxGeometry args={[1.8, 0.38, 0.02]} />
      </mesh>
      {/* ── Side windows right ───────────── */}
      <mesh position={[0.2, 0.82, -0.87]} material={glassMat}>
        <boxGeometry args={[1.8, 0.38, 0.02]} />
      </mesh>

      {/* ── A-pillar left ────────────────── */}
      <mesh position={[-0.75, 0.82, 0.82]} rotation={[0, 0, 0.3]} material={bodyMat}>
        <boxGeometry args={[0.08, 0.52, 0.12]} />
      </mesh>
      {/* ── A-pillar right ───────────────── */}
      <mesh position={[-0.75, 0.82, -0.82]} rotation={[0, 0, 0.3]} material={bodyMat}>
        <boxGeometry args={[0.08, 0.52, 0.12]} />
      </mesh>

      {/* ── Hood (sculpted, long) ────────── */}
      <mesh position={[-1.7, 0.5, 0]} castShadow material={bodyMat}>
        <boxGeometry args={[1.2, 0.22, 1.78]} />
      </mesh>
      {/* Hood slope */}
      <mesh position={[-2.15, 0.42, 0]} rotation={[0, 0, -0.1]} castShadow material={bodyMat}>
        <boxGeometry args={[0.35, 0.16, 1.7]} />
      </mesh>

      {/* ── Trunk / rear ────────────────── */}
      <mesh position={[1.65, 0.5, 0]} castShadow material={bodyMat}>
        <boxGeometry args={[0.9, 0.25, 1.78]} />
      </mesh>
      {/* Rear spoiler lip */}
      <mesh position={[1.55, 1.06, 0]} material={darkTrimMat}>
        <boxGeometry args={[0.4, 0.03, 1.4]} />
      </mesh>

      {/* ── Front bumper ────────────────── */}
      <mesh position={[-2.28, 0.25, 0]} castShadow material={bodyMat}>
        <boxGeometry args={[0.12, 0.32, 1.82]} />
      </mesh>
      {/* Lower front grille (dark) */}
      <mesh position={[-2.32, 0.18, 0]} material={darkTrimMat}>
        <boxGeometry args={[0.06, 0.15, 1.5]} />
      </mesh>

      {/* ── Rear bumper ─────────────────── */}
      <mesh position={[2.12, 0.25, 0]} castShadow material={bodyMat}>
        <boxGeometry args={[0.1, 0.3, 1.8]} />
      </mesh>
      {/* Rear diffuser */}
      <mesh position={[2.15, 0.12, 0]} material={darkTrimMat}>
        <boxGeometry args={[0.06, 0.1, 1.4]} />
      </mesh>

      {/* ── Chrome trim strip ───────────── */}
      <mesh position={[0, 0.56, 0.93]} material={chromeMat}>
        <boxGeometry args={[4.0, 0.015, 0.015]} />
      </mesh>
      <mesh position={[0, 0.56, -0.93]} material={chromeMat}>
        <boxGeometry args={[4.0, 0.015, 0.015]} />
      </mesh>

      {/* ══════ WHEELS ══════ */}
      {[
        [-1.38, 0.08, 0.95],
        [-1.38, 0.08, -0.95],
        [1.38, 0.08, 0.95],
        [1.38, 0.08, -0.95],
      ].map((pos, i) => (
        <group key={`wheel-${i}`} position={pos as [number, number, number]}>
          {/* Tire */}
          <mesh rotation={[Math.PI / 2, 0, 0]} material={tireMat}>
            <torusGeometry args={[0.32, 0.12, 16, 32]} />
          </mesh>
          {/* Rim outer */}
          <mesh rotation={[Math.PI / 2, 0, 0]} material={rimMat}>
            <cylinderGeometry args={[0.24, 0.24, 0.18, 24]} />
          </mesh>
          {/* Rim hub */}
          <mesh rotation={[Math.PI / 2, 0, 0]} material={chromeMat}>
            <cylinderGeometry args={[0.06, 0.06, 0.2, 12]} />
          </mesh>
          {/* Spokes (6 radial bars) */}
          {Array.from({ length: 6 }, (_, s) => (
            <mesh
              key={s}
              rotation={[Math.PI / 2, 0, (s * Math.PI) / 3]}
              position={[0, 0, 0]}
              material={rimMat}
            >
              <boxGeometry args={[0.04, 0.19, 0.4]} />
            </mesh>
          ))}
          {/* Brake disc */}
          <mesh rotation={[Math.PI / 2, 0, 0]} material={chromeMat}>
            <cylinderGeometry args={[0.18, 0.18, 0.04, 24]} />
          </mesh>
        </group>
      ))}

      {/* ══════ HEADLIGHTS ══════ */}
      {/* Left DRL strip */}
      <mesh position={[-2.28, 0.42, 0.65]} material={drlMat}>
        <boxGeometry args={[0.04, 0.035, 0.35]} />
      </mesh>
      {/* Right DRL strip */}
      <mesh position={[-2.28, 0.42, -0.65]} material={drlMat}>
        <boxGeometry args={[0.04, 0.035, 0.35]} />
      </mesh>
      {/* Left lower DRL */}
      <mesh position={[-2.32, 0.15, 0.7]} material={drlMat}>
        <boxGeometry args={[0.03, 0.02, 0.25]} />
      </mesh>
      {/* Right lower DRL */}
      <mesh position={[-2.32, 0.15, -0.7]} material={drlMat}>
        <boxGeometry args={[0.03, 0.02, 0.25]} />
      </mesh>

      {/* ══════ TAILLIGHTS ══════ */}
      {/* Full-width rear taillight bar */}
      <mesh position={[2.14, 0.45, 0]} material={tailMat}>
        <boxGeometry args={[0.03, 0.04, 1.4]} />
      </mesh>
      {/* Left tail */}
      <mesh position={[2.14, 0.43, 0.75]} material={tailMat}>
        <boxGeometry args={[0.04, 0.06, 0.2]} />
      </mesh>
      {/* Right tail */}
      <mesh position={[2.14, 0.43, -0.75]} material={tailMat}>
        <boxGeometry args={[0.04, 0.06, 0.2]} />
      </mesh>

      {/* ── Side mirrors ─────────────────── */}
      <mesh position={[-0.65, 0.72, 0.98]} material={bodyMat}>
        <boxGeometry args={[0.15, 0.08, 0.08]} />
      </mesh>
      <mesh position={[-0.65, 0.72, -0.98]} material={bodyMat}>
        <boxGeometry args={[0.15, 0.08, 0.08]} />
      </mesh>

      {/* ═══ Floating Labels ═══ */}
      <Html position={[-1.6, 1.3, 0.3]} distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div className="glass-ivory px-4 py-2 rounded-xl whitespace-nowrap shadow-lg" style={{ animation: 'fadeIn 1s ease-out' }}>
          <div className="text-[10px] text-brand-primary font-bold uppercase tracking-wider">Front Motor</div>
          <div className="text-xs font-mono text-surface-900 font-semibold">245 kW • OK</div>
        </div>
      </Html>

      <Html position={[0.5, -0.1, -0.7]} distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div className="glass-ivory px-4 py-2 rounded-xl whitespace-nowrap shadow-lg" style={{ animation: 'fadeIn 1.5s ease-out' }}>
          <div className="text-[10px] text-brand-secondary font-bold uppercase tracking-wider">Battery Pack</div>
          <div className="text-xs font-mono text-surface-900 font-semibold">75 kWh • 29°C</div>
        </div>
      </Html>

      <Html position={[1.8, 0.8, 0.5]} distanceFactor={5} style={{ pointerEvents: 'none' }}>
        <div className="glass-ivory px-4 py-2 rounded-xl whitespace-nowrap shadow-lg" style={{ animation: 'fadeIn 2s ease-out' }}>
          <div className="text-[10px] text-accent-success font-bold uppercase tracking-wider">Regen Brake</div>
          <div className="text-xs font-mono text-surface-900 font-semibold">Active</div>
        </div>
      </Html>
    </group>
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
        camera={{ position: [5, 2.8, 5], fov: 38 }}
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

        <EVCrossover />

        <ContactShadows
          position={[0, -0.43, 0]}
          opacity={0.3}
          scale={14}
          blur={2.5}
          far={5}
          color="#2c3e50"
        />

        <OrbitControls
          enableZoom={true}
          enablePan={false}
          minDistance={4}
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
