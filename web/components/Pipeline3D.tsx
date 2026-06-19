"use client";

// 运行可视化（WebGL）：把一次真实分析的「样本 → 编码器 → 融合 → p⁰ → 三状态 → 有界增强 → p*」
// 渲染成可旋转/缩放的高质感 3D 管线 —— 辉光后期、网格地面、景深雾、能量导管、仪表化三状态、HUD。
// 所有几何与读数均由该样本的 AnalyzeResult 数值驱动；换样本即重绘（父级 seq 触发 Scene 重挂载）。
// three 由本地打包（不依赖 CDN），标签用 drei <Html> 复用页面字体；不支持 WebGL 时回退 2D。

import { useEffect, useMemo, useRef, useState, Component, type ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html, Line, RoundedBox, Grid, Float } from "@react-three/drei";
import { EffectComposer, Bloom, Vignette } from "@react-three/postprocessing";
import * as THREE from "three";
import type { AnalyzeResult } from "@/lib/types";
import { EMOTION_LABELS, LABEL_ZH } from "@/lib/types";

type Vec = [number, number, number];

const EMO: Record<string, string> = {
  happy: "#f0b53e", sad: "#5e8fe0", angry: "#ef5a43", fear: "#9a7cf0",
  surprise: "#37c4d2", disgust: "#7fb755", neutral: "#aab2c4",
};
const MOD_ZH: Record<string, string> = { video: "视频", audio: "音频", text: "文本" };
const MOD_COLOR: Record<string, string> = { video: "#5e8fe0", audio: "#37c4d2", text: "#f0b53e" };
const ZH = LABEL_ZH;
const SPINE_Y = 0.6, BASE_Y = -3.0, BAR_SCALE = 2.7;
const ENC_X = -11, FUS_X = -6.5, P0_X = -2, ST_X = 4.6, PS_X = 10.2;
const encY = (i: number) => SPINE_Y + ([1.8, 0, -1.8][i] ?? 0);
// 点击聚焦：每个阶段的相机注视点(t)与机位(c)。
const FOCUS_AT: Record<string, { t: Vec; c: Vec }> = {
  enc0: { t: [ENC_X, encY(0), 0], c: [ENC_X + 4.5, encY(0) + 2.4, 7.5] },
  enc1: { t: [ENC_X, encY(1), 0], c: [ENC_X + 4.5, encY(1) + 1.4, 7.5] },
  enc2: { t: [ENC_X, encY(2), 0], c: [ENC_X + 4.5, encY(2) + 0.6, 7.5] },
  fusion: { t: [FUS_X, SPINE_Y, 0], c: [FUS_X + 2.5, SPINE_Y + 2.4, 8] },
  p0: { t: [P0_X, SPINE_Y - 1.0, 0], c: [P0_X + 1.6, SPINE_Y + 1.2, 8.5] },
  states: { t: [ST_X, 0, 0], c: [ST_X + 3, 2.2, 11.5] },
  pstar: { t: [PS_X, SPINE_Y - 1.0, 0], c: [PS_X - 1.6, SPINE_Y + 1.2, 8.5] },
};
const QUALITY_FIELDS: Record<string, [string, string][]> = {
  video: [["face_detection_rate", "人脸检出率"], ["brightness", "亮度"], ["blur", "清晰度(Lap)"], ["frame_loss_rate", "丢帧率"]],
  audio: [["snr_db", "信噪比(dB)"], ["voiced_ratio", "有声比例"], ["silence_ratio", "静音比例"]],
  text: [["text_length", "长度"], ["valid_token_ratio", "有效token比例"], ["low_freq_subword_ratio", "低频子词比例"]],
};

function topOf(d: Record<string, number>) {
  return EMOTION_LABELS.reduce((a, b) => ((d[b] ?? 0) > (d[a] ?? 0) ? b : a), EMOTION_LABELS[0]);
}
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

// 入场：从 0 平滑放大到 1（错峰 delay），让管线“搭建”出来。
function Appear({ position, delay = 0, children }: { position: Vec; delay?: number; children: ReactNode }) {
  const ref = useRef<THREE.Group>(null);
  const t = useRef(0);
  useFrame((_, dt) => {
    t.current += dt;
    const p = clamp01((t.current - delay) / 0.45);
    const s = p <= 0 ? 0.0001 : p * p * (3 - 2 * p);
    ref.current?.scale.setScalar(s);
  });
  return (
    <group ref={ref} position={position}>
      {children}
    </group>
  );
}

// 自适应相机：竖屏（手机）时自动后退 + 抬高 fov，保证整条横向管线完整入镜。
function Rig({ controls }: { controls: { current: any } }) {
  const { camera, size } = useThree();
  useEffect(() => {
    const aspect = size.width / Math.max(1, size.height);
    const cam = camera as THREE.PerspectiveCamera;
    cam.fov = aspect < 0.7 ? 54 : aspect < 1 ? 47 : aspect < 1.4 ? 43 : 42;
    const dist = aspect >= 1.4 ? 21 : 21 * Math.min(2.5, 1.5 / Math.max(aspect, 0.42));
    cam.position.set(0, aspect < 1 ? 5.5 : 5, dist);
    cam.updateProjectionMatrix();
    if (controls.current) {
      controls.current.target.set(0, 0.4, 0);
      controls.current.update();
    }
  }, [size.width, size.height, camera, controls]);
  return null;
}

// 聚焦控制：focus 非空时把相机与注视点平滑移到该阶段。
function FocusController({ focus, controls }: { focus: string | null; controls: { current: any } }) {
  const { camera } = useThree();
  const tt = useRef(new THREE.Vector3());
  const cc = useRef(new THREE.Vector3());
  useFrame((_, dt) => {
    if (!focus || !controls.current || !FOCUS_AT[focus]) return;
    const f = FOCUS_AT[focus];
    tt.current.set(f.t[0], f.t[1], f.t[2]);
    cc.current.set(f.c[0], f.c[1], f.c[2]);
    const k = Math.min(1, dt * 2.6);
    controls.current.target.lerp(tt.current, k);
    camera.position.lerp(cc.current, k);
    controls.current.update();
  });
  return null;
}

// 聚焦光环：标出当前查看的阶段。
function FocusRing({ pos }: { pos: Vec }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame((s) => {
    if (ref.current) {
      ref.current.rotation.z = s.clock.elapsedTime * 1.1;
      const k = 1 + Math.sin(s.clock.elapsedTime * 4) * 0.06;
      ref.current.scale.set(k, k, k);
    }
  });
  return (
    <mesh ref={ref} position={pos} rotation={[Math.PI / 2.2, 0, 0]}>
      <torusGeometry args={[1.55, 0.045, 12, 50]} />
      <meshStandardMaterial color="#aecbff" emissive="#aecbff" emissiveIntensity={1.5} toneMapped={false} transparent opacity={0.85} />
    </mesh>
  );
}

function Tag({ position, children, accent }: { position: Vec; children: ReactNode; accent?: string }) {
  return (
    <Html position={position} center distanceFactor={18} zIndexRange={[8, 0]} pointerEvents="none">
      <div className="p3d-tag" style={accent ? { borderColor: accent, color: "#eaf0ff" } : undefined}>
        {children}
      </div>
    </Html>
  );
}

// 圆台基座：让每个阶段“落地”、成组。
function Pad({ x, y = -3.12, r = 1.7, color = "#1b2740" }: { x: number; y?: number; r?: number; color?: string }) {
  return (
    <mesh position={[x, y, 0]} rotation={[0, 0, 0]}>
      <cylinderGeometry args={[r, r + 0.15, 0.16, 40]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.35} metalness={0.4} roughness={0.5} />
    </mesh>
  );
}

// 一根分布柱：圆角、辉光、从 0 生长；top 类脉冲。
function GlowBar({ z, height, color, top, base }: { z: number; height: number; color: string; top: boolean; base: number }) {
  const ref = useRef<THREE.Group>(null);
  const mat = useRef<THREE.MeshStandardMaterial>(null);
  const g = useRef(0);
  useFrame((s, dt) => {
    g.current = Math.min(1, g.current + dt * 1.5);
    const h = Math.max(0.001, g.current);
    if (ref.current) { ref.current.scale.y = h; ref.current.position.y = base + (height * h) / 2; }
    if (mat.current && top) mat.current.emissiveIntensity = 1.1 + Math.sin(s.clock.elapsedTime * 3) * 0.5;
  });
  return (
    <group ref={ref} position={[0, 0, z]}>
      <RoundedBox args={[0.5, height, 0.5]} radius={0.1} smoothness={3}>
        <meshStandardMaterial
          ref={mat} color={color} emissive={color}
          emissiveIntensity={top ? 1.3 : 0.5} metalness={0.3} roughness={0.35} toneMapped={false}
        />
      </RoundedBox>
    </group>
  );
}

function Histogram({ dist, x, highlight, title }: { dist: Record<string, number>; x: number; highlight?: string; title: string }) {
  const top = highlight ?? topOf(dist);
  return (
    <group position={[x, 0, 0]}>
      <Pad x={0} y={BASE_Y - 0.12} r={2.5} />
      {EMOTION_LABELS.map((lbl, i) => (
        <GlowBar key={lbl} z={(i - 3) * 0.62} height={Math.max(0.05, (dist[lbl] ?? 0) * BAR_SCALE)} color={EMO[lbl]} top={top === lbl} base={BASE_Y} />
      ))}
      <Tag position={[0, 1.55, 0]} accent={EMO[top]}>{title}</Tag>
    </group>
  );
}

// 能量导管：发光直线（按端点半径精确裁剪到两端模块表面，杜绝悬空/穿模）+ 流动光点。
function Conduit({ from, to, color = "#5a86f2", count = 4, r0 = 0, r1 = 0 }: { from: Vec; to: Vec; color?: string; count?: number; r0?: number; r1?: number }) {
  const { s, e, pts } = useMemo(() => {
    const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to);
    const dir = b.clone().sub(a); const L = Math.max(1e-3, dir.length()); dir.divideScalar(L);
    const ss = a.clone().add(dir.clone().multiplyScalar(r0));
    const ee = b.clone().sub(dir.clone().multiplyScalar(r1));
    return { s: ss, e: ee, pts: [[ss.x, ss.y, ss.z], [ee.x, ee.y, ee.z]] as Vec[] };
  }, [from, to, r0, r1]);
  const grp = useRef<THREE.Group>(null);
  useFrame((st) => {
    const t = st.clock.elapsedTime;
    grp.current?.children.forEach((c, i) => {
      const f = (t * 0.42 + i / count) % 1;
      c.position.lerpVectors(s, e, f);
    });
  });
  return (
    <group>
      <Line points={pts} color={color} lineWidth={2.4} transparent opacity={0.62} toneMapped={false} />
      <group ref={grp}>
        {Array.from({ length: count }).map((_, i) => (
          <mesh key={i}>
            <sphereGeometry args={[0.095, 10, 10]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={2.2} toneMapped={false} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

// 阶段枢纽（连接锚点）+ 细茎（枢纽 → 该阶段详细可视化）。
function Hub({ pos, color = "#aecbff", r = 0.34, pulse = false }: { pos: Vec; color?: string; r?: number; pulse?: boolean }) {
  const m = useRef<THREE.MeshStandardMaterial>(null);
  useFrame((s) => { if (m.current && pulse) m.current.emissiveIntensity = 1.5 + Math.sin(s.clock.elapsedTime * 3) * 0.45; });
  return (
    <mesh position={pos}>
      <sphereGeometry args={[r, 22, 22]} />
      <meshStandardMaterial ref={m} color={color} emissive={color} emissiveIntensity={1.6} toneMapped={false} />
    </mesh>
  );
}
function Stem({ from, to, color = "#5f7bc0" }: { from: Vec; to: Vec; color?: string }) {
  return <Line points={[from, to]} color={color} lineWidth={1.5} transparent opacity={0.42} toneMapped={false} />;
}

// 模态编码器节点。
function Encoder({ pos, color, label, available, quality }: { pos: Vec; color: string; label: string; available: boolean; quality?: number | null }) {
  const ref = useRef<THREE.Group>(null);
  useFrame((s) => { if (ref.current) ref.current.rotation.y = Math.sin(s.clock.elapsedTime * 0.5 + pos[1]) * 0.18; });
  return (
    <group position={pos}>
      <group ref={ref}>
        <RoundedBox args={[1.05, 1, 1.05]} radius={0.12} smoothness={3}>
          <meshStandardMaterial color={available ? color : "#2a3550"} emissive={color}
            emissiveIntensity={available ? 0.7 : 0.06} metalness={0.45} roughness={0.4}
            transparent opacity={available ? 1 : 0.35} toneMapped={false} />
        </RoundedBox>
      </group>
      <Tag position={[0, 0.95, 0]} accent={available ? color : undefined}>{label}{available ? "" : "（缺失）"}</Tag>
    </group>
  );
}

// 融合主干：实体 + 线框双层、旋转。
function FusionCore({ pos }: { pos: Vec }) {
  const ref = useRef<THREE.Group>(null);
  useFrame((s, dt) => { if (ref.current) { ref.current.rotation.y += dt * 0.5; ref.current.rotation.x += dt * 0.18; } });
  return (
    <group position={pos}>
      <group ref={ref}>
        <mesh>
          <icosahedronGeometry args={[0.95, 0]} />
          <meshStandardMaterial color="#3a72e8" emissive="#3a72e8" emissiveIntensity={0.9} metalness={0.5} roughness={0.3} toneMapped={false} />
        </mesh>
        <mesh scale={1.18}>
          <icosahedronGeometry args={[0.95, 0]} />
          <meshBasicMaterial color="#9fc0ff" wireframe transparent opacity={0.4} toneMapped={false} />
        </mesh>
      </group>
      <Tag position={[0, 1.7, 0]} accent="#3a72e8">融合主干</Tag>
    </group>
  );
}

// 边界评分 B：半圆仪表（轨道 + 指针 + 数字），颜色随 B。
function BoundaryGauge({ pos, score }: { pos: Vec; score: number }) {
  const s = clamp01(score);
  const color = s > 0.66 ? "#ef5a43" : s > 0.33 ? "#f0b53e" : "#3aa66f";
  const needle = useRef<THREE.Group>(null);
  const target = Math.PI * (1 - s); // 0→右, 1→左
  useFrame((_, dt) => { if (needle.current) needle.current.rotation.z += (target - needle.current.rotation.z) * Math.min(1, dt * 4); });
  return (
    <group position={pos}>
      <mesh rotation={[0, 0, 0]}>
        <torusGeometry args={[1.1, 0.07, 12, 48, Math.PI]} />
        <meshStandardMaterial color="#2a3550" emissive="#2a3550" emissiveIntensity={0.3} toneMapped={false} />
      </mesh>
      <mesh>
        <torusGeometry args={[1.1, 0.11, 12, 48, Math.PI * s]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.2} toneMapped={false} />
      </mesh>
      <group ref={needle}>
        <mesh position={[0.5, 0, 0]}>
          <boxGeometry args={[1, 0.07, 0.07]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.4} toneMapped={false} />
        </mesh>
      </group>
      <mesh><sphereGeometry args={[0.12, 16, 16]} /><meshStandardMaterial color="#eaf0ff" emissive="#eaf0ff" emissiveIntensity={1} toneMapped={false} /></mesh>
      <Tag position={[0, 1.62, 0]} accent={color}>边界 B</Tag>
    </group>
  );
}

// 类别竞争 C：天平 —— 横梁随 top1/top2 概率差倾斜，两端球大小随概率。
function CompetitionScale({ pos, top1, top2, dist }: { pos: Vec; top1: string; top2: string; dist: Record<string, number> }) {
  const p1 = dist[top1] ?? 0, p2 = dist[top2] ?? 0;
  const beam = useRef<THREE.Group>(null);
  const tilt = Math.max(-0.5, Math.min(0.5, (p1 - p2) * 0.9));
  useFrame((_, dt) => { if (beam.current) beam.current.rotation.z += (tilt - beam.current.rotation.z) * Math.min(1, dt * 4); });
  return (
    <group position={pos}>
      <mesh position={[0, -0.7, 0]}><cylinderGeometry args={[0.08, 0.12, 1.2, 12]} /><meshStandardMaterial color="#2a3550" emissive="#2a3550" emissiveIntensity={0.3} toneMapped={false} /></mesh>
      <group ref={beam} position={[0, 0, 0]}>
        <mesh><boxGeometry args={[0.08, 0.08, 2.6]} /><meshStandardMaterial color="#8aa0c8" emissive="#8aa0c8" emissiveIntensity={0.5} toneMapped={false} /></mesh>
        <mesh position={[0, 0, -1.3]} scale={0.3 + p1}><sphereGeometry args={[0.4, 18, 18]} /><meshStandardMaterial color="#3a72e8" emissive="#3a72e8" emissiveIntensity={1} toneMapped={false} /></mesh>
        <mesh position={[0, 0, 1.3]} scale={0.3 + p2}><sphereGeometry args={[0.4, 18, 18]} /><meshStandardMaterial color="#f0b53e" emissive="#f0b53e" emissiveIntensity={1} toneMapped={false} /></mesh>
      </group>
      <Tag position={[0, 1.45, 0]} accent="#f0b53e">竞争 C</Tag>
    </group>
  );
}

// 模态可靠性 R：三根带刻度槽的发光量柱。
function ReliabilityMeters({ pos, per }: { pos: Vec; per: Record<string, number> }) {
  const mods = Object.keys(per);
  return (
    <group position={pos}>
      {mods.map((m, i) => {
        const v = clamp01(per[m] ?? 0);
        const z = (i - (mods.length - 1) / 2) * 0.85;
        const H = 2.2;
        return (
          <group key={m} position={[0, 0, z]}>
            <mesh position={[0, 0, 0]}><boxGeometry args={[0.4, H, 0.4]} /><meshStandardMaterial color="#1b2740" emissive="#1b2740" emissiveIntensity={0.25} transparent opacity={0.6} toneMapped={false} /></mesh>
            <mesh position={[0, -H / 2 + (v * H) / 2, 0]}><boxGeometry args={[0.42, Math.max(0.04, v * H), 0.42]} /><meshStandardMaterial color="#3aa66f" emissive="#3aa66f" emissiveIntensity={1} toneMapped={false} /></mesh>
          </group>
        );
      })}
      <Tag position={[0, 1.7, 0]} accent="#3aa66f">可靠性 R</Tag>
    </group>
  );
}

// 背景尘埃（景深）。
function Dust() {
  const positions = useMemo(() => {
    const a = new Float32Array(280 * 3);
    let seed = 7;
    const rnd = () => { seed = (seed * 9301 + 49297) % 233280; return seed / 233280; };
    for (let i = 0; i < 280; i++) { a[i * 3] = (rnd() - 0.5) * 46; a[i * 3 + 1] = rnd() * 18 - 4; a[i * 3 + 2] = (rnd() - 0.5) * 30; }
    return a;
  }, []);
  const ref = useRef<THREE.Points>(null);
  useFrame((s) => { if (ref.current) ref.current.rotation.y = s.clock.elapsedTime * 0.02; });
  return (
    <points ref={ref}>
      <bufferGeometry><bufferAttribute attach="attributes-position" count={280} array={positions} itemSize={3} /></bufferGeometry>
      <pointsMaterial size={0.06} color="#4a6aaa" transparent opacity={0.5} />
    </points>
  );
}

// 修正叠加层：在 p* 柱上叠加 p⁰ 参考刻度（白）+ 每类 Δ=p*−p⁰ 段（绿升/红降）。
// identity 基线时 Δ=0，白刻度正好落在柱顶；增强网络 D 训练后产生非零修正即自动显现。
function CorrectionMarks({ p0, pStar }: { p0: Record<string, number>; pStar: Record<string, number> }) {
  return (
    <group>
      {EMOTION_LABELS.map((lbl, i) => {
        const z = (i - 3) * 0.62;
        const yA = BASE_Y + (p0[lbl] ?? 0) * BAR_SCALE;
        const yB = BASE_Y + (pStar[lbl] ?? 0) * BAR_SCALE;
        const d = yB - yA;
        return (
          <group key={lbl} position={[0, 0, z]}>
            <mesh position={[0, yA, 0]}>
              <boxGeometry args={[0.66, 0.05, 0.66]} />
              <meshStandardMaterial color="#eaf0ff" emissive="#eaf0ff" emissiveIntensity={0.75} transparent opacity={0.85} toneMapped={false} />
            </mesh>
            {Math.abs(d) > 1e-4 && (
              <mesh position={[0, (yA + yB) / 2, 0]}>
                <boxGeometry args={[0.12, Math.abs(d), 0.12]} />
                <meshStandardMaterial color={d > 0 ? "#3aa66f" : "#e0533d"} emissive={d > 0 ? "#3aa66f" : "#e0533d"} emissiveIntensity={1.3} toneMapped={false} />
              </mesh>
            )}
          </group>
        );
      })}
    </group>
  );
}

function Scene({ result, focus, onFocus }: { result: AnalyzeResult; focus: string | null; onFocus: (k: string) => void }) {
  const ENC = ENC_X, FUS = FUS_X, P0 = P0_X, ST = ST_X, PS = PS_X, SY = SPINE_Y;
  const HUB_R = 0.34, FUS_R = 1.08;
  const mods = result.modalities;
  const pick = (key: string) => ({
    onClick: (e: any) => { e.stopPropagation(); onFocus(key); },
    onPointerOver: (e: any) => { e.stopPropagation(); document.body.style.cursor = "pointer"; },
    onPointerOut: () => { document.body.style.cursor = "auto"; },
  });
  return (
    <group>
      <ambientLight intensity={0.55} />
      <directionalLight position={[6, 12, 8]} intensity={0.7} />
      <pointLight position={[-10, 5, 7]} intensity={0.6} color="#3a72e8" />
      <pointLight position={[10, 5, 7]} intensity={0.5} color="#8a6fd1" />
      <Dust />
      <Grid position={[0, -4, 0]} args={[70, 44]} cellSize={1} cellThickness={0.5} sectionSize={5} sectionThickness={1}
        cellColor="#1b2740" sectionColor="#2b3f66" fadeDistance={48} fadeStrength={1.4} infiniteGrid />

      {/* 编码器 → 融合（可点击聚焦） */}
      {mods.map((m, i) => {
        const y = SY + ([1.8, 0, -1.8][i] ?? 0);
        const col = MOD_COLOR[m.modality] ?? "#aab2c4";
        return (
          <group key={m.modality}>
            <group {...pick("enc" + i)}>
              <Appear position={[ENC, y, 0]} delay={i * 0.08}>
                <Encoder pos={[0, 0, 0]} color={col} label={MOD_ZH[m.modality]} available={m.available} />
              </Appear>
            </group>
            <Conduit from={[ENC + 0.52, y, 0]} to={[FUS, SY, 0]} color={col} r1={FUS_R} />
          </group>
        );
      })}

      <group {...pick("fusion")}>
        <Appear position={[FUS, SY, 0]} delay={0.28}>
          <FusionCore pos={[0, 0, 0]} />
        </Appear>
      </group>
      <Conduit from={[FUS, SY, 0]} to={[P0, SY, 0]} r0={FUS_R} r1={HUB_R} />

      {/* 初始预测 p⁰ */}
      <group {...pick("p0")}>
        <Hub pos={[P0, SY, 0]} pulse />
        <Stem from={[P0, SY - HUB_R, 0]} to={[P0, BASE_Y + 1.4, 0]} />
        <Appear position={[P0, 0, 0]} delay={0.42}>
          <Histogram dist={result.initial.distribution} x={0} title="初始预测 p⁰" />
        </Appear>
      </group>
      <Conduit from={[P0, SY, 0]} to={[ST, SY, 0]} r0={HUB_R} r1={HUB_R} />

      {/* 三状态 */}
      <group {...pick("states")}>
        <Hub pos={[ST, SY, 0]} pulse />
        <Stem from={[ST, SY + 0.2, 0]} to={[ST, 1.9, 0]} />
        <Stem from={[ST, SY - 0.2, 0]} to={[ST, -0.7, 0]} />
        <Stem from={[ST, -1.1, 0]} to={[ST, -2.2, 0]} />
        <Appear position={[ST, 2.9, 0]} delay={0.6}>
          <BoundaryGauge pos={[0, 0, 0]} score={result.boundary.score} />
        </Appear>
        <Appear position={[ST, -0.4, 0]} delay={0.68}>
          <CompetitionScale pos={[0, 0, 0]} top1={result.competition.top1} top2={result.competition.top2} dist={result.initial.distribution} />
        </Appear>
        <Appear position={[ST, -2.7, 0]} delay={0.76}>
          <ReliabilityMeters pos={[0, 0, 0]} per={result.reliability.per_modality} />
        </Appear>
      </group>
      <Conduit from={[ST, SY, 0]} to={[PS, SY, 0]} color="#8a6fd1" r0={HUB_R} r1={HUB_R} />

      {/* 最终预测 p* + 修正叠加（白刻度=p⁰参考，绿/红=Δ；详见点击面板） */}
      <group {...pick("pstar")}>
        <Hub pos={[PS, SY, 0]} color="#c9b6ff" pulse />
        <Stem from={[PS, SY - HUB_R, 0]} to={[PS, BASE_Y + 1.4, 0]} color="#8a6fd1" />
        <Appear position={[PS, 0, 0]} delay={0.9}>
          <Histogram dist={result.enhancement.final_distribution} x={0} highlight={result.final_emotion} title="最终预测 p*" />
          <CorrectionMarks p0={result.initial.distribution} pStar={result.enhancement.final_distribution} />
        </Appear>
      </group>

      {focus && FOCUS_AT[focus] && <FocusRing pos={FOCUS_AT[focus].t} />}
    </group>
  );
}

class ErrBoundary extends Component<{ children: ReactNode; onError: () => void }, { dead: boolean }> {
  state = { dead: false };
  static getDerivedStateFromError() { return { dead: true }; }
  componentDidCatch() { this.props.onError(); }
  render() { return this.state.dead ? null : this.props.children; }
}

// 小型 2D 分布条（详情面板内）。
function MiniDist({ dist, highlight }: { dist: Record<string, number>; highlight?: string }) {
  const top = highlight ?? topOf(dist);
  return (
    <div className="p3d-mini">
      {EMOTION_LABELS.map((l) => {
        const v = dist[l] ?? 0;
        return (
          <div className="p3d-mini-row" key={l}>
            <span>{ZH[l]}</span>
            <div className="p3d-mini-track"><div className="p3d-mini-fill" style={{ width: `${Math.max(2, v * 100)}%`, background: l === top ? EMO[l] : "#46506e" }} /></div>
            <span className="mono">{(v * 100).toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

// 各阶段的完整数据（取自 AnalyzeResult）——「所有流程」按需展开。
function detailRows(focus: string, r: AnalyzeResult): { title: string; secs: any[] } {
  if (focus.startsWith("enc")) {
    const i = +focus.slice(3); const m = r.modalities[i];
    if (!m) return { title: "模态", secs: [] };
    const qf = QUALITY_FIELDS[m.modality] ?? [];
    const qrows = qf.map(([k, lab]) => [lab, (m.quality as any)[k] == null ? "—" : String((m.quality as any)[k]), true]);
    const trained = String(m.backend).startsWith("trained");
    const secs: any[] = [
      { rows: [["模态", MOD_ZH[m.modality]], ["状态", m.available ? "可用" : "缺失 / 降级"], ["编码器", `${trained ? "真模型" : "桩 / 基线"}（${m.backend}）`]] },
      { h: "信号质量（预处理 · 确定性）", rows: qrows.length ? qrows : [["—", "无"]] },
      m.distribution ? { h: "单模态分布", bars: m.distribution } : { h: "单模态分布", rows: [["—", "桩：无单模态分布"]] },
    ];
    return { title: `① 编码 · ${MOD_ZH[m.modality]}`, secs };
  }
  if (focus === "fusion") {
    return { title: "② 融合主干", secs: [
      { rows: [["主干", r.initial.backbone], ["机制", "跨模态注意力 + 低秩张量融合（当前晚融合基线）"]] },
      { h: "初始预测 p⁰", bars: r.initial.distribution },
    ] };
  }
  if (focus === "p0") return { title: "③ 初始预测 p⁰", secs: [{ h: "7 类分布", bars: r.initial.distribution }] };
  if (focus === "states") {
    const b = r.boundary, c = r.competition, rel = r.reliability;
    const dirRows = Object.entries(c.per_modality_direction).map(([k, v]) => [MOD_ZH[k] ?? k, `${v >= 0 ? "▲偏" + ZH[c.top1] : "▼偏" + ZH[c.top2]} ${v.toFixed(2)}`]);
    const relRows = Object.entries(rel.per_modality).map(([k, v]) => [MOD_ZH[k] ?? k, v.toFixed(3), true]);
    return { title: "④ 三状态诊断", secs: [
      { h: "边界评分 B", rows: [["分值 B", b.score.toFixed(3), true], ["最大置信", b.max_confidence.toFixed(3), true], ["Top-2 间隔", b.top2_margin.toFixed(3), true], ["归一化熵", b.norm_entropy.toFixed(3), true]] },
      { h: "类别竞争 C", rows: [["Top-1 / Top-2", `${ZH[c.top1]} / ${ZH[c.top2]}`], ["margin", c.margin.toFixed(3), true], ["熵", c.entropy.toFixed(3), true], ...dirRows] },
      { h: "模态可靠性 R", rows: relRows.length ? relRows : [["—", "无可用模态"]] },
    ] };
  }
  if (focus === "pstar") {
    const e = r.enhancement;
    return { title: "⑤ 有界增强 → p*", secs: [
      { rows: [["增强", e.enhancement], ["有界", e.bounded ? "是" : "否"], ["‖p*−p⁰‖", e.correction_magnitude.toFixed(4) + (e.correction_magnitude < 1e-4 ? "（identity 基线）" : ""), true], ["最终情感", ZH[r.final_emotion]]] },
      { h: "最终分布 p*", bars: e.final_distribution, hl: r.final_emotion },
    ] };
  }
  return { title: "", secs: [] };
}

function DetailPanel({ focus, result, onClose }: { focus: string; result: AnalyzeResult; onClose: () => void }) {
  const { title, secs } = detailRows(focus, result);
  return (
    <div className="p3d-panel">
      <div className="p3d-panel-head">
        <b>{title}</b>
        <button className="p3d-panel-x" onClick={onClose}>返回全览 ✕</button>
      </div>
      <div className="p3d-panel-body">
        {secs.map((sec: any, i: number) => (
          <div className="p3d-panel-sec" key={i}>
            {sec.h && <div className="p3d-panel-h">{sec.h}</div>}
            {sec.rows && sec.rows.map((row: any, j: number) => (
              <div className="p3d-panel-row" key={j}><span>{row[0]}</span><b className={row[2] ? "mono" : ""}>{row[1]}</b></div>
            ))}
            {sec.bars && <MiniDist dist={sec.bars} highlight={sec.hl} />}
          </div>
        ))}
      </div>
    </div>
  );
}

export function Pipeline3D({ result, seq }: { result: AnalyzeResult; seq: number }) {
  const [ok, setOk] = useState<boolean | null>(null);
  const [coarse, setCoarse] = useState(false);
  const [focus, setFocus] = useState<string | null>(null);
  const controls = useRef<any>(null);
  useEffect(() => {
    try { const c = document.createElement("canvas"); setOk(!!(c.getContext("webgl2") || c.getContext("webgl"))); }
    catch { setOk(false); }
    try { setCoarse(window.matchMedia?.("(pointer: coarse)").matches ?? false); } catch { /* noop */ }
  }, []);
  useEffect(() => { setFocus(null); }, [seq]);

  if (ok === false) return <div className="card muted small">当前环境不支持 WebGL，已回退到下方 2D 详情视图。</div>;

  const rel = Object.values(result.reliability.per_modality);
  const relAvg = rel.length ? rel.reduce((a, b) => a + b, 0) / rel.length : 0;
  const corr = result.enhancement.correction_magnitude ?? 0;

  return (
    <div className="card p3d-card">
      <div className="p3d-head">
        <b>运行可视化</b> · 你的样本流经的内部计算结构
        <span className="muted small">
          {coarse ? "（点击阶段看详情 · 单指翻页 · 双指旋转/缩放）" : "（点击阶段看详情 · 拖拽旋转 · 滚轮缩放）"}
        </span>
      </div>
      <div className="p3d-stage">
        {/* HUD 读数面板 */}
        <div className="p3d-hud">
          <div className="p3d-hud-row"><span>最终</span><b style={{ color: EMO[result.final_emotion] }}>{ZH[result.final_emotion]}</b></div>
          <div className="p3d-hud-row"><span>边界 B</span><b className="mono">{result.boundary.score.toFixed(3)}</b></div>
          <div className="p3d-hud-row"><span>竞争 C</span><b>{ZH[result.competition.top1]}↔{ZH[result.competition.top2]}</b></div>
          <div className="p3d-hud-row"><span>可靠性 R̄</span><b className="mono">{relAvg.toFixed(2)}</b></div>
          <div className="p3d-hud-row"><span>‖p*−p⁰‖</span><b className="mono">{corr.toFixed(4)}</b></div>
        </div>
        {/* 情感图例 */}
        <div className="p3d-legend">
          {EMOTION_LABELS.map((l) => (
            <span key={l} className="p3d-leg"><i style={{ background: EMO[l] }} />{ZH[l]}</span>
          ))}
        </div>
        <div className="p3d-canvas">
          <ErrBoundary onError={() => setOk(false)}>
            <Canvas camera={{ position: [0, 5, 21], fov: 42 }} dpr={[1, coarse ? 1.5 : 2]} gl={{ antialias: !coarse }} onPointerMissed={() => setFocus(null)}>
              <color attach="background" args={["#0a0e18"]} />
              <fog attach="fog" args={["#0a0e18", 26, 60]} />
              <Float speed={1} rotationIntensity={0} floatIntensity={0.3}>
                <Scene key={seq} result={result} focus={focus} onFocus={setFocus} />
              </Float>
              <Rig controls={controls} />
              <FocusController focus={focus} controls={controls} />
              <OrbitControls
                ref={controls}
                enablePan={false}
                minDistance={10}
                maxDistance={62}
                target={[0, 0.4, 0]}
                autoRotate={focus === null}
                autoRotateSpeed={0.45}
                touches={{ ONE: coarse ? undefined : THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_ROTATE }}
              />
              <EffectComposer>
                <Bloom intensity={0.85} luminanceThreshold={0.22} luminanceSmoothing={0.3} mipmapBlur radius={0.72} />
                <Vignette offset={0.24} darkness={0.72} eskil={false} />
              </EffectComposer>
            </Canvas>
          </ErrBoundary>
        </div>
        {focus && <DetailPanel focus={focus} result={result} onClose={() => setFocus(null)} />}
      </div>
    </div>
  );
}
