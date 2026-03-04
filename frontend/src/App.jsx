import { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Cpu,
  Layers,
  Settings,
  Database,
  CheckCircle2,
  AlertTriangle,
  Play,
  BarChart3,
  Terminal,
  Save,
  History,
  GitBranch,
  RefreshCw,
  Search,
  ExternalLink,
  Maximize2,
  Crosshair,
  TrendingUp,
  FileCode,
  Eye,
} from 'lucide-react';

// --- Configuration & Mock Data ---
const SPECS = [
  { id: 'vref', label: 'Vref (mV)', target: 1200, unit: 'mV', tol: 10 },
  { id: 'tc', label: 'TempCo (ppm/C)', target: 20, unit: 'ppm', tol: 5 },
  { id: 'psrr', label: 'PSRR (dB)', target: -60, unit: 'dB', tol: 5 },
  { id: 'iq', label: 'I-Quiescent (µA)', target: 10, unit: 'µA', tol: 2 },
];

const DESIGN_VARS = [
  { id: 'w_m1', label: 'W (M1/M2)', min: 1, max: 20, default: 5, unit: 'µm' },
  { id: 'r_ratio', label: 'R-Ratio (N)', min: 4, max: 12, default: 8, unit: '' },
  { id: 'i_bias', label: 'I-Bias', min: 1, max: 50, default: 10, unit: 'µA' },
];

const CANDIDATES = [
  { id: 'BG-001', variables: 'N=8.2, W=5.4', surrogate: '12.1', spice: '12.4', power: '8.2µW', status: 'pass' },
  { id: 'BG-002', variables: 'N=7.9, W=4.2', surrogate: '18.4', spice: '19.1', power: '6.4µW', status: 'pass' },
  { id: 'BG-003', variables: 'N=9.1, W=6.8', surrogate: '22.1', spice: '24.5', power: '10.5µW', status: 'fail' },
];

const CORNER_DATA = [
  { name: 'TT_25C_1.8V', status: 'pass', error: '0.1%' },
  { name: 'SS_-40C_1.6V', status: 'pass', error: '0.4%' },
  { name: 'FF_125C_2.0V', status: 'warn', error: '1.2%' },
  { name: 'FS_25C_1.8V', status: 'pass', error: '0.2%' },
];

const LOG_LINES = [
  '[00:00.001] INFO  Optimizer initialized (Bayesian/GP, 50 max iterations)',
  '[00:00.123] INFO  Loading surrogate model from ml/checkpoints/surrogate_v3.pkl',
  '[00:00.245] INFO  Spec targets loaded from bandgap/specs.yaml',
  '[00:01.312] INFO  Iteration  1/50 — loss=0.842  best_tc=22.1ppm',
  '[00:02.198] INFO  Iteration  5/50 — loss=0.531  best_tc=18.7ppm',
  '[00:03.445] INFO  Iteration 12/50 — loss=0.294  best_tc=14.2ppm',
  '[00:04.887] INFO  Iteration 20/50 — loss=0.178  best_tc=12.4ppm',
  '[00:05.021] WARN  FF_125C_2.0V corner error = 1.2% (threshold 1.0%)',
  '[00:05.512] INFO  Iteration 28/50 — loss=0.091  best_tc=12.1ppm',
  '[00:06.002] INFO  Convergence criterion met at iteration 28',
  '[00:06.003] INFO  Top-3 candidates written to results/candidates.json',
  '[00:06.010] INFO  Done. Elapsed: 6.01s',
];

// Fake convergence data (iteration vs best loss)
const CONVERGENCE = Array.from({ length: 28 }, (_, i) => ({
  iter: i + 1,
  loss: Math.max(0.07, 0.85 * Math.exp(-0.18 * i)),
}));

// --- Styles ---
const S = {
  appWrapper: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    backgroundColor: '#0f172a',
    color: '#f8fafc',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    overflow: 'hidden',
  },
  topBar: {
    height: '56px',
    backgroundColor: '#1e293b',
    borderBottom: '1px solid #334155',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 1.5rem',
    flexShrink: 0,
  },
  topBarLeft: { display: 'flex', alignItems: 'center', gap: '0.75rem' },
  topBarRight: { display: 'flex', alignItems: 'center', gap: '0.5rem' },
  logoText: { fontWeight: '700', fontSize: '1rem', color: '#38bdf8', letterSpacing: '0.02em' },
  subText: { fontSize: '0.75rem', color: '#64748b' },
  contentArea: { display: 'flex', flex: 1, overflow: 'hidden' },
  sidebar: {
    width: '210px',
    backgroundColor: '#0f172a',
    borderRight: '1px solid #1e293b',
    padding: '1rem 0.5rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.25rem',
    flexShrink: 0,
  },
  sideSection: {
    fontSize: '0.65rem',
    color: '#475569',
    fontWeight: '600',
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    padding: '0.5rem 1rem 0.25rem',
  },
  navItem: (active) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '0.625rem',
    padding: '0.625rem 0.875rem',
    borderRadius: '0.5rem',
    cursor: 'pointer',
    backgroundColor: active ? '#1e293b' : 'transparent',
    color: active ? '#38bdf8' : '#94a3b8',
    fontSize: '0.8125rem',
    fontWeight: active ? '600' : '400',
    border: 'none',
    width: '100%',
    textAlign: 'left',
  }),
  main: {
    flex: 1,
    overflowY: 'auto',
    padding: '1.25rem',
    backgroundColor: '#020617',
  },
  bottomBar: {
    height: '32px',
    backgroundColor: '#1e293b',
    borderTop: '1px solid #334155',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 1rem',
    fontSize: '0.7rem',
    color: '#64748b',
    flexShrink: 0,
  },
  card: (extra = {}) => ({
    backgroundColor: '#1e293b',
    borderRadius: '0.625rem',
    padding: '1rem',
    border: '1px solid #334155',
    display: 'flex',
    flexDirection: 'column',
    ...extra,
  }),
  cardTitle: {
    fontSize: '0.8125rem',
    fontWeight: '600',
    color: '#cbd5e1',
    marginBottom: '0.75rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  btn: (variant = 'secondary', extra = {}) => ({
    padding: '0.4rem 0.875rem',
    borderRadius: '0.375rem',
    border: 'none',
    cursor: 'pointer',
    fontSize: '0.8125rem',
    fontWeight: '600',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.375rem',
    backgroundColor: variant === 'primary' ? '#38bdf8' : '#334155',
    color: variant === 'primary' ? '#0f172a' : '#f8fafc',
    ...extra,
  }),
  badge: (status) => ({
    padding: '0.1rem 0.45rem',
    borderRadius: '0.25rem',
    fontSize: '0.625rem',
    fontWeight: '700',
    textTransform: 'uppercase',
    backgroundColor: status === 'pass' ? '#064e3b' : status === 'warn' ? '#78350f' : '#7f1d1d',
    color: status === 'pass' ? '#6ee7b7' : status === 'warn' ? '#fbbf24' : '#fca5a5',
  }),
  label: { fontSize: '0.75rem', color: '#94a3b8' },
};

// --- Sub-components ---

function SliderRow({ varDef, value, onChange }) {
  return (
    <div style={{ marginBottom: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
        <span style={S.label}>{varDef.label}</span>
        <span style={{ fontSize: '0.75rem', color: '#38bdf8', fontWeight: '600' }}>
          {value.toFixed(1)}{varDef.unit}
        </span>
      </div>
      <input
        type="range"
        min={varDef.min}
        max={varDef.max}
        step={(varDef.max - varDef.min) / 100}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: '100%', accentColor: '#38bdf8', cursor: 'pointer' }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.125rem' }}>
        <span style={{ fontSize: '0.65rem', color: '#475569' }}>{varDef.min}</span>
        <span style={{ fontSize: '0.65rem', color: '#475569' }}>{varDef.max}</span>
      </div>
    </div>
  );
}

function SpecRow({ spec }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0.375rem 0',
      borderBottom: '1px solid #0f172a',
    }}>
      <span style={S.label}>{spec.label}</span>
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <span style={{ fontSize: '0.75rem', color: '#f8fafc', fontWeight: '500' }}>{spec.target}</span>
        <span style={{ fontSize: '0.65rem', color: '#475569' }}>±{spec.tol} {spec.unit}</span>
      </div>
    </div>
  );
}

function CandidateRow({ c, selected, onClick, bestEffort }) {
  return (
    <div
      onClick={onClick}
      style={{
        display: 'grid',
        gridTemplateColumns: '70px 1fr 55px 55px 60px 50px',
        gap: '0.5rem',
        alignItems: 'center',
        padding: '0.5rem 0.625rem',
        borderRadius: '0.375rem',
        cursor: 'pointer',
        backgroundColor: selected ? '#0f172a' : 'transparent',
        borderLeft: selected ? '2px solid #38bdf8' : bestEffort ? '2px solid #fbbf24' : '2px solid transparent',
        marginBottom: '0.25rem',
      }}
    >
      <span style={{ fontSize: '0.75rem', color: '#38bdf8', fontWeight: '600' }}>{c.id}</span>
      <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{c.variables}</span>
      <span style={{ fontSize: '0.7rem', color: '#f8fafc', textAlign: 'right' }}>{c.vref_mV ?? c.surrogate}</span>
      <span style={{ fontSize: '0.7rem', color: '#94a3b8', textAlign: 'right' }}>{c.err_mV != null ? `±${c.err_mV}` : c.spice}</span>
      <span style={{ fontSize: '0.7rem', color: '#94a3b8', textAlign: 'right' }}>{c.power}</span>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <span style={S.badge(bestEffort ? 'warn' : c.status)}>{bestEffort ? 'best' : c.status}</span>
      </div>
    </div>
  );
}

// Mini SVG convergence chart
function ConvergenceChart({ data }) {
  const W = 280;
  const H = 90;
  const pad = { t: 8, r: 8, b: 20, l: 32 };
  const iW = W - pad.l - pad.r;
  const iH = H - pad.t - pad.b;
  const maxIter = data[data.length - 1].iter;
  const maxLoss = 0.9;
  const minLoss = 0;
  const xScale = (iter) => pad.l + (iter / maxIter) * iW;
  const yScale = (loss) => pad.t + (1 - (loss - minLoss) / (maxLoss - minLoss)) * iH;
  const pathD = data
    .map((d, i) => `${i === 0 ? 'M' : 'L'}${xScale(d.iter).toFixed(1)},${yScale(d.loss).toFixed(1)}`)
    .join(' ');

  return (
    <svg width={W} height={H} style={{ display: 'block' }}>
      {[0.2, 0.4, 0.6, 0.8].map((v) => (
        <line key={v} x1={pad.l} x2={W - pad.r} y1={yScale(v)} y2={yScale(v)}
          stroke="#1e293b" strokeWidth="1" />
      ))}
      <path d={`${pathD} L${xScale(maxIter)},${pad.t + iH} L${pad.l},${pad.t + iH} Z`}
        fill="rgba(56,189,248,0.08)" />
      <path d={pathD} fill="none" stroke="#38bdf8" strokeWidth="1.5" strokeLinejoin="round" />
      <text x={pad.l} y={H - 4} fill="#475569" fontSize="9" textAnchor="middle">0</text>
      <text x={W - pad.r} y={H - 4} fill="#475569" fontSize="9" textAnchor="middle">{maxIter}</text>
      <text x={pad.l - 4} y={yScale(maxLoss) + 3} fill="#475569" fontSize="9" textAnchor="end">0.9</text>
      <text x={pad.l - 4} y={yScale(0.1) + 3} fill="#475569" fontSize="9" textAnchor="end">0.1</text>
      <text x={W / 2} y={H - 2} fill="#475569" fontSize="9" textAnchor="middle">Iteration</text>
      <text
        x={10}
        y={pad.t + iH / 2}
        fill="#475569"
        fontSize="9"
        textAnchor="middle"
        transform={`rotate(-90 10 ${pad.t + iH / 2})`}
      >
        Best |Vref-target| (V)
      </text>
    </svg>
  );
}

// Simple layout mock (SVG floorplan)
function LayoutViewer({ layer, layoutData }) {
  const layerColors = {
    diff: '#15803d',
    poly: '#b45309',
    contact: '#7c3aed',
    metal1: '#1d4ed8',
    via1: '#a855f7',
    metal2: '#0ea5e9',
    nwell: '#f59e0b',
    pwell: '#f97316',
  };

  const patch = layoutData?.patch;
  const layerMap = layoutData?.layer_map || {};
  if (!patch || !patch.length) {
    return (
      <div style={{
        backgroundColor: '#020617', borderRadius: '0.375rem', border: '1px solid #334155',
        minHeight: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b',
      }}>
        Layout preview unavailable
      </div>
    );
  }

  const patchSize = patch[0].length;
  const cellSize = 7;
  const layerEntries = Object.entries(layerMap).map(([idx, name]) => ({ idx: Number(idx), name }));
  const selectedLayer = layerEntries.find((entry) => entry.name === layer);

  const cells = [];
  for (let y = 0; y < patchSize; y += 1) {
    for (let x = 0; x < patchSize; x += 1) {
      let color = null;
      if (layer === 'all') {
        const hit = layerEntries.find((entry) => patch[entry.idx]?.[y]?.[x] === 1);
        if (hit) color = layerColors[hit.name] || '#38bdf8';
      } else if (selectedLayer && patch[selectedLayer.idx]?.[y]?.[x] === 1) {
        color = layerColors[selectedLayer.name] || '#38bdf8';
      }

      if (color) {
        cells.push(
          <rect
            key={`${x}-${y}`}
            x={x * cellSize}
            y={y * cellSize}
            width={cellSize - 0.5}
            height={cellSize - 0.5}
            fill={color}
            opacity="0.85"
          />
        );
      }
    }
  }

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${patchSize * cellSize} ${patchSize * cellSize}`}
      style={{ backgroundColor: '#020617', borderRadius: '0.375rem', border: '1px solid #334155' }}
    >
      {cells}
    </svg>
  );
}

// --- Tab views ---

function OptimizationTab({ isSimulating, onRun, selectedCandidate, setSelectedCandidate, designValues, setDesignValues, candidates, convergenceData, optimResult, liveEstimate, isEstimating, estimateError }) {
  const displayCandidates = candidates || CANDIDATES;
  const displayConvergence = convergenceData?.length ? convergenceData : CONVERGENCE;
  const anyPass = displayCandidates.some((c) => c.status === 'pass');
  const bestEffortId = anyPass ? null : displayCandidates[0]?.id;
  const lastEntry = displayConvergence[displayConvergence.length - 1];
  const bestLoss = lastEntry?.loss ?? lastEntry?.best_error_V;
  const convergedAt = displayConvergence.length;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr 300px', gap: '1rem', minHeight: '0' }}>
      {/* Left: Controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={S.card()}>
          <div style={S.cardTitle}><Crosshair size={13} />Spec Targets</div>
          {SPECS.map((s) => <SpecRow key={s.id} spec={s} />)}
        </div>

        <div style={S.card({ flex: 1 })}>
          <div style={S.cardTitle}><Settings size={13} />Design Variables</div>
          {DESIGN_VARS.map((v) => (
            <SliderRow
              key={v.id}
              varDef={v}
              value={designValues[v.id]}
              onChange={(val) => setDesignValues((prev) => ({ ...prev, [v.id]: val }))}
            />
          ))}
          <div style={{ backgroundColor: '#0f172a', borderRadius: '0.375rem', padding: '0.5rem 0.625rem', marginTop: '0.25rem' }}>
            <div style={{ ...S.label, marginBottom: '0.25rem' }}>
              Live estimate {isEstimating ? '(updating...)' : ''}
            </div>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
              <span style={S.label}>Vref: <strong style={{ color: '#38bdf8' }}>{liveEstimate?.vref_mV != null ? `${liveEstimate.vref_mV.toFixed(1)} mV` : '—'}</strong></span>
              <span style={S.label}>Iq: <strong style={{ color: '#38bdf8' }}>{liveEstimate?.iq_uA != null ? `${liveEstimate.iq_uA.toFixed(2)} µA` : '—'}</strong></span>
              <span style={S.label}>Vref Spec: <strong style={{ color: liveEstimate?.spec_vref ? '#6ee7b7' : '#fca5a5' }}>{liveEstimate?.spec_vref == null ? '—' : liveEstimate.spec_vref ? 'PASS' : 'FAIL'}</strong></span>
            </div>
            {estimateError && <div style={{ ...S.label, color: '#fbbf24', marginTop: '0.25rem' }}>{estimateError}</div>}
          </div>
          <div style={{ marginTop: 'auto', paddingTop: '0.75rem', display: 'flex', gap: '0.5rem' }}>
            <button style={S.btn('primary', { flex: 1 })} onClick={onRun} disabled={isSimulating}>
              {isSimulating ? <RefreshCw size={13} /> : <Play size={13} />}
              {isSimulating ? 'Running…' : 'Run Optimizer'}
            </button>
          </div>
        </div>
      </div>

      {/* Center: Results */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={S.card({ flex: 1 })}>
          <div style={{ ...S.cardTitle, justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <TrendingUp size={13} />Top Candidates
            </span>
            <button style={S.btn('secondary', { fontSize: '0.7rem', padding: '0.2rem 0.5rem' })}>
              <Save size={11} />Export
            </button>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '70px 1fr 55px 55px 60px 50px',
            gap: '0.5rem',
            padding: '0 0.625rem 0.375rem',
            borderBottom: '1px solid #334155',
          }}>
            {['ID', 'Variables', 'Vref (mV)', 'Err (mV)', 'Power', 'Status'].map((h) => (
              <span key={h} style={{ fontSize: '0.65rem', color: '#475569', fontWeight: '600', textTransform: 'uppercase' }}>{h}</span>
            ))}
          </div>
          <div style={{ marginTop: '0.375rem' }}>
            {displayCandidates.map((c) => (
              <CandidateRow
                key={c.id}
                c={c}
                selected={selectedCandidate?.id === c.id}
                onClick={() => setSelectedCandidate(c)}
                bestEffort={bestEffortId === c.id}
              />
            ))}
          </div>
        </div>

        <div style={S.card()}>
          <div style={S.cardTitle}><BarChart3 size={13} />Optimizer Convergence</div>
          <ConvergenceChart data={displayConvergence} />
          <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
            <span style={S.label}>Best |Vref-target|: <strong style={{ color: '#38bdf8' }}>{bestLoss != null ? `${(bestLoss * 1000).toFixed(1)} mV` : '—'}</strong></span>
            <span style={S.label}>Iterations: <strong style={{ color: '#38bdf8' }}>{convergedAt}</strong></span>
          </div>
        </div>
      </div>

      {/* Right: Selected candidate + corner analysis */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={S.card()}>
          <div style={S.cardTitle}><Eye size={13} />Selected: {selectedCandidate?.id}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
            {[
              { label: 'Vref', value: selectedCandidate ? `${parseFloat(selectedCandidate.vref_mV ?? selectedCandidate.surrogate ?? 0).toFixed(1)} mV` : '—' },
              { label: '|Err| from 1200mV', value: selectedCandidate?.err_mV != null ? `${selectedCandidate.err_mV} mV` : '—' },
              { label: 'PSRR', value: selectedCandidate?.psrr_dB != null ? `${selectedCandidate.psrr_dB} dB` : optimResult ? '—' : '-61.2 dB' },
              { label: 'Iq', value: selectedCandidate?.iq_uA != null ? `${selectedCandidate.iq_uA} µA` : optimResult ? '—' : '9.8 µA' },
              { label: 'Power', value: selectedCandidate?.power ?? '—' },
              {
                label: 'Status',
                value: selectedCandidate ? (selectedCandidate.status === 'pass' ? '✓ Pass' : '✗ Fail') : '—',
              },
            ].map(({ label, value }) => (
              <div key={label} style={{ backgroundColor: '#0f172a', borderRadius: '0.375rem', padding: '0.5rem 0.625rem' }}>
                <div style={{ fontSize: '0.65rem', color: '#475569', marginBottom: '0.2rem' }}>{label}</div>
                <div style={{ fontSize: '0.8125rem', color: '#f8fafc', fontWeight: '600' }}>{value}</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: '0.75rem', display: 'flex', gap: '0.5rem' }}>
            <button style={S.btn('primary', { flex: 1, fontSize: '0.75rem' })}>
              <FileCode size={12} />Export Netlist
            </button>
          </div>
        </div>

        <div style={S.card({ flex: 1 })}>
          <div style={S.cardTitle}><Activity size={13} />Corner Analysis</div>
          {CORNER_DATA.map((c) => (
            <div key={c.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #0f172a' }}>
              <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{c.name}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{c.error}</span>
                <span style={S.badge(c.status)}>{c.status}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function LayoutTab({ layer, setLayer, layoutData, layoutError }) {
  const layerNames = layoutData?.layer_map ? Object.values(layoutData.layer_map) : ['diff', 'poly', 'contact', 'metal1'];
  const layerOptions = ['all', ...layerNames];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 240px', gap: '1rem' }}>
      <div style={S.card({ gap: '0.75rem' })}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={S.cardTitle}><Layers size={13} />Layout Viewer — Bandgap Core</div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button style={S.btn()}><Search size={12} /></button>
            <button style={S.btn()}><Maximize2 size={12} /></button>
            <button style={S.btn()}><ExternalLink size={12} /></button>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '0.375rem', flexWrap: 'wrap' }}>
          {layerOptions.map((l) => (
            <button key={l} onClick={() => setLayer(l)}
              style={S.btn(l === layer ? 'primary' : 'secondary', { fontSize: '0.7rem', padding: '0.2rem 0.625rem' })}>
              {l}
            </button>
          ))}
        </div>
        {layoutError && <span style={{ ...S.label, color: '#fbbf24' }}>Layout API warning: {layoutError}</span>}
        <LayoutViewer layer={layer} layoutData={layoutData} />
        <div style={{ display: 'flex', gap: '1rem' }}>
          <span style={S.label}>DRC: <strong style={{ color: '#f8fafc' }}>{layoutData?.drc?.n_violations ?? '—'} violations</strong></span>
          <span style={S.label}>Pass Rate: <strong style={{ color: '#6ee7b7' }}>{layoutData?.drc?.pass_rate != null ? `${(layoutData.drc.pass_rate * 100).toFixed(0)}%` : '—'}</strong></span>
          <span style={S.label}>Patch: <strong style={{ color: '#f8fafc' }}>{layoutData?.patch_size ?? '—'}×{layoutData?.patch_size ?? '—'}</strong></span>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={S.card()}>
          <div style={S.cardTitle}><Database size={13} />Layer Legend</div>
          {[
            { color: '#15803d', label: 'diff' },
            { color: '#b45309', label: 'poly' },
            { color: '#7c3aed', label: 'contact' },
            { color: '#1d4ed8', label: 'metal1' },
            { color: '#a855f7', label: 'via1' },
            { color: '#0ea5e9', label: 'metal2' },
            { color: '#f59e0b', label: 'nwell' },
            { color: '#f97316', label: 'pwell' },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.625rem', marginBottom: '0.375rem' }}>
              <div style={{ width: '14px', height: '14px', backgroundColor: color + '60', border: `1px solid ${color}`, borderRadius: '2px' }} />
              <span style={S.label}>{label}</span>
            </div>
          ))}
        </div>

        <div style={S.card({ flex: 1 })}>
          <div style={S.cardTitle}><Settings size={13} />Design Rules</div>
          {[
            { rule: 'Min Width (M1)', value: '0.12 µm', ok: true },
            { rule: 'Min Space (M1)', value: '0.12 µm', ok: true },
            { rule: 'Poly-Diff overlap', value: '0.06 µm', ok: true },
            { rule: 'Via enclosure', value: '0.04 µm', ok: true },
          ].map(({ rule, value, ok }) => (
            <div key={rule} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.3rem 0', borderBottom: '1px solid #0f172a' }}>
              <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{rule}</span>
              <div style={{ display: 'flex', gap: '0.375rem', alignItems: 'center' }}>
                <span style={{ fontSize: '0.7rem', color: '#f8fafc' }}>{value}</span>
                {ok ? <CheckCircle2 size={11} color="#6ee7b7" /> : <AlertTriangle size={11} color="#fbbf24" />}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function VerificationTab() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
      <div style={S.card()}>
        <div style={S.cardTitle}><CheckCircle2 size={13} />Spec Verification Summary</div>
        {SPECS.map((s) => {
          const simResults = { vref: 1199.4, tc: 12.4, psrr: -61.2, iq: 9.8 };
          const value = simResults[s.id];
          const ok = Math.abs(value - s.target) <= s.tol;
          return (
            <div key={s.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #0f172a' }}>
              <span style={S.label}>{s.label}</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.625rem' }}>
                <span style={{ fontSize: '0.75rem', color: '#f8fafc' }}>{value} {s.unit}</span>
                <span style={{ fontSize: '0.7rem', color: '#64748b' }}>target {s.target}±{s.tol}</span>
                {ok
                  ? <CheckCircle2 size={13} color="#6ee7b7" />
                  : <AlertTriangle size={13} color="#fbbf24" />}
              </div>
            </div>
          );
        })}
      </div>

      <div style={S.card()}>
        <div style={S.cardTitle}><Activity size={13} />Corner Sweep Results</div>
        {CORNER_DATA.map((c) => (
          <div key={c.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid #0f172a' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{c.name}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Err: {c.error}</span>
              <span style={S.badge(c.status)}>{c.status}</span>
            </div>
          </div>
        ))}
        <div style={{ marginTop: '0.75rem', padding: '0.5rem 0.75rem', backgroundColor: '#0f172a', borderRadius: '0.375rem' }}>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
            3/4 corners pass. FF corner at 125°C exceeds 1% error threshold.
          </span>
        </div>
      </div>

      <div style={{ ...S.card(), gridColumn: 'span 2' }}>
        <div style={S.cardTitle}><GitBranch size={13} />Simulation History</div>
        <div style={{ display: 'grid', gridTemplateColumns: '120px 100px 80px 80px 80px 1fr', gap: '0.5rem', padding: '0 0.625rem 0.375rem', borderBottom: '1px solid #334155' }}>
          {['Run ID', 'Timestamp', 'N', 'W (µm)', 'TC', 'Notes'].map((h) => (
            <span key={h} style={{ fontSize: '0.65rem', color: '#475569', fontWeight: '600', textTransform: 'uppercase' }}>{h}</span>
          ))}
        </div>
        {[
          { id: 'run-028', ts: '02:54:12', n: '8.2', w: '5.4', tc: '12.4', note: 'Best — exported' },
          { id: 'run-027', ts: '02:53:01', n: '7.9', w: '4.2', tc: '19.1', note: '' },
          { id: 'run-015', ts: '02:50:44', n: '9.1', w: '6.8', tc: '24.5', note: 'FF corner fail' },
        ].map((r) => (
          <div key={r.id} style={{ display: 'grid', gridTemplateColumns: '120px 100px 80px 80px 80px 1fr', gap: '0.5rem', padding: '0.375rem 0.625rem', borderBottom: '1px solid #0f172a' }}>
            <span style={{ fontSize: '0.75rem', color: '#38bdf8' }}>{r.id}</span>
            <span style={{ fontSize: '0.7rem', color: '#64748b' }}>{r.ts}</span>
            <span style={{ fontSize: '0.7rem', color: '#f8fafc' }}>{r.n}</span>
            <span style={{ fontSize: '0.7rem', color: '#f8fafc' }}>{r.w}</span>
            <span style={{ fontSize: '0.7rem', color: '#f8fafc' }}>{r.tc}</span>
            <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{r.note}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function LogsTab({ logLines, logRef }) {
  const lines = logLines || LOG_LINES;
  return (
    <div style={{ ...S.card(), height: '100%', gap: '0.75rem' }}>
      <div style={{ ...S.cardTitle, justifyContent: 'space-between' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Terminal size={13} />Optimizer Log
        </span>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button style={S.btn('secondary', { fontSize: '0.7rem', padding: '0.2rem 0.5rem' })}>
            <RefreshCw size={11} />Clear
          </button>
          <button style={S.btn('secondary', { fontSize: '0.7rem', padding: '0.2rem 0.5rem' })}>
            <Save size={11} />Save
          </button>
        </div>
      </div>
      <div style={{
        backgroundColor: '#020617',
        borderRadius: '0.375rem',
        border: '1px solid #334155',
        padding: '0.75rem 1rem',
        fontFamily: 'monospace',
        fontSize: '0.75rem',
        color: '#94a3b8',
        flex: 1,
        overflowY: 'auto',
        lineHeight: '1.7',
      }} ref={logRef}>
        {lines.map((line, i) => {
          const isWarn = line.includes('WARN') || line.includes('FAIL');
          const isApi = line.includes('[API]');
          return (
            <div key={i} style={{ color: isWarn ? '#fbbf24' : isApi ? '#38bdf8' : '#94a3b8' }}>
              {line}
            </div>
          );
        })}
        <div style={{ color: '#38bdf8', marginTop: '0.5rem' }}>█</div>
      </div>
    </div>
  );
}

// --- Main App ---

const NAV_ITEMS = [
  { id: 'optimization', label: 'Optimization', icon: TrendingUp },
  { id: 'layout', label: 'Layout Viewer', icon: Layers },
  { id: 'verification', label: 'Verification', icon: CheckCircle2 },
  { id: 'logs', label: 'Logs', icon: Terminal },
];

/** Convert raw optimizer history entries into candidate rows for the table. */
function _historyToCandidates(history) {
  return [...history]
    .filter((e) => e.vref_V != null)
    .sort((a, b) => Math.abs(a.vref_V - 1.2) - Math.abs(b.vref_V - 1.2))
    .slice(0, 3)
    .map((e) => {
      const checks = [e.spec_vref_pass, e.spec_iq_pass, e.spec_psrr_pass].filter((v) => v != null);
      const allKnownChecksPass = checks.length > 0 && checks.every(Boolean);

      return {
        id: `BG-${String(e.iteration + 1).padStart(3, '0')}`,
        variables: `N=${e.params.N}, W=${((e.params.W_P || 4e-6) * 1e6).toFixed(1)}µm`,
        vref_mV: (e.vref_V * 1000).toFixed(1),
        err_mV: Math.abs(e.vref_V * 1000 - 1200).toFixed(1),
        iq_uA: e.iq_uA != null ? Number(e.iq_uA.toFixed(2)) : null,
        psrr_dB: e.psrr_dB != null ? Number(e.psrr_dB.toFixed(1)) : null,
        power: e.iq_uA != null ? `${(e.iq_uA * 1.8).toFixed(1)}µW` : '—',
        status: allKnownChecksPass ? 'pass' : 'fail',
      };
    });
}

export default function App() {
  const [activeTab, setActiveTab] = useState('optimization');
  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedCandidate, setSelectedCandidate] = useState(CANDIDATES[0]);
  const [layoutLayer, setLayoutLayer] = useState('all');
  const [designValues, setDesignValues] = useState(
    Object.fromEntries(DESIGN_VARS.map((v) => [v.id, v.default]))
  );

  // --- Backend state ---
  const [serverStatus, setServerStatus] = useState({ ok: null, ngspice_available: null });
  const [optimResult, setOptimResult] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [liveEstimate, setLiveEstimate] = useState(null);
  const [isEstimating, setIsEstimating] = useState(false);
  const [estimateError, setEstimateError] = useState(null);
  const [layoutData, setLayoutData] = useState(null);
  const [layoutError, setLayoutError] = useState(null);
  const logRef = useRef(null);
  const streamRef = useRef(null);

  // Probe backend status on mount
  useEffect(() => {
    fetch('/api/status')
      .then((r) => r.json())
      .then((data) => setServerStatus(data))
      .catch(() => setServerStatus({ ok: false, ngspice_available: false }));

    fetch('/api/layout/preview?seed=42&patch_size=32')
      .then((r) => r.json())
      .then((data) => setLayoutData(data))
      .catch(() => setLayoutError('Could not load layout preview'));
  }, []);

  // Scroll log panel to bottom when new results arrive
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [optimResult]);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.close();
        streamRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const timer = setTimeout(async () => {
      setIsEstimating(true);
      setEstimateError(null);

      const nRatio = Math.max(1, Math.round(designValues.r_ratio));
      const iBiasA = Math.max(0.1, designValues.i_bias) * 1e-6;
      const r1 = 1.8 / iBiasA;
      const simParams = {
        N: nRatio,
        R1: r1,
        R2: r1 / nRatio,
        W_P: designValues.w_m1 * 1e-6,
        L_P: 1e-6,
      };

      try {
        const resp = await fetch('/api/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ params: simParams }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
          throw new Error(data.error || 'estimate failed');
        }

        setLiveEstimate({
          vref_mV: data.vref_V != null ? data.vref_V * 1000 : null,
          iq_uA: data.iq_uA ?? null,
          spec_vref: data.spec_checks?.vref ?? null,
        });
      } catch (err) {
        setEstimateError(err.message);
      } finally {
        setIsEstimating(false);
      }
    }, 350);

    return () => clearTimeout(timer);
  }, [designValues]);

  // Derive live candidates from optimizer history (top 3 by closest Vref)
  const liveCandidates = optimResult
    ? _historyToCandidates(optimResult.history)
    : CANDIDATES;

  // Derive live convergence from optimizer response
  const liveConvergence = optimResult?.convergence?.length
    ? optimResult.convergence
        .filter((c) => c.best_error_V != null)
        .map((c) => ({ iter: c.iter + 1, loss: c.best_error_V }))
    : CONVERGENCE;

  // Derive live log lines from optimizer history
  const liveLogs = optimResult
    ? [
        `[API] Optimizer finished — ${optimResult.n_simulations} simulations, ` +
          `${optimResult.n_spec_pass} spec-pass (${(optimResult.spec_pass_rate * 100).toFixed(0)}%)`,
        `[API] Best Vref: ${optimResult.best_vref_V != null ? (optimResult.best_vref_V * 1000).toFixed(2) + ' mV' : 'N/A'}`,
        ...optimResult.history.map(
          (e) =>
            `[${String(e.iteration).padStart(2, '0')}] ${e.source.toUpperCase().padEnd(3)} ` +
            `vref=${e.vref_V != null ? (e.vref_V * 1000).toFixed(2) + ' mV' : 'err'} ` +
            `spec=${e.spec_vref_pass ? 'PASS' : 'FAIL'} ` +
            `(${e.sim_time_s}s)`
        ),
      ]
    : LOG_LINES;

  const handleRunOptimizer = async () => {
    setIsSimulating(true);
    setApiError(null);
    setOptimResult({
      best_params: {},
      best_vref_V: null,
      n_simulations: 0,
      n_spec_pass: 0,
      spec_pass_rate: 0,
      history: [],
      convergence: [],
    });

    if (streamRef.current) {
      streamRef.current.close();
      streamRef.current = null;
    }

    const query = new URLSearchParams({ budget: '20', n_init: '5', seed: '42' });
    const source = new EventSource(`/api/optimize/stream?${query.toString()}`);
    streamRef.current = source;

    source.addEventListener('progress', (evt) => {
      const payload = JSON.parse(evt.data);
      setOptimResult((prev) => {
        const current = prev || {
          best_params: {},
          best_vref_V: null,
          n_simulations: 0,
          n_spec_pass: 0,
          spec_pass_rate: 0,
          history: [],
          convergence: [],
        };

        const nextHistory = [...(current.history || []), payload.entry];
        const nextConvergence = [
          ...(current.convergence || []),
          { iter: payload.iteration, best_error_V: payload.best_error_V },
        ];

        let bestVref = current.best_vref_V;
        if (payload.entry?.vref_V != null) {
          if (bestVref == null || Math.abs(payload.entry.vref_V - 1.2) < Math.abs(bestVref - 1.2)) {
            bestVref = payload.entry.vref_V;
          }
        }

        return {
          ...current,
          history: nextHistory,
          convergence: nextConvergence,
          n_simulations: payload.n_simulations,
          n_spec_pass: payload.n_spec_pass,
          spec_pass_rate: payload.spec_pass_rate,
          best_vref_V: bestVref,
        };
      });
    });

    source.addEventListener('final', (evt) => {
      const data = JSON.parse(evt.data);
      setOptimResult(data);
      if (data.history?.length) {
        const topCands = _historyToCandidates(data.history);
        if (topCands.length) setSelectedCandidate(topCands[0]);
      }
    });

    source.addEventListener('api_error', (evt) => {
      const data = JSON.parse(evt.data);
      setApiError(data.error || 'Streaming optimize failed');
    });

    source.addEventListener('done', () => {
      setIsSimulating(false);
      source.close();
      if (streamRef.current === source) streamRef.current = null;
    });

    source.onerror = () => {
      if (streamRef.current === source) {
        setApiError('Optimization stream disconnected');
        setIsSimulating(false);
        source.close();
        streamRef.current = null;
      }
    };
  };

  // Summarise backend status for bottom bar
  const statusDot = serverStatus.ok === false
    ? '#f87171'
    : serverStatus.ngspice_available
      ? '#6ee7b7'
      : '#fbbf24';
  const statusText = serverStatus.ok === null
    ? 'Connecting…'
    : serverStatus.ok === false
      ? 'API offline'
      : serverStatus.ngspice_available
        ? 'Ready (ngspice)'
        : 'Ready (synthetic)';

  const bestEntry = optimResult?.history?.filter((e) => e.vref_V != null)
    .sort((a, b) => Math.abs(a.vref_V - 1.2) - Math.abs(b.vref_V - 1.2))[0];

  return (
    <div style={S.appWrapper}>
      {/* Top bar */}
      <div style={S.topBar}>
        <div style={S.topBarLeft}>
          <Cpu size={20} color="#38bdf8" />
          <div>
            <div style={S.logoText}>VLSI-AI Design Studio</div>
            <div style={S.subText}>Bandgap Reference — v0.1 · ngspice + GP surrogate</div>
          </div>
        </div>
        <div style={S.topBarRight}>
          <button style={S.btn()} title="Simulation history"><History size={14} /></button>
          <button style={S.btn()} title="Branch"><GitBranch size={14} /></button>
          <button style={S.btn()} title="Settings"><Settings size={14} /></button>
          <button style={S.btn('primary')}>
            <Save size={13} />Save Project
          </button>
        </div>
      </div>

      <div style={S.contentArea}>
        {/* Sidebar */}
        <div style={S.sidebar}>
          <div style={S.sideSection}>Workflow</div>
          {NAV_ITEMS.map((item) => (
            <button key={item.id} style={S.navItem(activeTab === item.id)} onClick={() => setActiveTab(item.id)}>
              <item.icon size={15} />
              {item.label}
            </button>
          ))}
          <div style={{ marginTop: 'auto' }}>
            <div style={S.sideSection}>Project</div>
            <button style={S.navItem(false)}>
              <Database size={15} />Datasets
            </button>
            <button style={S.navItem(false)}>
              <FileCode size={15} />Netlists
            </button>
          </div>
        </div>

        {/* Main content */}
        <div style={S.main}>
          {apiError && (
            <div style={{
              backgroundColor: '#7f1d1d', color: '#fca5a5',
              borderRadius: '0.375rem', padding: '0.5rem 0.875rem',
              marginBottom: '0.75rem', fontSize: '0.8rem',
              display: 'flex', alignItems: 'center', gap: '0.5rem',
            }}>
              <AlertTriangle size={14} />
              <span>API error: {apiError}</span>
            </div>
          )}
          {activeTab === 'optimization' && (
            <OptimizationTab
              isSimulating={isSimulating}
              onRun={handleRunOptimizer}
              selectedCandidate={selectedCandidate}
              setSelectedCandidate={setSelectedCandidate}
              designValues={designValues}
              setDesignValues={setDesignValues}
              candidates={liveCandidates}
              convergenceData={liveConvergence}
              optimResult={optimResult}
              liveEstimate={liveEstimate}
              isEstimating={isEstimating}
              estimateError={estimateError}
            />
          )}
          {activeTab === 'layout' && (
            <LayoutTab
              layer={layoutLayer}
              setLayer={setLayoutLayer}
              layoutData={layoutData}
              layoutError={layoutError}
            />
          )}
          {activeTab === 'verification' && <VerificationTab />}
          {activeTab === 'logs' && <LogsTab logLines={liveLogs} logRef={logRef} />}
        </div>
      </div>

      {/* Bottom bar */}
      <div style={S.bottomBar}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: statusDot }} />
            {statusText}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {optimResult ? (
            <>
              <span>{optimResult.n_spec_pass}/{optimResult.n_simulations} spec-pass</span>
              <span>Best Vref: {bestEntry ? (bestEntry.vref_V * 1000).toFixed(1) + ' mV' : '—'}</span>
              <span>iter {optimResult.n_simulations}</span>
            </>
          ) : (
            <>
              <span>3/4 corners pass</span>
              <span>Best TC: 12.4 ppm/C</span>
              <span>iter 28/50</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
