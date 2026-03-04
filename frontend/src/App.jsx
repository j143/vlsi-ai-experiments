import { useState } from 'react';
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

function CandidateRow({ c, selected, onClick }) {
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
        borderLeft: selected ? '2px solid #38bdf8' : '2px solid transparent',
        marginBottom: '0.25rem',
      }}
    >
      <span style={{ fontSize: '0.75rem', color: '#38bdf8', fontWeight: '600' }}>{c.id}</span>
      <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{c.variables}</span>
      <span style={{ fontSize: '0.7rem', color: '#f8fafc', textAlign: 'right' }}>{c.surrogate}</span>
      <span style={{ fontSize: '0.7rem', color: '#94a3b8', textAlign: 'right' }}>{c.spice}</span>
      <span style={{ fontSize: '0.7rem', color: '#94a3b8', textAlign: 'right' }}>{c.power}</span>
      <div style={{ display: 'flex', justifyContent: 'center' }}>
        <span style={S.badge(c.status)}>{c.status}</span>
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
    </svg>
  );
}

// Simple layout mock (SVG floorplan)
function LayoutViewer({ layer }) {
  const layerColorMap = {
    all: ['#1d4ed8', '#15803d', '#b45309', '#7c3aed'],
    metal1: ['#1d4ed8'],
    diffusion: ['#15803d'],
    poly: ['#b45309'],
    via: ['#7c3aed'],
  };
  const cols = layerColorMap[layer] || layerColorMap.all;

  const cells = [
    { x: 20, y: 20, w: 80, h: 40, label: 'M1/M2', layerIdx: 0 },
    { x: 120, y: 20, w: 60, h: 40, label: 'Q1', layerIdx: 1 },
    { x: 200, y: 20, w: 60, h: 40, label: 'Q2', layerIdx: 1 },
    { x: 20, y: 80, w: 100, h: 35, label: 'R1', layerIdx: 2 },
    { x: 140, y: 80, w: 50, h: 35, label: 'R2', layerIdx: 2 },
    { x: 210, y: 80, w: 50, h: 35, label: 'C1', layerIdx: 3 },
    { x: 20, y: 135, w: 240, h: 20, label: 'GND Rail', layerIdx: 0 },
  ];

  return (
    <svg
      width="100%"
      viewBox="0 0 290 175"
      style={{ backgroundColor: '#020617', borderRadius: '0.375rem', border: '1px solid #334155' }}
    >
      {Array.from({ length: 10 }).map((_, i) => (
        <line key={`h${i}`} x1={0} x2={290} y1={i * 20} y2={i * 20} stroke="#0f172a" strokeWidth="0.5" />
      ))}
      {Array.from({ length: 15 }).map((_, i) => (
        <line key={`v${i}`} x1={i * 20} x2={i * 20} y1={0} y2={175} stroke="#0f172a" strokeWidth="0.5" />
      ))}
      {cells.map((c) => {
        const fill = cols[c.layerIdx % cols.length];
        return (
          <g key={c.label}>
            <rect x={c.x} y={c.y} width={c.w} height={c.h}
              fill={fill + '40'} stroke={fill} strokeWidth="1.2" rx="2" />
            <text x={c.x + c.w / 2} y={c.y + c.h / 2 + 4}
              fill="#f8fafc" fontSize="9" textAnchor="middle">{c.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

// --- Tab views ---

function OptimizationTab({ isSimulating, onRun, selectedCandidate, setSelectedCandidate, designValues, setDesignValues }) {
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
            {['ID', 'Variables', 'ML TC', 'Spice TC', 'Power', 'Status'].map((h) => (
              <span key={h} style={{ fontSize: '0.65rem', color: '#475569', fontWeight: '600', textTransform: 'uppercase' }}>{h}</span>
            ))}
          </div>
          <div style={{ marginTop: '0.375rem' }}>
            {CANDIDATES.map((c) => (
              <CandidateRow
                key={c.id}
                c={c}
                selected={selectedCandidate?.id === c.id}
                onClick={() => setSelectedCandidate(c)}
              />
            ))}
          </div>
        </div>

        <div style={S.card()}>
          <div style={S.cardTitle}><BarChart3 size={13} />Optimizer Convergence</div>
          <ConvergenceChart data={CONVERGENCE} />
          <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
            <span style={S.label}>Best loss: <strong style={{ color: '#38bdf8' }}>0.091</strong></span>
            <span style={S.label}>Converged at iter: <strong style={{ color: '#38bdf8' }}>28</strong></span>
          </div>
        </div>
      </div>

      {/* Right: Selected candidate + corner analysis */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={S.card()}>
          <div style={S.cardTitle}><Eye size={13} />Selected: {selectedCandidate?.id}</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
            {[
              { label: 'Vref', value: '1199.4 mV' },
              { label: 'TempCo', value: (selectedCandidate?.spice ?? '—') + ' ppm/C' },
              { label: 'PSRR', value: '-61.2 dB' },
              { label: 'Iq', value: '9.8 µA' },
              { label: 'Power', value: selectedCandidate?.power ?? '—' },
              {
                label: 'ML Error',
                value: selectedCandidate
                  ? `${(Math.abs(parseFloat(selectedCandidate.surrogate) - parseFloat(selectedCandidate.spice)) / parseFloat(selectedCandidate.spice) * 100).toFixed(1)}%`
                  : '—',
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

function LayoutTab({ layer, setLayer }) {
  const layerOptions = ['all', 'metal1', 'diffusion', 'poly', 'via'];

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
        <LayoutViewer layer={layer} />
        <div style={{ display: 'flex', gap: '1rem' }}>
          <span style={S.label}>DRC: <strong style={{ color: '#6ee7b7' }}>0 errors</strong></span>
          <span style={S.label}>LVS: <strong style={{ color: '#6ee7b7' }}>clean</strong></span>
          <span style={S.label}>Area: <strong style={{ color: '#f8fafc' }}>42.6 µm²</strong></span>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={S.card()}>
          <div style={S.cardTitle}><Database size={13} />Layer Legend</div>
          {[
            { color: '#1d4ed8', label: 'Metal 1 / Power Rail' },
            { color: '#15803d', label: 'Active / Diffusion' },
            { color: '#b45309', label: 'Poly / Gate' },
            { color: '#7c3aed', label: 'Via / Contact' },
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

function LogsTab() {
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
      }}>
        {LOG_LINES.map((line, i) => {
          const isWarn = line.includes('WARN');
          const isInfo = line.includes('INFO');
          return (
            <div key={i} style={{ color: isWarn ? '#fbbf24' : isInfo ? '#94a3b8' : '#f8fafc' }}>
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

export default function App() {
  const [activeTab, setActiveTab] = useState('optimization');
  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedCandidate, setSelectedCandidate] = useState(CANDIDATES[0]);
  const [layoutLayer, setLayoutLayer] = useState('all');
  const [designValues, setDesignValues] = useState(
    Object.fromEntries(DESIGN_VARS.map((v) => [v.id, v.default]))
  );

  const handleRunOptimizer = () => {
    setIsSimulating(true);
    setTimeout(() => setIsSimulating(false), 2000);
  };

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
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button key={id} style={S.navItem(activeTab === id)} onClick={() => setActiveTab(id)}>
              <Icon size={15} />
              {label}
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
          {activeTab === 'optimization' && (
            <OptimizationTab
              isSimulating={isSimulating}
              onRun={handleRunOptimizer}
              selectedCandidate={selectedCandidate}
              setSelectedCandidate={setSelectedCandidate}
              designValues={designValues}
              setDesignValues={setDesignValues}
            />
          )}
          {activeTab === 'layout' && (
            <LayoutTab layer={layoutLayer} setLayer={setLayoutLayer} />
          )}
          {activeTab === 'verification' && <VerificationTab />}
          {activeTab === 'logs' && <LogsTab />}
        </div>
      </div>

      {/* Bottom bar */}
      <div style={S.bottomBar}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#6ee7b7' }} />
            Ready
          </span>
          <span>ngspice 41 · Python 3.10</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span>3/4 corners pass</span>
          <span>Best TC: 12.4 ppm/C</span>
          <span>iter 28/50</span>
        </div>
      </div>
    </div>
  );
}
