import { useState, useRef } from "react";

const MODELS = [
  { id: "llama-3.1-8b", name: "Llama 3.1 8B", family: "Meta", params: "8B", precision: ["fp16", "int8", "int4"], maxCtx: 128000, gpuMem: { fp16: 16, int8: 8, int4: 5 } },
  { id: "qwen2.5-7b", name: "Qwen 2.5 7B", family: "Alibaba", params: "7B", precision: ["fp16", "int8", "int4"], maxCtx: 131072, gpuMem: { fp16: 14, int8: 7, int4: 4 } },
  { id: "gemma-2-9b", name: "Gemma 2 9B", family: "Google", params: "9B", precision: ["fp16", "int8"], maxCtx: 8192, gpuMem: { fp16: 18, int8: 9, int4: null } },
  { id: "mistral-7b", name: "Mistral 7B v0.3", family: "Mistral AI", params: "7B", precision: ["fp16", "int8", "int4"], maxCtx: 32768, gpuMem: { fp16: 14, int8: 7, int4: 4 } },
  { id: "llama-3.1-70b", name: "Llama 3.1 70B", family: "Meta", params: "70B", precision: ["int8", "int4"], maxCtx: 128000, gpuMem: { fp16: null, int8: 70, int4: 40 } },
];

const RUNTIMES = [
  { id: "vllm", name: "vLLM", desc: "High-throughput continuous batching", badge: "Recommended", features: ["PagedAttention", "Continuous batching", "Tensor parallelism", "Flash Attention"] },
  { id: "sglang", name: "SGLang", desc: "Structured generation & RadixAttention", badge: "Fast TTFT", features: ["RadixAttention", "Structured output", "Speculative decoding", "Low TTFT"] },
  { id: "ollama", name: "Ollama", desc: "Easy local deployment", badge: "Easiest", features: ["Simple API", "Auto quantization", "Model library", "CPU fallback"] },
  { id: "triton", name: "Triton + TRT-LLM", desc: "NVIDIA-optimized inference", badge: "Max GPU", features: ["TensorRT kernels", "In-flight batching", "FP8 precision", "Multi-GPU"] },
];

const GPU_TYPES = [
  { id: "a100-40", name: "A100 40GB", mem: 40, tflops: 312 },
  { id: "a100-80", name: "A100 80GB", mem: 80, tflops: 312 },
  { id: "h100", name: "H100 80GB", mem: 80, tflops: 989 },
  { id: "l40s", name: "L40S 48GB", mem: 48, tflops: 362 },
];

const SEQ_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"];
const TABS = ["Deploy", "Benchmark", "KV Cache", "API Explorer"];

const DEPLOYMENTS = [
  { model: "Llama 3.1 8B", runtime: "vLLM", status: "running", rps: 42, gpu: "A100 40GB" },
  { model: "Qwen 2.5 7B", runtime: "SGLang", status: "running", rps: 18, gpu: "L40S" },
  { model: "Gemma 2 9B", runtime: "Ollama", status: "stopped", rps: 0, gpu: "A100 40GB" },
];

const MOCK_RESPONSES = {
  "Explain KV cache in 2 sentences.": "A KV (Key-Value) cache in LLM inference stores the attention keys and values computed for previous tokens so they don't need recomputing on each new generation step. This reduces decoding cost from O(n²) to O(n) per step, at the expense of GPU memory proportional to sequence length.",
  default: "The Jarvis Labs inference API provides OpenAI-compatible endpoints for serving open-source models. Point your existing OpenAI SDK to our base URL and your code works without modification.",
};

// ── Shared UI ──
function Badge({ children, color = "info" }) {
  const bg = { info: "#dbeafe", success: "#dcfce7", warning: "#fef9c3", danger: "#fee2e2" };
  const tc = { info: "#1d4ed8", success: "#166534", warning: "#854d0e", danger: "#991b1b" };
  return (
    <span style={{ fontSize: 10, fontWeight: 500, padding: "2px 8px", borderRadius: 999, background: bg[color], color: tc[color], textTransform: "uppercase", letterSpacing: ".06em", whiteSpace: "nowrap" }}>
      {children}
    </span>
  );
}

function MetricCard({ label, value, unit, sub, color }) {
  return (
    <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: "12px 16px", minWidth: 0 }}>
      <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 4, textTransform: "uppercase", letterSpacing: ".05em" }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 500, color: color || "var(--color-text-primary)", lineHeight: 1.2 }}>
        {value}<span style={{ fontSize: 13, fontWeight: 400, color: "var(--color-text-secondary)", marginLeft: 4 }}>{unit}</span>
      </div>
      {sub && <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function StatusDot({ status }) {
  const c = { running: "#22c55e", stopped: "#6b7280", error: "#ef4444" };
  return <span style={{ display: "inline-block", width: 7, height: 7, borderRadius: "50%", background: c[status] || "#6b7280", marginRight: 6 }} />;
}

// ── Deploy Tab ──
function DeployTab() {
  const [step, setStep] = useState(0);
  const [model, setModel] = useState(MODELS[0]);
  const [runtime, setRuntime] = useState(RUNTIMES[0]);
  const [gpu, setGpu] = useState(GPU_TYPES[0]);
  const [precision, setPrecision] = useState("fp16");
  const [replicas, setReplicas] = useState(1);
  const [ctxLen, setCtxLen] = useState(4096);
  const [maxBatch, setMaxBatch] = useState(32);
  const [deployed, setDeployed] = useState(false);
  const [deploying, setDeploying] = useState(false);
  const [logs, setLogs] = useState([]);
  const logsRef = useRef(null);

  const memNeeded = model.gpuMem[precision] || 0;
  const memFits = gpu.mem >= memNeeded;
  const gpusNeeded = Math.ceil(memNeeded / gpu.mem);
  const kvCacheAvail = ((gpu.mem * replicas - memNeeded) * 0.85).toFixed(1);
  const rm = { vllm: 1, sglang: 1.15, ollama: 0.7, triton: 1.3 };
  const pm = { int4: 1.4, int8: 1.2, fp16: 1.0 };
  const estThr = Math.round((rm[runtime.id] || 1) * (gpu.tflops / 312) * (pm[precision] || 1) * 180 * replicas);
  const estLat = Math.round(200 / ((gpu.tflops / 312) * (pm[precision] || 1) * (runtime.id === "sglang" ? 1.2 : 1)));

  const DEPLOY_LOGS = [
    "[INFO] Pulling model weights...",
    `[INFO] Loading ${model.name} (${precision})`,
    `[INFO] Initializing ${runtime.name} runtime`,
    `[INFO] Allocating GPU memory on ${gpu.name}`,
    `[INFO] KV cache pool: ${kvCacheAvail} GB`,
    `[INFO] Setting max batch size: ${maxBatch}`,
    `[INFO] Context length: ${ctxLen.toLocaleString()} tokens`,
    "[INFO] Starting HTTP server on :8000",
    "[SUCCESS] Endpoint ready → https://api.jarvislabs.ai/v1",
  ];

  function startDeploy() {
    setDeploying(true); setLogs([]);
    DEPLOY_LOGS.forEach((l, i) => {
      setTimeout(() => {
        setLogs(prev => [...prev, l]);
        if (logsRef.current) logsRef.current.scrollTop = 9999;
        if (i === DEPLOY_LOGS.length - 1) { setDeploying(false); setDeployed(true); }
      }, i * 420);
    });
  }

  const steps = ["Model", "Runtime", "Hardware", "Config", "Deploy"];
  const nav = (
    <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)", marginBottom: 24 }}>
      {steps.map((s, i) => (
        <button key={s} onClick={() => setStep(i)} style={{ padding: "8px 14px", fontSize: 13, fontWeight: step === i ? 500 : 400, color: step === i ? "var(--color-text-primary)" : "var(--color-text-secondary)", background: "none", border: "none", borderBottom: step === i ? "2px solid var(--color-text-primary)" : "2px solid transparent", cursor: "pointer", marginBottom: -1 }}>
          {i + 1}. {s}
        </button>
      ))}
    </div>
  );

  const btnStyle = { padding: "8px 20px", fontSize: 13, borderRadius: 8, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-primary)", color: "var(--color-text-primary)", cursor: "pointer" };

  return (
    <div>
      {nav}
      {step === 0 && (
        <div>
          <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 16 }}>Choose an open model to serve.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
            {MODELS.map(m => (
              <div key={m.id} onClick={() => { setModel(m); setPrecision(m.precision[0]); }} style={{ border: model.id === m.id ? "2px solid #3b82f6" : "0.5px solid var(--color-border-tertiary)", borderRadius: 12, padding: 14, cursor: "pointer", background: "var(--color-background-primary)" }}>
                <div style={{ fontWeight: 500, fontSize: 13, marginBottom: 4 }}>{m.name}</div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 8 }}>{m.family} · {m.params}</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>{m.precision.map(p => <Badge key={p} color="info">{p}</Badge>)}</div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 8 }}>Max ctx: {(m.maxCtx / 1000).toFixed(0)}K tokens</div>
              </div>
            ))}
          </div>
          <button style={{ ...btnStyle, marginTop: 20 }} onClick={() => setStep(1)}>Next →</button>
        </div>
      )}

      {step === 1 && (
        <div>
          <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 16 }}>Select an inference runtime.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
            {RUNTIMES.map(r => (
              <div key={r.id} onClick={() => setRuntime(r)} style={{ border: runtime.id === r.id ? "2px solid #3b82f6" : "0.5px solid var(--color-border-tertiary)", borderRadius: 12, padding: 14, cursor: "pointer", background: "var(--color-background-primary)" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
                  <div style={{ fontWeight: 500, fontSize: 14 }}>{r.name}</div><Badge color="success">{r.badge}</Badge>
                </div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 10 }}>{r.desc}</div>
                {r.features.map(f => <div key={f} style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 2 }}>· {f}</div>)}
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 20 }}>
            <button style={btnStyle} onClick={() => setStep(0)}>← Back</button>
            <button style={btnStyle} onClick={() => setStep(2)}>Next →</button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div>
          <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 16 }}>Choose GPU type and precision.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10, marginBottom: 20 }}>
            {GPU_TYPES.map(g => (
              <div key={g.id} onClick={() => setGpu(g)} style={{ border: gpu.id === g.id ? "2px solid #3b82f6" : "0.5px solid var(--color-border-tertiary)", borderRadius: 12, padding: 14, cursor: "pointer", background: "var(--color-background-primary)" }}>
                <div style={{ fontWeight: 500, fontSize: 13 }}>{g.name}</div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 4 }}>{g.mem} GB VRAM</div>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{g.tflops} TFLOPS</div>
              </div>
            ))}
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>Quantization / Precision</div>
            <div style={{ display: "flex", gap: 8 }}>
              {model.precision.map(p => (
                <button key={p} onClick={() => setPrecision(p)} style={{ padding: "6px 16px", fontSize: 12, cursor: "pointer", borderRadius: 8, border: precision === p ? "2px solid #3b82f6" : "0.5px solid var(--color-border-secondary)", background: precision === p ? "#dbeafe" : "var(--color-background-primary)", color: precision === p ? "#1d4ed8" : "var(--color-text-primary)" }}>{p}</button>
              ))}
            </div>
          </div>
          <div style={{ padding: "12px 16px", borderRadius: 8, border: "0.5px solid var(--color-border-tertiary)", background: "var(--color-background-secondary)", fontSize: 12, marginBottom: 16 }}>
            <div style={{ marginBottom: 4 }}>Model VRAM: <strong>{memNeeded} GB</strong> · GPU VRAM: <strong>{gpu.mem} GB</strong></div>
            {memFits ? <div style={{ color: "#166534" }}>✓ Fits on 1× {gpu.name}</div> : <div style={{ color: "#854d0e" }}>⚠ Needs {gpusNeeded}× {gpu.name} (tensor parallelism)</div>}
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <button style={btnStyle} onClick={() => setStep(1)}>← Back</button>
            <button style={btnStyle} onClick={() => setStep(3)}>Next →</button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div>
          <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 20 }}>Tune serving parameters.</p>
          {[
            { label: "Replicas", min: 1, max: 8, val: replicas, set: setReplicas, unit: "x" },
            { label: "Max context length", min: 512, max: model.maxCtx, val: ctxLen, set: setCtxLen, unit: "tokens", step: 512 },
            { label: "Max batch size", min: 1, max: 256, val: maxBatch, set: setMaxBatch, unit: "req" },
          ].map(({ label, min, max, val, set, unit, step: s }) => (
            <div key={label} style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontSize: 13, fontWeight: 500 }}>{label}</span>
                <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>{val.toLocaleString()} {unit}</span>
              </div>
              <input type="range" min={min} max={max} step={s || 1} value={val} onChange={e => set(Number(e.target.value))} />
            </div>
          ))}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 20 }}>
            <MetricCard label="Est. Throughput" value={estThr.toLocaleString()} unit="tok/s" />
            <MetricCard label="Est. TTFT" value={estLat} unit="ms" />
            <MetricCard label="KV Cache" value={kvCacheAvail} unit="GB" />
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <button style={btnStyle} onClick={() => setStep(2)}>← Back</button>
            <button style={btnStyle} onClick={() => setStep(4)}>Review →</button>
          </div>
        </div>
      )}

      {step === 4 && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
            {[["Model", model.name], ["Runtime", runtime.name], ["GPU", `${replicas}× ${gpu.name}`], ["Precision", precision], ["Context", `${ctxLen.toLocaleString()} tokens`], ["Batch", `${maxBatch} req`]].map(([k, v]) => (
              <div key={k} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", paddingBottom: 10 }}>
                <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 2 }}>{k}</div>
                <div style={{ fontSize: 14, fontWeight: 500 }}>{v}</div>
              </div>
            ))}
          </div>
          {!deployed && (
            <button onClick={startDeploy} disabled={deploying} style={{ ...btnStyle, padding: "10px 28px", fontSize: 14, fontWeight: 500, opacity: deploying ? 0.6 : 1 }}>
              {deploying ? "Deploying..." : "Deploy Endpoint"}
            </button>
          )}
          {logs.length > 0 && (
            <div ref={logsRef} style={{ marginTop: 16, background: "#0f1117", borderRadius: 8, padding: "14px 16px", fontFamily: "var(--font-mono)", fontSize: 11, color: "#d1fae5", maxHeight: 200, overflowY: "auto" }}>
              {logs.map((l, i) => <div key={i} style={{ marginBottom: 4, color: l.includes("SUCCESS") ? "#4ade80" : l.includes("INFO") ? "#93c5fd" : "#d1fae5" }}>{l}</div>)}
            </div>
          )}
          {deployed && (
            <div style={{ marginTop: 16, padding: "14px 16px", borderRadius: 8, border: "0.5px solid #86efac", background: "#dcfce7" }}>
              <div style={{ fontWeight: 500, fontSize: 13, color: "#166534", marginBottom: 4 }}>Endpoint live</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "#555" }}>https://api.jarvislabs.ai/v1/chat/completions</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Benchmark Tab ──
function BenchmarkTab() {
  const [bModel, setBModel] = useState(MODELS[0].id);
  const [bRuntime, setBRuntime] = useState("vllm");
  const [concurrency, setConcurrency] = useState(16);
  const [inputLen, setInputLen] = useState(512);
  const [outputLen, setOutputLen] = useState(256);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const btnStyle = { padding: "10px 28px", fontSize: 13, fontWeight: 500, borderRadius: 8, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-primary)", color: "var(--color-text-primary)", cursor: "pointer" };

  function runBenchmark() {
    setRunning(true); setResults(null); setProgress(0);
    const iv = setInterval(() => { setProgress(p => { if (p >= 100) { clearInterval(iv); return 100; } return p + 4; }); }, 80);
    setTimeout(() => {
      const rm = { vllm: 1, sglang: 1.18, ollama: 0.55, triton: 1.35 };
      const mult = rm[bRuntime] || 1;
      const base = 180 + Math.random() * 20;
      const res = Array.from({ length: 8 }, (_, i) => {
        const load = (i + 1) * (concurrency / 8);
        return { concurrency: Math.round(load), throughput: Math.round(base * mult * (1 - 0.003 * load) * (inputLen < 512 ? 1.1 : 1)), latency: Math.round(50 + load * 3.2 + (outputLen / 256) * 30 + Math.random() * 10), ttft: Math.round(20 + load * 1.1 + Math.random() * 5) };
      });
      setResults(res); setRunning(false);
    }, 2200);
  }

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div>
          <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 6 }}>Model</label>
          <select value={bModel} onChange={e => setBModel(e.target.value)}>{MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select>
        </div>
        <div>
          <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 6 }}>Runtime</label>
          <select value={bRuntime} onChange={e => setBRuntime(e.target.value)}>
            {[{ id: "vllm", name: "vLLM" }, { id: "sglang", name: "SGLang" }, { id: "ollama", name: "Ollama" }, { id: "triton", name: "Triton+TRT-LLM" }].map(r => <option key={r.id} value={r.id}>{r.name}</option>)}
          </select>
        </div>
      </div>
      {[{ label: "Max Concurrency", val: concurrency, set: setConcurrency, min: 1, max: 128 }, { label: "Input Length", val: inputLen, set: setInputLen, min: 64, max: 4096, step: 64, unit: "tokens" }, { label: "Output Length", val: outputLen, set: setOutputLen, min: 32, max: 2048, step: 32, unit: "tokens" }].map(({ label, val, set, min, max, step, unit }) => (
        <div key={label} style={{ marginBottom: 14 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontSize: 13 }}>{label}</span>
            <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>{val.toLocaleString()}{unit ? " " + unit : ""}</span>
          </div>
          <input type="range" min={min} max={max} step={step || 1} value={val} onChange={e => set(Number(e.target.value))} />
        </div>
      ))}
      <button onClick={runBenchmark} disabled={running} style={{ ...btnStyle, marginBottom: 20, opacity: running ? 0.6 : 1 }}>{running ? "Running benchmark..." : "Run Benchmark"}</button>
      {running && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 6 }}>Sending {concurrency} concurrent requests... {progress}%</div>
          <div style={{ height: 4, background: "var(--color-background-secondary)", borderRadius: 99, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${progress}%`, background: "var(--color-text-primary)", transition: "width .1s" }} />
          </div>
        </div>
      )}
      {results && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 20 }}>
            <MetricCard label="Peak Throughput" value={Math.max(...results.map(r => r.throughput)).toLocaleString()} unit="tok/s" color="#166534" />
            <MetricCard label="P50 Latency" value={results[Math.floor(results.length / 2)].latency} unit="ms" />
            <MetricCard label="Avg TTFT" value={Math.round(results.reduce((a, r) => a + r.ttft, 0) / results.length)} unit="ms" />
          </div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", fontSize: 12, borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                  {["Concurrency", "Throughput (tok/s)", "P50 Latency (ms)", "TTFT (ms)"].map(h => <th key={h} style={{ textAlign: "left", padding: "6px 12px", fontWeight: 500, color: "var(--color-text-secondary)", fontSize: 11 }}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
                    <td style={{ padding: "8px 12px" }}>{r.concurrency}</td>
                    <td style={{ padding: "8px 12px", fontWeight: 500 }}>{r.throughput.toLocaleString()}</td>
                    <td style={{ padding: "8px 12px" }}>{r.latency}</td>
                    <td style={{ padding: "8px 12px" }}>{r.ttft}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ── KV Cache Tab ──
function KVCacheTab() {
  const SLOTS = 48, PAGE_SIZE = 16, MAX_CTX = 4096;
  const [seqLen, setSeqLen] = useState(512);
  const [numSeqs, setNumSeqs] = useState(4);
  const [prefixCache, setPrefixCache] = useState(true);
  const [pagedAttn, setPagedAttn] = useState(true);

  const totalPages = Math.ceil(MAX_CTX / PAGE_SIZE);
  const pagesPerSeq = Math.ceil(seqLen / PAGE_SIZE);
  const prefixShared = prefixCache ? Math.floor(pagesPerSeq * 0.3) : 0;
  const effectiveUsed = pagesPerSeq * numSeqs - prefixShared * (numSeqs - 1);
  const utilPct = Math.min(100, Math.round((effectiveUsed / totalPages) * 100));
  const fragPct = pagedAttn ? Math.round(2 + numSeqs * 0.5) : Math.round(12 + numSeqs * 2);

  const blocks = Array.from({ length: SLOTS }, (_, i) => {
    let seqIdx = -1, isPrefix = false;
    for (let s = 0; s < numSeqs; s++) {
      const startPage = pagedAttn ? (s * (pagesPerSeq - prefixShared)) % (totalPages - prefixShared) : s * pagesPerSeq;
      if (i >= startPage && i < startPage + pagesPerSeq) { seqIdx = s; isPrefix = prefixCache && (i - startPage) < prefixShared; break; }
    }
    return { seqIdx, isPrefix };
  });

  return (
    <div>
      <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 20, lineHeight: 1.6 }}>
        Visualizes how GPU memory is partitioned into KV cache pages for concurrent sequences.
        PagedAttention eliminates fragmentation by decoupling logical and physical memory.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 18 }}>
        {[{ label: "Sequence length", val: seqLen, set: setSeqLen, min: 64, max: MAX_CTX, step: 64, unit: "tokens" }, { label: "Concurrent sequences", val: numSeqs, set: setNumSeqs, min: 1, max: 8 }].map(({ label, val, set, min, max, step, unit }) => (
          <div key={label}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 13 }}>{label}</span>
              <span style={{ fontSize: 13, color: "var(--color-text-secondary)" }}>{val.toLocaleString()}{unit ? " " + unit : ""}</span>
            </div>
            <input type="range" min={min} max={max} step={step || 1} value={val} onChange={e => set(Number(e.target.value))} />
          </div>
        ))}
      </div>
      <div style={{ display: "flex", gap: 20, marginBottom: 18 }}>
        {[{ label: "PagedAttention", val: pagedAttn, set: setPagedAttn }, { label: "Prefix caching", val: prefixCache, set: setPrefixCache }].map(({ label, val, set }) => (
          <label key={label} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, cursor: "pointer" }}>
            <input type="checkbox" checked={val} onChange={e => set(e.target.checked)} /> {label}
          </label>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10, marginBottom: 20 }}>
        <MetricCard label="KV Cache Used" value={utilPct} unit="%" color={utilPct > 85 ? "#991b1b" : "#166534"} />
        <MetricCard label="Active Sequences" value={numSeqs} unit="seqs" />
        <MetricCard label="Fragmentation" value={fragPct} unit="%" sub={pagedAttn ? "with PagedAttention" : "without PagedAttention"} />
        <MetricCard label="Pages/Seq" value={pagesPerSeq} unit={`×${PAGE_SIZE}tok`} />
      </div>
      <div style={{ marginBottom: 8, fontSize: 12, color: "var(--color-text-secondary)", fontWeight: 500 }}>{SLOTS} pages shown — page size = {PAGE_SIZE} tokens</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(16, 1fr)", gap: 3, marginBottom: 16 }}>
        {blocks.map((b, i) => (
          <div key={i} title={b.seqIdx >= 0 ? `Seq ${b.seqIdx + 1}${b.isPrefix ? " (shared prefix)" : ""}` : "Free"} style={{
            height: 26, borderRadius: 3, border: "0.5px solid rgba(0,0,0,0.1)", opacity: b.seqIdx >= 0 ? 1 : 0.35, transition: "background .2s",
            background: b.seqIdx >= 0 ? (b.isPrefix ? `repeating-linear-gradient(45deg,${SEQ_COLORS[b.seqIdx % 8]},${SEQ_COLORS[b.seqIdx % 8]} 4px,rgba(255,255,255,.4) 4px,rgba(255,255,255,.4) 8px)` : SEQ_COLORS[b.seqIdx % 8]) : "var(--color-background-secondary)"
          }} />
        ))}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
        {Array.from({ length: numSeqs }, (_, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
            <div style={{ width: 12, height: 12, borderRadius: 2, background: SEQ_COLORS[i % 8] }} />
            <span style={{ color: "var(--color-text-secondary)" }}>Seq {i + 1}</span>
          </div>
        ))}
        {prefixCache && <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
          <div style={{ width: 12, height: 12, borderRadius: 2, background: "repeating-linear-gradient(45deg,#3b82f6,#3b82f6 3px,rgba(255,255,255,.4) 3px,rgba(255,255,255,.4) 6px)" }} />
          <span style={{ color: "var(--color-text-secondary)" }}>Shared prefix</span>
        </div>}
        <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12 }}>
          <div style={{ width: 12, height: 12, borderRadius: 2, background: "var(--color-background-secondary)", border: "0.5px solid var(--color-border-tertiary)" }} />
          <span style={{ color: "var(--color-text-secondary)" }}>Free</span>
        </div>
      </div>
    </div>
  );
}

// ── API Explorer Tab ──
function APITab() {
  const [aModel, setAModel] = useState("llama-3.1-8b");
  const [prompt, setPrompt] = useState("Explain KV cache in 2 sentences.");
  const [temp, setTemp] = useState(0.7);
  const [maxTok, setMaxTok] = useState(256);
  const [stream, setStream] = useState(true);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [codeTab, setCodeTab] = useState("curl");

  const curlCmd = `curl https://api.jarvislabs.ai/v1/chat/completions \\
  -H "Authorization: Bearer $JARVIS_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${aModel}",
    "messages": [{"role": "user", "content": "${prompt}"}],
    "temperature": ${temp.toFixed(1)},
    "max_tokens": ${maxTok},
    "stream": ${stream}
  }'`;

  const pyCode = `from openai import OpenAI

client = OpenAI(
    base_url="https://api.jarvislabs.ai/v1",
    api_key="your_api_key"
)

response = client.chat.completions.create(
    model="${aModel}",
    messages=[{"role": "user", "content": "${prompt}"}],
    temperature=${temp.toFixed(1)},
    max_tokens=${maxTok},
    stream=${stream ? "True" : "False"}
)
${stream ? `for chunk in response:\n    print(chunk.choices[0].delta.content, end="")` : `print(response.choices[0].message.content)`}`;

  function testEndpoint() {
    setLoading(true); setResponse(null);
    setTimeout(() => {
      const text = MOCK_RESPONSES[prompt] || MOCK_RESPONSES.default;
      setResponse({ id: "chatcmpl-" + Math.random().toString(36).slice(2, 10), text, tokens: Math.round(text.split(" ").length * 1.3), latency: Math.round(80 + Math.random() * 40), throughput: Math.round(150 + Math.random() * 80) });
      setLoading(false);
    }, 900 + Math.random() * 600);
  }

  const btnStyle = { padding: "8px 20px", fontSize: 13, fontWeight: 500, borderRadius: 8, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-primary)", color: "var(--color-text-primary)", cursor: "pointer" };

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div>
          <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 6 }}>Model</label>
          <select value={aModel} onChange={e => setAModel(e.target.value)}>{MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select>
        </div>
        <div>
          <label style={{ fontSize: 12, color: "var(--color-text-secondary)", display: "block", marginBottom: 6 }}>Prompt</label>
          <input type="text" value={prompt} onChange={e => setPrompt(e.target.value)} />
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        {[{ label: "Temperature", val: temp, set: setTemp, min: 0, max: 2, step: 0.1, fmt: v => v.toFixed(1) }, { label: "Max tokens", val: maxTok, set: setMaxTok, min: 32, max: 2048, step: 32, fmt: v => v }].map(({ label, val, set, min, max, step, fmt }) => (
          <div key={label}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
              <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>{label}</span>
              <span style={{ fontSize: 12 }}>{fmt(val)}</span>
            </div>
            <input type="range" min={min} max={max} step={step} value={val} onChange={e => set(Number(e.target.value))} />
          </div>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20 }}>
        <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, cursor: "pointer" }}>
          <input type="checkbox" checked={stream} onChange={e => setStream(e.target.checked)} /> Streaming (SSE)
        </label>
        <button onClick={testEndpoint} disabled={loading} style={{ ...btnStyle, opacity: loading ? 0.6 : 1 }}>{loading ? "Calling..." : "Test Endpoint"}</button>
      </div>
      <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
        {["curl", "python"].map(t => (
          <button key={t} onClick={() => setCodeTab(t)} style={{ padding: "6px 14px", fontSize: 12, cursor: "pointer", background: "none", border: "none", borderBottom: codeTab === t ? "2px solid var(--color-text-primary)" : "2px solid transparent", color: codeTab === t ? "var(--color-text-primary)" : "var(--color-text-secondary)", fontFamily: "var(--font-mono)", marginBottom: -1 }}>{t}</button>
        ))}
      </div>
      <pre style={{ background: "#0f1117", borderRadius: "0 0 8px 8px", padding: "14px 16px", fontFamily: "var(--font-mono)", fontSize: 11, color: "#93c5fd", overflowX: "auto", margin: 0, lineHeight: 1.7, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{codeTab === "curl" ? curlCmd : pyCode}</pre>
      {response && (
        <div style={{ marginTop: 20 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10, marginBottom: 16 }}>
            <MetricCard label="Latency" value={response.latency} unit="ms" />
            <MetricCard label="Output tokens" value={response.tokens} unit="tok" />
            <MetricCard label="Throughput" value={response.throughput} unit="tok/s" color="#166534" />
          </div>
          <div style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: "14px 16px" }}>
            <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginBottom: 8, fontFamily: "var(--font-mono)" }}>id: {response.id}</div>
            <div style={{ fontSize: 13, lineHeight: 1.7 }}>{response.text}</div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main App ──
export default function App() {
  const [tab, setTab] = useState(0);
  const tabContent = [<DeployTab />, <BenchmarkTab />, <KVCacheTab />, <APITab />];

  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "32px 24px", fontFamily: "var(--font-sans)", color: "var(--color-text-primary)" }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 4 }}>
          <div>
            <span style={{ fontSize: 20, fontWeight: 600 }}>Jarvis Labs</span>
            <span style={{ fontSize: 13, color: "var(--color-text-secondary)", marginLeft: 10 }}>Inference-as-a-Service · E2E Networks</span>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <Badge color="success">3 endpoints</Badge>
            <Badge color="info">OpenAI-compatible</Badge>
          </div>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 8, marginTop: 12 }}>
          {DEPLOYMENTS.map(d => (
            <div key={d.model} style={{ background: "var(--color-background-secondary)", borderRadius: 8, padding: "10px 12px" }}>
              <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 2 }}><StatusDot status={d.status} />{d.model}</div>
              <div style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{d.runtime} · {d.gpu}</div>
              {d.status === "running" && <div style={{ fontSize: 11, color: "#166534", marginTop: 3 }}>{d.rps} req/s</div>}
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", marginBottom: 24 }}>
        <div style={{ display: "flex" }}>
          {TABS.map((t, i) => (
            <button key={t} onClick={() => setTab(i)} style={{ padding: "8px 16px", fontSize: 13, fontWeight: tab === i ? 500 : 400, color: tab === i ? "var(--color-text-primary)" : "var(--color-text-secondary)", background: "none", border: "none", borderBottom: tab === i ? "2px solid var(--color-text-primary)" : "2px solid transparent", cursor: "pointer", marginBottom: -1 }}>{t}</button>
          ))}
        </div>
      </div>

      {tabContent[tab]}
    </div>
  );
}
