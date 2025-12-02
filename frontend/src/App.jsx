import React, { useState, useEffect } from "react";
import Papa from "papaparse";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

export default function App() {
  const [rows, setRows] = useState([]);
  const [headers, setHeaders] = useState([]);
  const [mapping, setMapping] = useState({});
  const [previewRows, setPreviewRows] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [inference, setInference] = useState(null);
  const [loading, setLoading] = useState(false);
  const [feature, setFeature] = useState("");

  useEffect(() => {
    setPreviewRows(rows.slice(0, 10));
    if (headers.length && !feature) setFeature(headers[0]);
  }, [rows, headers]);

  function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      complete: (res) => {
        const data = res.data.map((r) => {
          const converted = {};
          Object.keys(r).forEach(k => {
            const v = r[k];
            const n = Number(v);
            converted[k] = Number.isFinite(n) ? n : v;
          });
          return converted;
        });
        setRows(data);
        setHeaders(Object.keys(data[0] || {}));
        // naive mapping
        const defaultMap = {};
        const keyMap = {
          CO2_ppm: ["CO2_ppm", "co2", "co2_ppm"],
          H2S_ppm: ["H2S_ppm", "h2s"],
          pH: ["pH", "ph"],
          temperature_C: ["temperature_C", "temp"],
          flow_m_s: ["flow_m_s", "flow"],
          inhibitor_eff: ["inhibitor_eff", "inhibitor"],
          CP_voltage: ["CP_voltage", "cp_voltage", "cp"],
          P_corrosion: ["P_corrosion","p_corrosion"],
          Corrosion: ["Corrosion","corrosion","label"]
        };
        Object.keys(keyMap).forEach(k => {
          for (const cand of keyMap[k]) {
            const found = Object.keys(data[0]||{}).find(h => h.toLowerCase() === cand.toLowerCase());
            if (found) { defaultMap[k] = found; break; }
          }
        });
        setMapping(defaultMap);
      }
    });
  }

  function histogramData(col) {
    const vals = rows.map(r => r[col]).filter(v => typeof v === "number" && !isNaN(v));
    if (!vals.length) return [];
    const nb = 25;
    const min = Math.min(...vals), max = Math.max(...vals);
    const w = (max - min) / nb || 1;
    const buckets = new Array(nb).fill(0);
    vals.forEach(v => {
      let i = Math.floor((v - min) / w);
      if (i < 0) i = 0;
      if (i >= nb) i = nb - 1;
      buckets[i] += 1;
    });
    return buckets.map((c,i)=>({ name: (min + i*w).toFixed(1), count: c }));
  }

  function buildPayloadFromRow(row) {
  // safe getter: mapping may or may not contain the key
  const get = (k) => {
    if (!row) return null;
    const mapped = mapping?.[k];
    if (mapped && Object.prototype.hasOwnProperty.call(row, mapped)) return row[mapped];
    if (Object.prototype.hasOwnProperty.call(row, k)) return row[k];
    return null;
  };

  return {
    CO2_ppm: get("CO2_ppm"),
    H2S_ppm: get("H2S_ppm"),
    pH: get("pH"),
    temperature_C: get("temperature_C"),
    flow_m_s: get("flow_m_s"),
    inhibitor_eff: get("inhibitor_eff"),
    CP_voltage: get("CP_voltage"),
  };
}


  // The above accidental characters break JS — fix: remove the stray line.
  // (This comment is only for the assistant — the code below is corrected.)

  function buildPayloadCorrect(row) {
    return {
      CO2_ppm: row[mapping.CO2_ppm] ?? row.CO2_ppm ?? null,
      H2S_ppm: row[mapping.H2S_ppm] ?? row.H2S_ppm ?? null,
      pH: row[mapping.pH] ?? row.pH ?? null,
      temperature_C: row[mapping.temperature_C] ?? row.temperature_C ?? null,
      flow_m_s: row[mapping.flow_m_s] ?? row.flow_m_s ?? null,
      inhibitor_eff: row[mapping.inhibitor_eff] ?? row.inhibitor_eff ?? null,
      CP_voltage: row[mapping.CP_voltage] ?? row.CP_voltage ?? null,
    };
  }

  async function runInference() {
    setLoading(true);
    setInference(null);
    try {
      if (!rows.length) throw new Error("No data uploaded");
      const row = rows[selectedIndex] ?? rows[0];
      const payload = buildPayloadCorrect(row);
      const resp = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (resp.ok) {
        const data = await resp.json();
        setInference({ from: "backend", ...data });
      } else {
        // fallback to local heuristic
        const data = await resp.text();
        console.warn("Backend returned error:", data);
        const p = localHeuristic(payload);
        setInference({ from: "fallback", P_corrosion: p });
      }
    } catch (err) {
      console.error(err);
      const p = localHeuristic(buildPayloadCorrect(rows[selectedIndex] ?? rows[0] ?? {}));
      setInference({ from: "fallback", P_corrosion: p, error: err.message });
    } finally {
      setLoading(false);
    }
  }

  function localHeuristic(payload) {
    const CO2_ppm = Number(payload.CO2_ppm || 0);
    const H2S_ppm = Number(payload.H2S_ppm || 0);
    const pH = Number(payload.pH || 7);
    const temp = Number(payload.temperature_C || 50);
    const flow = Number(payload.flow_m_s || 1);
    const inhibitor = Number(payload.inhibitor_eff || 1);
    const cpv = Number(payload.CP_voltage || -0.8);

    let risk = 0;
    risk += 0.003 * CO2_ppm;
    risk += 0.05 * (H2S_ppm > 5 ? 1 : 0);
    risk += 0.5 * (pH < 6.0 ? 1 : 0);
    risk += 0.02 * Math.max(0, temp - 50);
    risk += 0.8 * (flow < 0.3 ? 1 : 0);
    risk += 1.2 * (inhibitor < 0.3 ? 1 : 0);
    risk += 0.7 * (cpv > -0.6 ? 1 : 0);
    const p = 1 / (1 + Math.exp(-(risk - 1.5)));
    return Number(p.toFixed(4));
  }

  return (
    <div className="container">
      <div className="header">
        <div>
          <h1>Corrosion BN Dashboard</h1>
          <div className="small">Upload CSV → map columns → preview → run inference</div>
        </div>
      </div>

      <div style={{display: 'grid', gridTemplateColumns: '360px 1fr', gap: 16, marginTop: 16}}>
        <div className="card">
          <div style={{display:'flex', gap:8, flexDirection:'column'}} className="controls">
            <div>
              <label className="small">Upload CSV</label>
              <input type="file" accept=".csv" onChange={handleFile} />
            </div>

            <div>
              <label className="small">Row to test</label>
              <input type="number" value={selectedIndex} min={0} max={Math.max(0, rows.length-1)} onChange={(e)=>setSelectedIndex(Number(e.target.value))} />
            </div>

            <div>
              <label className="small">Column mappings (auto-suggest)</label>
              {["CO2_ppm","H2S_ppm","pH","temperature_C","flow_m_s","inhibitor_eff","CP_voltage"].map(k => (
                <div key={k} style={{marginTop:8}}>
                  <div className="small">{k}</div>
                  <select value={mapping[k] ?? ""} onChange={(e) => setMapping(m => ({...m, [k]: e.target.value}))}>
                    <option value="">— select —</option>
                    {headers.map(h => <option key={h} value={h}>{h}</option>)}
                  </select>
                </div>
              ))}
            </div>

            <div>
              <button className="primary" onClick={runInference} disabled={loading}>{loading ? "Running..." : "Run Inference"}</button>
            </div>

            <div className="small" style={{marginTop:8}}>
              Tip: The frontend proxies /api to the backend (see vite config). If backend not running, the app runs a local heuristic.
            </div>
          </div>
        </div>

        <div style={{display:'grid', gap:12}}>
          <div className="card">
            <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
              <div>
                <h3>Dataset preview</h3>
                <div className="small">{rows.length} rows</div>
              </div>
            </div>

            <div className="table-preview" style={{marginTop:12}}>
              <table style={{width:"100%", borderCollapse:"collapse"}}>
                <thead>
                  <tr>
                    <th style={{textAlign:"left", padding:6}}>#</th>
                    {headers.slice(0,8).map(h => <th key={h} style={{textAlign:"left", padding:6}}>{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {previewRows.map((r,i) => (
                    <tr key={i} style={{borderTop:"1px solid #eef2f7"}}>
                      <td style={{padding:6}}>{i}</td>
                      {headers.slice(0,8).map(h => <td key={h} style={{padding:6}}>{String(r[h])}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={{display:'grid', gridTemplateColumns:'1fr 320px', gap:12}}>
            <div className="card">
              <h4>Feature distribution</h4>
              <div style={{marginBottom:8}}>
                <select value={feature} onChange={(e)=>setFeature(e.target.value)}>
                  {headers.map(h => <option key={h} value={h}>{h}</option>)}
                </select>
              </div>
              <div className="hist">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={feature ? histogramData(feature) : []}>
                    <XAxis dataKey="name" hide />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <h4>Inference result</h4>
              {inference ? (
                <div>
                  <div className="result-badge" style={{marginBottom:12}}>
                    <div style={{fontSize:34, fontWeight:700}}>{Math.round((inference.P_corrosion ?? 0) * 100)}%</div>
                    <div className="small">P(corrosion)</div>
                    <div className="small" style={{marginTop:8}}>source: {inference.from}</div>
                  </div>

                  <div>
                    <div className="small">Recommended action</div>
                    <ul style={{marginTop:8}}>
                      { (inference.P_corrosion ?? 0) >= 0.7 ? (
                        <>
                          <li>High risk — schedule ILI / dig verification immediately.</li>
                          <li>Consider temporary shut / chemical dosing where possible.</li>
                        </>
                      ) : (inference.P_corrosion ?? 0) >= 0.3 ? (
                        <li>Moderate risk — increase monitoring frequency and review CP / inhibitor dosing.</li>
                      ) : (
                        <li>Low risk — normal monitoring.</li>
                      )}
                    </ul>
                  </div>
                </div>
              ) : <div className="small">No inference yet.</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
