import { useState } from "react";

const modelData = [
  { name: "Random Forest", mae: 4.82, rmse: 8.13, r2: 0.7452, mape: 167.3, best: true },
  { name: "Ridge Regression", mae: 4.87, rmse: 7.96, r2: 0.7557, mape: 164.9, best: false },
  { name: "Linear Regression", mae: 4.87, rmse: 7.96, r2: 0.7554, mape: 163.4, best: false },
  { name: "Gradient Boosting", mae: 5.56, rmse: 9.97, r2: 0.6169, mape: 165.6, best: false },
];

const bessStrategies = [
  { name: "Perfect Foresight", testRev: 241578, annualRev: 958435, capture: 100.0, color: "#10b981" },
  { name: "ML Forecast (RF)", testRev: 207094, annualRev: 821625, capture: 85.7, color: "#f59e0b" },
  { name: "Threshold (30/60)", testRev: 19409, annualRev: 77003, capture: 8.0, color: "#6366f1" },
];

const featureImportance = [
  { feature: "lmp_lag_1h", importance: 0.50 },
  { feature: "load_wind_ratio", importance: 0.15 },
  { feature: "lmp_lag_24h", importance: 0.07 },
  { feature: "lmp_lag_2h", importance: 0.06 },
  { feature: "same_hour_yest", importance: 0.04 },
  { feature: "net_load", importance: 0.03 },
  { feature: "gas_price", importance: 0.03 },
  { feature: "lmp_lag_168h", importance: 0.02 },
  { feature: "renewable_pen", importance: 0.01 },
  { feature: "wind_gen", importance: 0.01 },
];

const thresholdSensitivity = [
  { thresh: "15/45", rev: 96501 },
  { thresh: "20/50", rev: 79760 },
  { thresh: "25/55", rev: 77737 },
  { thresh: "30/60", rev: 77003 },
  { thresh: "35/65", rev: 77003 },
  { thresh: "40/70", rev: 77003 },
  { thresh: "25/80", rev: 77737 },
];

const bessConfig = {
  capacity: "100 MWh",
  power: "25 MW",
  duration: "4-hour",
  efficiency: "87% RT",
  minSOC: "10%",
  maxSOC: "90%",
  degradation: "$2/MWh",
};

const marketStats = {
  records: "17,544",
  period: "Jan 2024 – Dec 2025",
  meanLMP: "$23.79",
  medianLMP: "$21.96",
  minLMP: "-$20.00",
  maxLMP: "$2,606.85",
  p95: "$65.98",
};

function KPI({ label, value, sub, accent }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.04)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 12,
      padding: "20px 24px",
      flex: "1 1 200px",
      minWidth: 180,
    }}>
      <div style={{ fontSize: 12, color: "#94a3b8", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, color: accent || "#f8fafc", fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function BarChart({ data, valueKey, labelKey, maxVal, color, format }) {
  const max = maxVal || Math.max(...data.map(d => d[valueKey]));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {data.map((d, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 120, fontSize: 12, color: "#94a3b8", textAlign: "right", flexShrink: 0 }}>
            {d[labelKey]}
          </div>
          <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", borderRadius: 6, height: 28, position: "relative", overflow: "hidden" }}>
            <div style={{
              width: `${(d[valueKey] / max) * 100}%`,
              height: "100%",
              background: typeof color === "function" ? color(d, i) : (color || "#6366f1"),
              borderRadius: 6,
              transition: "width 0.6s ease",
            }} />
            <span style={{
              position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)",
              fontSize: 12, fontWeight: 600, color: "#e2e8f0",
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {format ? format(d[valueKey]) : d[valueKey].toFixed(2)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function Section({ title, children, id }) {
  return (
    <div id={id} style={{ marginBottom: 48 }}>
      <h2 style={{
        fontSize: 20, fontWeight: 700, color: "#f8fafc", marginBottom: 20,
        paddingBottom: 10, borderBottom: "1px solid rgba(255,255,255,0.08)",
        display: "flex", alignItems: "center", gap: 10,
      }}>
        <span style={{ width: 4, height: 20, background: "#6366f1", borderRadius: 2 }} />
        {title}
      </h2>
      {children}
    </div>
  );
}

function Card({ children, style }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      border: "1px solid rgba(255,255,255,0.06)",
      borderRadius: 16,
      padding: 24,
      ...style,
    }}>
      {children}
    </div>
  );
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("overview");

  const tabs = [
    { id: "overview", label: "Overview", icon: "⚡" },
    { id: "forecasting", label: "Forecasting", icon: "📈" },
    { id: "bess", label: "BESS Arbitrage", icon: "🔋" },
    { id: "sensitivity", label: "Sensitivity", icon: "⚙" },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0f1a",
      color: "#e2e8f0",
      fontFamily: "'Inter', -apple-system, sans-serif",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <div style={{
        background: "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
        padding: "32px 40px 24px",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
          <span style={{ fontSize: 28 }}>⚡</span>
          <h1 style={{ fontSize: 26, fontWeight: 700, color: "#f8fafc", margin: 0, letterSpacing: -0.5 }}>
            ERCOT Electricity Price Forecasting
          </h1>
        </div>
        <p style={{ fontSize: 14, color: "#64748b", margin: "6px 0 20px 40px" }}>
          Battery Energy Storage System Arbitrage Optimization &bull; Day-Ahead LMP Analysis
        </p>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginLeft: 40 }}>
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              style={{
                padding: "8px 18px",
                border: "none",
                borderRadius: "8px 8px 0 0",
                cursor: "pointer",
                fontSize: 13,
                fontWeight: 600,
                transition: "all 0.2s",
                background: activeTab === t.id ? "rgba(99,102,241,0.2)" : "transparent",
                color: activeTab === t.id ? "#a5b4fc" : "#64748b",
                borderBottom: activeTab === t.id ? "2px solid #6366f1" : "2px solid transparent",
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 40px 60px" }}>

        {/* OVERVIEW TAB */}
        {activeTab === "overview" && (
          <>
            <Section title="Market Data Overview">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginBottom: 24 }}>
                <KPI label="Hourly Records" value={marketStats.records} sub={marketStats.period} />
                <KPI label="Mean LMP" value={marketStats.meanLMP} sub="/MWh" accent="#f59e0b" />
                <KPI label="Median LMP" value={marketStats.medianLMP} sub="/MWh" accent="#10b981" />
                <KPI label="95th Percentile" value={marketStats.p95} sub="/MWh" accent="#ef4444" />
              </div>
              <Card>
                <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.7 }}>
                  <strong style={{ color: "#f8fafc" }}>Dataset:</strong> 2 years of synthetic ERCOT-style day-ahead LMP data (Jan 2024 – Dec 2025) modeling real market characteristics including seasonal patterns, hourly profiles, weekend effects, heat wave scarcity events, wind/solar generation impacts, gas price correlations, and ERCOT-specific price spikes up to ${marketStats.maxLMP}/MWh.
                </div>
              </Card>
            </Section>

            <Section title="Key Results at a Glance">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                <Card>
                  <div style={{ fontSize: 13, color: "#6366f1", fontWeight: 600, marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 }}>Best Forecasting Model</div>
                  <div style={{ fontSize: 32, fontWeight: 700, color: "#f8fafc", fontFamily: "'JetBrains Mono', monospace" }}>Random Forest</div>
                  <div style={{ display: "flex", gap: 20, marginTop: 16 }}>
                    <div><span style={{ fontSize: 11, color: "#64748b" }}>MAE</span><br/><span style={{ fontSize: 18, fontWeight: 600, color: "#10b981", fontFamily: "'JetBrains Mono', monospace" }}>$4.82</span></div>
                    <div><span style={{ fontSize: 11, color: "#64748b" }}>RMSE</span><br/><span style={{ fontSize: 18, fontWeight: 600, color: "#f59e0b", fontFamily: "'JetBrains Mono', monospace" }}>$8.13</span></div>
                    <div><span style={{ fontSize: 11, color: "#64748b" }}>R²</span><br/><span style={{ fontSize: 18, fontWeight: 600, color: "#a5b4fc", fontFamily: "'JetBrains Mono', monospace" }}>0.745</span></div>
                  </div>
                </Card>
                <Card>
                  <div style={{ fontSize: 13, color: "#f59e0b", fontWeight: 600, marginBottom: 12, textTransform: "uppercase", letterSpacing: 1 }}>ML-Driven BESS Revenue</div>
                  <div style={{ fontSize: 32, fontWeight: 700, color: "#f8fafc", fontFamily: "'JetBrains Mono', monospace" }}>$821,625<span style={{ fontSize: 14, color: "#64748b" }}>/yr</span></div>
                  <div style={{ display: "flex", gap: 20, marginTop: 16 }}>
                    <div><span style={{ fontSize: 11, color: "#64748b" }}>vs Perfect</span><br/><span style={{ fontSize: 18, fontWeight: 600, color: "#10b981", fontFamily: "'JetBrains Mono', monospace" }}>85.7%</span></div>
                    <div><span style={{ fontSize: 11, color: "#64748b" }}>$/kW-yr</span><br/><span style={{ fontSize: 18, fontWeight: 600, color: "#f59e0b", fontFamily: "'JetBrains Mono', monospace" }}>$33</span></div>
                    <div><span style={{ fontSize: 11, color: "#64748b" }}>BESS Size</span><br/><span style={{ fontSize: 18, fontWeight: 600, color: "#a5b4fc", fontFamily: "'JetBrains Mono', monospace" }}>100MWh</span></div>
                  </div>
                </Card>
              </div>
            </Section>

            <Section title="Project Pipeline">
              <div style={{ display: "flex", gap: 2 }}>
                {[
                  { step: "1", title: "Data Generation", desc: "17,544 hourly ERCOT-style LMP records with load, wind, solar, gas" },
                  { step: "2", title: "Cleaning & EDA", desc: "Missing value imputation, outlier capping, seasonal/hourly analysis" },
                  { step: "3", title: "Feature Engineering", desc: "30 features: lags, rolling stats, cyclical encoding, fundamentals" },
                  { step: "4", title: "ML Forecasting", desc: "4 models benchmarked with time-series train/test split" },
                  { step: "5", title: "BESS Simulation", desc: "3 arbitrage strategies with SOC constraints & degradation costs" },
                ].map((s, i) => (
                  <div key={i} style={{
                    flex: 1, padding: "16px 12px", textAlign: "center",
                    background: `rgba(99,102,241,${0.05 + i * 0.04})`,
                    borderRadius: i === 0 ? "12px 0 0 12px" : i === 4 ? "0 12px 12px 0" : 0,
                  }}>
                    <div style={{
                      width: 28, height: 28, borderRadius: "50%", background: "#6366f1",
                      color: "#fff", display: "flex", alignItems: "center", justifyContent: "center",
                      margin: "0 auto 8px", fontSize: 13, fontWeight: 700,
                    }}>{s.step}</div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: "#e2e8f0", marginBottom: 4 }}>{s.title}</div>
                    <div style={{ fontSize: 10, color: "#64748b", lineHeight: 1.4 }}>{s.desc}</div>
                  </div>
                ))}
              </div>
            </Section>
          </>
        )}

        {/* FORECASTING TAB */}
        {activeTab === "forecasting" && (
          <>
            <Section title="Model Performance Comparison">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
                {modelData.map((m, i) => (
                  <Card key={i} style={m.best ? { border: "1px solid rgba(99,102,241,0.3)", background: "rgba(99,102,241,0.06)" } : {}}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
                      <span style={{ fontSize: 15, fontWeight: 600, color: "#f8fafc" }}>{m.name}</span>
                      {m.best && <span style={{ fontSize: 10, padding: "3px 10px", background: "#6366f1", borderRadius: 20, color: "#fff", fontWeight: 600 }}>BEST</span>}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 8 }}>
                      {[
                        { l: "MAE", v: `$${m.mae}`, c: "#10b981" },
                        { l: "RMSE", v: `$${m.rmse}`, c: "#f59e0b" },
                        { l: "R²", v: m.r2.toFixed(4), c: "#a5b4fc" },
                        { l: "MAPE", v: `${m.mape}%`, c: "#f87171" },
                      ].map((metric, j) => (
                        <div key={j} style={{ textAlign: "center" }}>
                          <div style={{ fontSize: 10, color: "#64748b" }}>{metric.l}</div>
                          <div style={{ fontSize: 15, fontWeight: 600, color: metric.c, fontFamily: "'JetBrains Mono', monospace" }}>{metric.v}</div>
                        </div>
                      ))}
                    </div>
                  </Card>
                ))}
              </div>
            </Section>

            <Section title="MAE Comparison">
              <Card>
                <BarChart
                  data={modelData.sort((a, b) => a.mae - b.mae)}
                  valueKey="mae"
                  labelKey="name"
                  color={(d) => d.best ? "#10b981" : "#6366f1"}
                  format={(v) => `$${v.toFixed(2)}/MWh`}
                />
              </Card>
            </Section>

            <Section title="Feature Importance — Gradient Boosting">
              <Card>
                <BarChart
                  data={featureImportance}
                  valueKey="importance"
                  labelKey="feature"
                  color="#8b5cf6"
                  format={(v) => (v * 100).toFixed(1) + "%"}
                />
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 16, lineHeight: 1.6 }}>
                  The 1-hour lag dominates (~50% importance), followed by load-wind ratio and 24h lag — consistent with electricity market autocorrelation and the fundamental supply-demand balance driving ERCOT prices.
                </div>
              </Card>
            </Section>

            <Section title="Training Configuration">
              <Card>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, fontSize: 13 }}>
                  <div>
                    <div style={{ color: "#6366f1", fontWeight: 600, marginBottom: 8 }}>Train Set</div>
                    <div style={{ color: "#94a3b8" }}>15,168 samples (Jan 2024 – Sep 2025)</div>
                    <div style={{ color: "#94a3b8", marginTop: 4 }}>Time-based split — no data leakage</div>
                  </div>
                  <div>
                    <div style={{ color: "#f59e0b", fontWeight: 600, marginBottom: 8 }}>Test Set</div>
                    <div style={{ color: "#94a3b8" }}>2,208 samples (Oct – Dec 2025)</div>
                    <div style={{ color: "#94a3b8", marginTop: 4 }}>3-month out-of-sample evaluation</div>
                  </div>
                  <div>
                    <div style={{ color: "#10b981", fontWeight: 600, marginBottom: 8 }}>Features (30 total)</div>
                    <div style={{ color: "#94a3b8" }}>Temporal cyclical encoding, lag features (1h–168h), rolling statistics (24h–168h), fundamental drivers (load, wind, solar, gas), interaction terms (net load, renewable penetration)</div>
                  </div>
                  <div>
                    <div style={{ color: "#ef4444", fontWeight: 600, marginBottom: 8 }}>Preprocessing</div>
                    <div style={{ color: "#94a3b8" }}>Forward/back-fill for missing data, 99.5th percentile cap ($91.44) for extreme spikes, StandardScaler for linear models</div>
                  </div>
                </div>
              </Card>
            </Section>
          </>
        )}

        {/* BESS TAB */}
        {activeTab === "bess" && (
          <>
            <Section title="BESS Configuration">
              <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>
                {Object.entries(bessConfig).map(([k, v]) => (
                  <div key={k} style={{
                    padding: "10px 16px", background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10,
                  }}>
                    <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase" }}>{k.replace(/([A-Z])/g, ' $1')}</div>
                    <div style={{ fontSize: 16, fontWeight: 600, color: "#f8fafc", fontFamily: "'JetBrains Mono', monospace" }}>{v}</div>
                  </div>
                ))}
              </div>
            </Section>

            <Section title="Arbitrage Strategy Comparison">
              {bessStrategies.map((s, i) => (
                <Card key={i} style={{ marginBottom: 12 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <div style={{ width: 12, height: 12, borderRadius: "50%", background: s.color }} />
                      <span style={{ fontSize: 15, fontWeight: 600, color: "#f8fafc" }}>{s.name}</span>
                    </div>
                    <div style={{ display: "flex", gap: 32 }}>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 10, color: "#64748b" }}>Test Period (3mo)</div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: s.color, fontFamily: "'JetBrains Mono', monospace" }}>${s.testRev.toLocaleString()}</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 10, color: "#64748b" }}>Annualized</div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: "#f8fafc", fontFamily: "'JetBrains Mono', monospace" }}>${s.annualRev.toLocaleString()}</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 10, color: "#64748b" }}>Capture Rate</div>
                        <div style={{ fontSize: 18, fontWeight: 700, color: s.capture > 80 ? "#10b981" : s.capture > 50 ? "#f59e0b" : "#ef4444", fontFamily: "'JetBrains Mono', monospace" }}>{s.capture}%</div>
                      </div>
                    </div>
                  </div>
                  {/* Revenue bar */}
                  <div style={{ marginTop: 12, height: 8, background: "rgba(255,255,255,0.05)", borderRadius: 4, overflow: "hidden" }}>
                    <div style={{ width: `${s.capture}%`, height: "100%", background: s.color, borderRadius: 4, transition: "width 0.8s ease" }} />
                  </div>
                </Card>
              ))}
            </Section>

            <Section title="Strategy Insights">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                <Card>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#10b981", marginBottom: 8 }}>Perfect Foresight</div>
                  <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
                    Upper bound benchmark. Charges in cheapest 6 hours and discharges in most expensive 6 hours each day. Achieves $958K/yr — the theoretical maximum for a 100 MWh / 25 MW system.
                  </div>
                </Card>
                <Card>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#f59e0b", marginBottom: 8 }}>ML Forecast</div>
                  <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
                    Uses Random Forest price predictions to schedule charge/discharge. Captures 85.7% of perfect foresight revenue — demonstrating that even modest forecasting accuracy translates to significant real-world arbitrage value.
                  </div>
                </Card>
                <Card>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#6366f1", marginBottom: 8 }}>Threshold Strategy</div>
                  <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
                    Simple rule: charge below $30, discharge above $60. Only captures 8% of potential — showing that naive strategies massively underperform ML-driven approaches in volatile markets.
                  </div>
                </Card>
              </div>
            </Section>
          </>
        )}

        {/* SENSITIVITY TAB */}
        {activeTab === "sensitivity" && (
          <>
            <Section title="Threshold Sensitivity Analysis">
              <Card>
                <BarChart
                  data={thresholdSensitivity}
                  valueKey="rev"
                  labelKey="thresh"
                  color={(d, i) => `hsl(${20 + i * 20}, 70%, 55%)`}
                  format={(v) => `$${(v / 1000).toFixed(0)}K/yr`}
                />
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 16, lineHeight: 1.6 }}>
                  The 15/45 threshold outperforms others by capturing more charging opportunities during low-price overnight hours. However, all threshold strategies significantly underperform the ML forecast approach ($822K/yr), reinforcing the value of predictive analytics.
                </div>
              </Card>
            </Section>

            <Section title="Key Risk Factors & Assumptions">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                {[
                  { title: "Price Volatility", desc: "Higher volatility = higher BESS revenue. The ERCOT market's lack of capacity market and extreme summer peaks create favorable conditions for storage arbitrage.", icon: "📊" },
                  { title: "Renewable Penetration", desc: "Growing wind/solar in ERCOT increases price volatility (duck curve effects, negative pricing), generally benefiting BESS economics.", icon: "🌱" },
                  { title: "Degradation Costs", desc: "Modeled at $2/MWh. Real-world degradation depends on chemistry (LFP vs NMC), depth of discharge, temperature, and cycling patterns.", icon: "🔧" },
                  { title: "Market Access", desc: "Assumes perfect market access with no transmission constraints, real-time settlement differences, or ancillary service stacking (which could further boost revenue).", icon: "🔌" },
                ].map((item, i) => (
                  <Card key={i}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                      <span style={{ fontSize: 18 }}>{item.icon}</span>
                      <span style={{ fontSize: 13, fontWeight: 600, color: "#f8fafc" }}>{item.title}</span>
                    </div>
                    <div style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>{item.desc}</div>
                  </Card>
                ))}
              </div>
            </Section>

            <Section title="Potential Enhancements">
              <Card>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, fontSize: 12, color: "#94a3b8", lineHeight: 1.6 }}>
                  <div>
                    <div style={{ color: "#6366f1", fontWeight: 600, marginBottom: 6 }}>Forecasting Improvements</div>
                    <div>XGBoost/LightGBM with hyperparameter tuning, LSTM/Transformer neural networks for sequence modeling, ensemble methods combining multiple model predictions, real-time weather data integration</div>
                  </div>
                  <div>
                    <div style={{ color: "#f59e0b", fontWeight: 600, marginBottom: 6 }}>Revenue Stacking</div>
                    <div>Ancillary services (regulation up/down, responsive reserve), real-time vs day-ahead spread trading, capacity market participation, congestion revenue rights</div>
                  </div>
                  <div>
                    <div style={{ color: "#10b981", fontWeight: 600, marginBottom: 6 }}>Optimization</div>
                    <div>Mixed-integer programming for optimal dispatch, stochastic optimization under forecast uncertainty, rolling horizon model predictive control, co-optimization with renewable assets</div>
                  </div>
                  <div>
                    <div style={{ color: "#ef4444", fontWeight: 600, marginBottom: 6 }}>Data Sources</div>
                    <div>Real ERCOT SCED/DAM data via ERCOT MIS, NOAA weather actuals and forecasts, EIA generation mix data, EPA emissions for carbon-optimized dispatch</div>
                  </div>
                </div>
              </Card>
            </Section>
          </>
        )}

        {/* Footer */}
        <div style={{
          marginTop: 48, padding: "20px 0", borderTop: "1px solid rgba(255,255,255,0.06)",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <div style={{ fontSize: 11, color: "#475569" }}>
            ERCOT Electricity Price Forecasting & BESS Optimization &bull; Portfolio Project
          </div>
          <div style={{ fontSize: 11, color: "#475569" }}>
            Python &bull; Pandas &bull; scikit-learn &bull; Matplotlib
          </div>
        </div>
      </div>
    </div>
  );
}
