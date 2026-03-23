import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { searchAssets } from "../api/api";
import Panel from "../components/Panel";
import { usePlatform } from "../context/PlatformContext";

const featureNarratives = {
  RSI_14: { label: "RSI", bullish: "Demand is still supporting the move.", bearish: "The asset looks stretched and vulnerable to pullbacks." },
  MACD: { label: "MACD", bullish: "Trend momentum is still accelerating.", bearish: "Trend momentum is flattening." },
  MACD_hist: { label: "MACD histogram", bullish: "The bullish spread is widening.", bearish: "Momentum spread is narrowing." },
  momentum_20: { label: "20-day momentum", bullish: "The medium-term trend is positive.", bearish: "Recent price action has started to fade." },
  volatility_20: { label: "20-day volatility", bullish: "Volatility is contained enough for upside continuation.", bearish: "Volatility is elevated, so risk is harder to control." },
  volume_zscore: { label: "Volume surge", bullish: "Participation is expanding behind the move.", bearish: "Participation is unstable or fading." },
  price_vs_sma20: { label: "Price vs SMA20", bullish: "Price is holding above the short-term trend anchor.", bearish: "Price is slipping under its short-term trend anchor." },
  price_vs_sma50: { label: "Price vs SMA50", bullish: "Longer trend structure still supports the move.", bearish: "Longer trend structure is weakening." },
};

const signalTone = {
  BUY: "text-pulse",
  HOLD: "text-frost",
  AVOID: "text-ember",
};

const regimeCopy = {
  bull: {
    title: "Bullish market state",
    detail: "Trend conditions are supportive. It is easier to reward strength than fade it.",
    action: "Lean into winners, but size positions based on your risk level.",
  },
  volatile: {
    title: "Volatile market state",
    detail: "Price swings are larger and confidence degrades faster than usual.",
    action: "Reduce position size and expect wider drawdowns.",
  },
  crisis: {
    title: "Crisis market state",
    detail: "The system is seeing unstable conditions with poor trend reliability.",
    action: "Protect capital first and wait for better asymmetry.",
  },
  normal: {
    title: "Normal market state",
    detail: "The market is neither aggressively trending nor breaking down.",
    action: "Use balanced sizing and rely on signal quality instead of aggression.",
  },
};

function formatCurrency(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value || 0);
}

export default function Dashboard() {
  const { dashboardData, selectedAsset, loadAssetAnalysis, session } = usePlatform();
  const [query, setQuery] = useState(selectedAsset);
  const [suggestions, setSuggestions] = useState([]);

  useEffect(() => {
    setQuery(selectedAsset);
  }, [selectedAsset]);

  useEffect(() => {
    if (!query.trim() || query.trim().length < 2) {
      setSuggestions([]);
      return undefined;
    }

    const timer = window.setTimeout(() => {
      searchAssets(query.trim(), 6)
        .then((payload) => setSuggestions(payload.results || []))
        .catch(() => setSuggestions([]));
    }, 220);

    return () => window.clearTimeout(timer);
  }, [query]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!query.trim()) return;
    setSuggestions([]);
    await loadAssetAnalysis(query.trim()).catch(() => {});
  };

  const featureRows = useMemo(
    () =>
      Object.entries(dashboardData.features || {})
        .map(([name, value]) => ({ name, label: name.replaceAll("_", " "), value: Number(value) }))
        .filter((item) => Number.isFinite(item.value))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 6),
    [dashboardData.features],
  );

  const featureNarrativeRows = useMemo(
    () =>
      featureRows.map((item) => {
        const copy = featureNarratives[item.name] || { label: item.label, bullish: "The feature is supportive.", bearish: "The feature is weakening." };
        const direction = item.value >= 0 ? "bullish" : "bearish";
        return {
          label: copy.label,
          direction,
          value: item.value,
          text: direction === "bullish" ? copy.bullish : copy.bearish,
        };
      }),
    [featureRows],
  );

  const recommendation = dashboardData.recommendation;
  const userCapital = Number(session.user?.capital || 100000);
  const simulationCapital = Math.max(userCapital * 0.1, 10000);
  const alpha = Number(recommendation?.alpha || 0);
  const confidence = Number(recommendation?.confidence || 0);
  const volatility = Math.abs(Number(dashboardData.features?.volatility_20 || 0.02));
  const expectedValue = simulationCapital * (1 + alpha * 3);
  const bestCaseValue = simulationCapital * (1 + Math.max(alpha, 0) * 6 + confidence * 0.08);
  const worstCaseValue = simulationCapital * (1 - Math.max(volatility * 4, 0.06));
  const regimeCard = regimeCopy[dashboardData.regime] || regimeCopy.normal;

  const decisionSummary = recommendation
    ? {
        insight:
          recommendation.recommendation === "BUY"
            ? "Momentum and current structure are supportive enough for selective upside exposure."
            : recommendation.recommendation === "AVOID"
              ? "The signal is not strong enough to justify fresh risk right now."
              : "The signal is mixed, so patience is better than forcing a trade.",
        action:
          recommendation.recommendation === "BUY"
            ? `Consider deploying ${session.user?.risk_level === "aggressive" ? "12-18%" : session.user?.risk_level === "balanced" ? "8-12%" : "4-8%"} of available capital.`
            : recommendation.recommendation === "AVOID"
              ? "Stay light and wait for momentum or volatility to improve."
              : "Hold existing exposure or watch for a clearer setup before adding.",
        horizon: dashboardData.regime === "crisis" ? "1-3 days risk window" : "3-10 days decision window",
      }
    : null;

  return (
    <main className="space-y-6">
      <section className="panel overflow-visible p-6 sm:p-8">
        <div className="grid gap-6 lg:grid-cols-[1.3fr_0.95fr] lg:items-end">
          <div>
            <p className="text-[0.7rem] uppercase tracking-[0.35em] text-pulse">Dashboard</p>
            <h2 className="mt-3 text-4xl font-semibold text-white sm:text-5xl">A clearer, more human explanation of what the model sees right now.</h2>
            <p className="mt-4 max-w-2xl text-base leading-8 text-mist/72">
              Search any supported stock or crypto, then view the recommendation, what is driving it, what the market regime means, and what that could look like for your capital.
            </p>
          </div>
          <form onSubmit={handleSubmit} className="relative rounded-[1.8rem] border border-white/10 bg-black/25 p-4">
            <label className="text-[0.65rem] uppercase tracking-[0.35em] text-frost/70">Analyze any asset</label>
            <div className="mt-3 flex flex-col gap-3 sm:flex-row">
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Apple, Reliance, BTC, NVDA"
                className="glass-input min-w-0 flex-1 px-4 py-3"
              />
              <button type="submit" className="rounded-full bg-pulse px-5 py-3 font-semibold text-ink transition hover:brightness-105">
                Analyze
              </button>
            </div>
            {suggestions.length ? (
              <div className="absolute left-4 right-4 top-[5.8rem] z-20 rounded-[1.4rem] border border-white/10 bg-[#07111d]/98 p-2 shadow-glow backdrop-blur-xl">
                {suggestions.map((item) => (
                  <button
                    key={`${item.symbol}-${item.source}`}
                    type="button"
                    onClick={() => {
                      setQuery(item.symbol);
                      setSuggestions([]);
                      loadAssetAnalysis(item.symbol).catch(() => {});
                    }}
                    className="flex w-full items-center justify-between rounded-[1rem] px-3 py-3 text-left transition hover:bg-white/6"
                  >
                    <span>
                      <span className="block font-semibold text-white">{item.symbol}</span>
                      <span className="text-xs text-mist/58">{item.name}</span>
                    </span>
                    <span className="text-[0.65rem] uppercase tracking-[0.22em] text-frost/70">{item.asset_type}</span>
                  </button>
                ))}
              </div>
            ) : null}
            <div className="mt-4 flex flex-wrap gap-3 text-sm text-mist/60">
              <span>Asset: {selectedAsset}</span>
              <span>Regime: {dashboardData.regime}</span>
              <span>Confidence: {recommendation ? `${Math.round(recommendation.confidence * 100)}%` : "-"}</span>
            </div>
          </form>
        </div>
      </section>

      {dashboardData.error ? (
        <div className="rounded-[1.5rem] border border-ember/30 bg-ember/12 px-5 py-4 text-sm text-ember">{dashboardData.error}</div>
      ) : null}

      <section className="grid gap-6 xl:grid-cols-[1.05fr_1.1fr_0.85fr]">
        <Panel title="Decision Summary" eyebrow="What the system would actually tell a normal investor">
          {dashboardData.loading ? <LoadingBlock /> : null}
          {!dashboardData.loading && decisionSummary ? (
            <div className="space-y-4">
              <div className="rounded-[1.6rem] border border-white/8 bg-black/25 p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">Recommendation</p>
                    <h3 className={`mt-3 text-4xl font-semibold ${signalTone[recommendation.recommendation] || "text-white"}`}>
                      {recommendation.recommendation}
                    </h3>
                  </div>
                  <div className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-sm text-mist/70">{selectedAsset}</div>
                </div>
                <p className="mt-5 text-base leading-7 text-mist/80">{decisionSummary.insight}</p>
              </div>
              <InsightRow label="Action" value={decisionSummary.action} />
              <InsightRow label="Horizon" value={decisionSummary.horizon} />
              <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-1 2xl:grid-cols-3">
                <Metric label="Alpha" value={alpha.toFixed(4)} tone="text-white" />
                <Metric label="Confidence" value={`${Math.round(confidence * 100)}%`} tone="text-frost" />
                <Metric label="Risk Level" value={session.user?.risk_level || "balanced"} tone="text-pulse" />
              </div>
            </div>
          ) : null}
        </Panel>

        <Panel title="Price Structure" eyebrow="Live market context">
          <div className="h-[320px] w-full">
            <ResponsiveContainer>
              <AreaChart data={dashboardData.priceData} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}>
                <defs>
                  <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6df0c2" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#6df0c2" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                <XAxis dataKey="timestamp" tick={{ fill: "#d9e8f5", fontSize: 11 }} minTickGap={28} />
                <YAxis tick={{ fill: "#d9e8f5", fontSize: 11 }} domain={["auto", "auto"]} />
                <Tooltip contentStyle={{ backgroundColor: "#08111d", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "16px" }} />
                <Area type="monotone" dataKey="close" stroke="#6df0c2" strokeWidth={2.2} fill="url(#priceFill)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-xs uppercase tracking-[0.22em] text-mist/48">Data as of {dashboardData.priceAsOf || "latest completed market bar"}</p>
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            <MiniStory title="Trend read" text={alpha >= 0 ? "Price structure still supports further upside if momentum holds." : "Trend quality is weakening and upside follow-through is less reliable."} />
            <MiniStory title="Risk read" text={volatility > 0.03 ? "Volatility is elevated, so this setup needs smaller sizing." : "Volatility is relatively controlled for the current setup."} />
          </div>
        </Panel>

        <Panel title="Regime Explanation" eyebrow="What the market state means">
          <div className="space-y-4">
            <div className="rounded-[1.6rem] border border-white/8 bg-black/25 p-5">
              <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">Market state</p>
              <h3 className="mt-3 text-3xl font-semibold text-white">{regimeCard.title}</h3>
              <p className="mt-4 text-sm leading-7 text-mist/76">{regimeCard.detail}</p>
            </div>
            <InsightRow label="What to do" value={regimeCard.action} />
            <InsightRow label="Why it matters" value={session.user?.risk_level === "aggressive" ? "You can still participate, but the setup should respect volatility." : "A calmer sizing plan matters more than raw upside here."} />
          </div>
        </Panel>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <Panel title="Why This Decision" eyebrow="Feature contribution story">
          <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
            <div className="h-[290px] w-full">
              <ResponsiveContainer>
                <BarChart data={featureRows} layout="vertical" margin={{ top: 0, right: 12, left: 20, bottom: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.08)" horizontal={false} />
                  <XAxis type="number" tick={{ fill: "#d9e8f5", fontSize: 11 }} />
                  <YAxis dataKey="label" type="category" width={105} tick={{ fill: "#d9e8f5", fontSize: 11 }} />
                  <Tooltip contentStyle={{ backgroundColor: "#08111d", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "16px" }} />
                  <Bar dataKey="value" radius={[0, 999, 999, 0]} fill="#91c8ff" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-3">
              {featureNarrativeRows.map((item) => (
                <div key={item.label} className="rounded-[1.3rem] border border-white/8 bg-white/5 px-4 py-4">
                  <div className="flex items-center justify-between gap-3">
                    <p className="font-semibold text-white">{item.label}</p>
                    <span className={`rounded-full px-2.5 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.18em] ${item.direction === "bullish" ? "bg-pulse/15 text-pulse" : "bg-ember/15 text-ember"}`}>
                      {item.direction}
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-mist/72">{item.text}</p>
                </div>
              ))}
            </div>
          </div>
        </Panel>

        <Panel title="What If You Invest?" eyebrow="Make the signal easier to understand">
          <div className="grid gap-4 md:grid-cols-3">
            <ScenarioCard label="Model allocation idea" value={formatCurrency(simulationCapital)} hint="Illustrative size based on your capital and risk profile" />
            <ScenarioCard label="Expected case" value={formatCurrency(expectedValue)} hint="Moderate outcome if the current alpha persists" />
            <ScenarioCard label="Best / Worst" value={`${formatCurrency(bestCaseValue)} / ${formatCurrency(worstCaseValue)}`} hint="A simple range to show upside vs downside" />
          </div>
          <div className="mt-6 rounded-[1.6rem] border border-white/8 bg-black/25 p-5">
            <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">Human translation</p>
            <p className="mt-3 text-base leading-7 text-mist/80">
              {recommendation?.recommendation === "BUY"
                ? "The model sees enough upside to justify a measured position, not an all-in decision."
                : recommendation?.recommendation === "AVOID"
                  ? "The risk-reward is not attractive enough right now, so preserving capital matters more than forcing an entry."
                  : "This is a watchlist name for now. The signal is useful, but not decisive enough to demand immediate action."}
            </p>
          </div>
        </Panel>
      </section>

      <Panel title="Alpha Timeline" eyebrow="Recent conviction and regime shifts">
        <p className="mb-4 text-xs uppercase tracking-[0.22em] text-mist/48">Alpha series is generated from the latest completed feature snapshot. Current as of {dashboardData.alphaAsOf || dashboardData.priceAsOf || "latest completed bar"}.</p>
        <div className="h-[260px] w-full">
          <ResponsiveContainer>
            <LineChart data={dashboardData.alphaSeries} margin={{ top: 10, right: 14, left: -16, bottom: 0 }}>
              <XAxis dataKey="timestamp" tick={{ fill: "#d9e8f5", fontSize: 11 }} minTickGap={24} />
              <YAxis tick={{ fill: "#d9e8f5", fontSize: 11 }} domain={["auto", "auto"]} />
              <Tooltip contentStyle={{ backgroundColor: "#08111d", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "16px" }} />
              <Line type="monotone" dataKey="alpha" stroke="#ff9f5a" strokeWidth={2.3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>
    </main>
  );
}

function Metric({ label, value, tone }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/5 px-4 py-4">
      <p className="text-[0.65rem] uppercase tracking-[0.3em] text-mist/55">{label}</p>
      <p className={`mt-3 text-2xl font-semibold ${tone}`}>{value}</p>
    </div>
  );
}

function InsightRow({ label, value }) {
  return (
    <div className="rounded-[1.3rem] border border-white/8 bg-white/5 px-4 py-4">
      <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">{label}</p>
      <p className="mt-3 text-sm leading-7 text-mist/80">{value}</p>
    </div>
  );
}

function MiniStory({ title, text }) {
  return (
    <div className="rounded-[1.3rem] border border-white/8 bg-white/5 px-4 py-4">
      <p className="text-sm font-semibold text-white">{title}</p>
      <p className="mt-2 text-sm leading-6 text-mist/72">{text}</p>
    </div>
  );
}

function ScenarioCard({ label, value, hint }) {
  return (
    <div className="rounded-[1.5rem] border border-white/8 bg-white/5 px-4 py-5">
      <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/55">{label}</p>
      <p className="mt-4 text-2xl font-semibold text-white">{value}</p>
      <p className="mt-2 text-sm leading-6 text-mist/68">{hint}</p>
    </div>
  );
}

function LoadingBlock() {
  return <div className="skeleton h-52 rounded-[1.6rem]" />;
}
