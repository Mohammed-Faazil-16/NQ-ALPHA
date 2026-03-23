import { useEffect, useMemo, useRef, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import Panel from "../components/Panel";
import { usePlatform } from "../context/PlatformContext";

const STORAGE_KEY = "neuroquant_strategy_lab";

function normalizeAmount(value) {
  return Math.max(0, Math.round(Number(value || 0)));
}

function createStrategyAsset(symbol, amount = 0, weight = 0) {
  return {
    id: `${symbol || "asset"}-${Math.random().toString(36).slice(2, 9)}`,
    symbol,
    amount: normalizeAmount(amount),
    weight,
  };
}

function loadStoredStrategy(defaultAssets, selectedAsset, capitalBase) {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {
        assets: defaultAssets(selectedAsset).map((asset) => createStrategyAsset(asset.symbol, normalizeAmount(capitalBase * asset.weight), asset.weight)),
        lookbackDays: 180,
      };
    }

    const parsed = JSON.parse(raw);
    const assets = Array.isArray(parsed.assets)
      ? parsed.assets.map((asset) => createStrategyAsset(
          asset.symbol || "",
          normalizeAmount(asset.amount || (capitalBase * Number(asset.weight || 0))),
          Number(asset.weight || 0),
        ))
      : [];
    return {
      assets,
      lookbackDays: Number(parsed.lookbackDays || 180),
    };
  } catch {
    return {
      assets: defaultAssets(selectedAsset).map((asset) => createStrategyAsset(asset.symbol, normalizeAmount(capitalBase * asset.weight), asset.weight)),
      lookbackDays: 180,
    };
  }
}

function MetricsStrip({ metrics, baseline }) {
  const cards = [
    { label: "Return", value: `${(metrics.final_return * 100).toFixed(2)}%`, tone: "text-white" },
    { label: "Sharpe", value: metrics.sharpe.toFixed(2), tone: "text-pulse" },
    { label: "Drawdown", value: `${(metrics.drawdown * 100).toFixed(2)}%`, tone: "text-ember" },
    { label: "Volatility", value: `${(metrics.volatility * 100).toFixed(2)}%`, tone: "text-frost" },
    { label: "Baseline", value: `${(baseline.final_return * 100).toFixed(2)}%`, tone: "text-white" },
  ];

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
      {cards.map((card) => (
        <div key={card.label} className="metric-chip">
          <p className="text-[0.65rem] uppercase tracking-[0.3em] text-mist/55">{card.label}</p>
          <p className={`mt-3 text-2xl font-semibold ${card.tone}`}>{card.value}</p>
        </div>
      ))}
    </div>
  );
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value || 0);
}

export default function StrategyLab() {
  const {
    selectedAsset,
    session,
    strategyState,
    allocationState,
    runStrategyBacktest,
    loadAllocation,
    saveAllocationPlan,
    defaultStrategyAssets,
    setActivePage,
  } = usePlatform();
  const capitalBase = Number(session.user?.capital || 100000);
  const initialState = useMemo(() => loadStoredStrategy(defaultStrategyAssets, selectedAsset, capitalBase), [capitalBase, selectedAsset]);
  const [assets, setAssets] = useState(initialState.assets);
  const [lookbackDays, setLookbackDays] = useState(initialState.lookbackDays);
  const [builderError, setBuilderError] = useState("");
  const hasHydratedFromServer = useRef(false);

  useEffect(() => {
    loadAllocation().catch(() => {});
    if (!strategyState.results && !strategyState.loading && assets.length) {
      runStrategyBacktest({
        assets: assets.map(({ symbol, amount }) => ({ symbol, amount })),
        lookbackDays,
        capital: capitalBase,
        investmentHorizon: session.user?.investment_horizon || "3-5 years",
      }).catch(() => {});
    }
  }, []);

  useEffect(() => {
    if (hasHydratedFromServer.current) return;
    const remoteAllocations = allocationState.data?.allocations || [];
    if (!remoteAllocations.length) return;
    hasHydratedFromServer.current = true;
    setAssets(remoteAllocations.map((asset) => createStrategyAsset(asset.symbol, asset.amount, asset.weight)));
    setLookbackDays(Number(allocationState.data?.lookback_days || 180));
  }, [allocationState.data]);

  useEffect(() => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        assets: assets.map(({ symbol, amount, weight }) => ({ symbol, amount, weight })),
        lookbackDays,
      }),
    );
  }, [assets, lookbackDays]);

  useEffect(() => {
    if (!session.token || !session.user?.id) return;
    const timer = window.setTimeout(() => {
      saveAllocationPlan({
        assets: assets.map(({ symbol, amount }) => ({ symbol, amount })),
        lookbackDays,
      }).catch(() => {});
    }, 600);

    return () => window.clearTimeout(timer);
  }, [assets, lookbackDays, session.user?.id]);

  const validAssets = useMemo(
    () => assets.filter((asset) => String(asset.symbol || "").trim() && Number(asset.amount || 0) > 0),
    [assets],
  );

  const allocatedCapital = validAssets.reduce((sum, asset) => sum + Number(asset.amount || 0), 0);
  const remainingCapital = Math.max(capitalBase - allocatedCapital, 0);
  const totalInvestedForWeights = allocatedCapital || 1;

  const allocationRows = useMemo(
    () =>
      validAssets.map((asset) => ({
        ...asset,
        normalizedWeight: Number(asset.amount || 0) / totalInvestedForWeights,
        portfolioWeight: capitalBase > 0 ? Number(asset.amount || 0) / capitalBase : 0,
      })),
    [validAssets, totalInvestedForWeights, capitalBase],
  );

  const portfolioIntelligence = useMemo(() => {
    const stable = allocationRows.filter((asset) => asset.portfolioWeight <= 0.25).reduce((sum, asset) => sum + asset.portfolioWeight, 0);
    const volatile = allocationRows.filter((asset) => asset.portfolioWeight > 0.25 && asset.portfolioWeight <= 0.4).reduce((sum, asset) => sum + asset.portfolioWeight, 0);
    const concentrated = allocationRows.filter((asset) => asset.portfolioWeight > 0.4).reduce((sum, asset) => sum + asset.portfolioWeight, 0);
    const topHolding = [...allocationRows].sort((a, b) => b.amount - a.amount)[0];
    const riskScore = Math.min(100, Math.round(concentrated * 100 + volatile * 70 + stable * 35));
    const suggestion = concentrated > 0.35
      ? `Reduce concentration in ${topHolding?.symbol || "the top holding"} and spread that exposure across one or two complementary names.`
      : volatile > 0.45
        ? "The portfolio is leaning into aggressive names. Pair them with steadier exposures if drawdown control matters."
        : "The allocation is relatively balanced. Focus on improving symbol selection rather than adding complexity.";

    return {
      stable,
      volatile,
      concentrated,
      topHolding,
      riskScore,
      suggestion,
    };
  }, [allocationRows]);

  const chartData = useMemo(() => {
    const strategyCurve = strategyState.results?.equity_curve ?? [];
    const baselineCurve = strategyState.results?.baseline_curve ?? [];
    return strategyCurve.map((point, index) => ({
      timestamp: point.timestamp,
      strategy: point.value,
      baseline: baselineCurve[index]?.value ?? null,
    }));
  }, [strategyState.results]);

  const projection = strategyState.results?.projection;

  const submitBacktest = async (event) => {
    event.preventDefault();
    setBuilderError("");

    const payloadAssets = assets
      .map(({ symbol, amount }) => ({ symbol: String(symbol || "").trim().toUpperCase(), amount: Number(amount || 0) }))
      .filter((asset) => asset.symbol && asset.amount > 0);

    if (!payloadAssets.length) {
      setBuilderError("Add at least one asset with a positive rupee amount before running the backtest.");
      return;
    }

    if (payloadAssets.reduce((sum, asset) => sum + asset.amount, 0) > capitalBase + 1) {
      setBuilderError("Allocated amount is greater than your capital base. Reduce the total or leave some cash unallocated.");
      return;
    }

    await saveAllocationPlan({
      assets: payloadAssets,
      lookbackDays,
    }).catch(() => {});
    await runStrategyBacktest({
      assets: payloadAssets,
      lookbackDays,
      capital: capitalBase,
      investmentHorizon: session.user?.investment_horizon || "3-5 years",
    }).catch(() => {});
  };

  return (
    <main className="space-y-6">
      <section className="panel p-6 sm:p-8">
        <p className="text-[0.7rem] uppercase tracking-[0.35em] text-ember/75">Strategy Lab</p>
        <h2 className="mt-3 text-4xl font-semibold text-white">Plan in rupees, keep cash unallocated if you want, and see what the portfolio could become over your chosen horizon.</h2>
      </section>

      <section className="grid gap-6 xl:grid-cols-4">
        <MetricTile label="Capital base" value={formatCurrency(capitalBase)} />
        <MetricTile label="Allocated now" value={formatCurrency(allocatedCapital)} />
        <MetricTile label="Remaining cash" value={formatCurrency(remainingCapital)} />
        <MetricTile label="Horizon" value={session.user?.investment_horizon || "3-5 years"} />
      </section>

      <section className="grid gap-6 xl:grid-cols-3">
        <Panel title="Portfolio Intelligence" eyebrow="What this mix looks like right now">
          <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-1">
            <RiskCard label="Stable weight" value={`${(portfolioIntelligence.stable * 100).toFixed(0)}%`} tone="text-pulse" />
            <RiskCard label="Volatile weight" value={`${(portfolioIntelligence.volatile * 100).toFixed(0)}%`} tone="text-frost" />
            <RiskCard label="High-risk concentration" value={`${(portfolioIntelligence.concentrated * 100).toFixed(0)}%`} tone="text-ember" />
          </div>
          <div className="mt-5 rounded-[1.5rem] border border-white/8 bg-black/25 p-5">
            <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">Risk meter</p>
            <div className="mt-4 h-3 rounded-full bg-white/8">
              <div className="h-full rounded-full bg-gradient-to-r from-pulse via-frost to-ember" style={{ width: `${portfolioIntelligence.riskScore}%` }} />
            </div>
            <p className="mt-3 text-sm text-mist/72">Score: {portfolioIntelligence.riskScore} / 100</p>
          </div>
        </Panel>

        <Panel title="Allocation Snapshot" eyebrow="What you are actually building">
          <div className="space-y-3">
            {allocationRows.length ? allocationRows.map((asset) => (
              <div key={asset.id} className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-lg font-semibold text-white">{asset.symbol}</p>
                  <p className="text-sm text-mist/70">{(asset.portfolioWeight * 100).toFixed(2)}%</p>
                </div>
                <div className="mt-3 h-2 rounded-full bg-white/8">
                  <div className="h-full rounded-full bg-gradient-to-r from-pulse to-frost" style={{ width: `${asset.portfolioWeight * 100}%` }} />
                </div>
                <p className="mt-3 text-sm text-mist/72">Planned capital: {formatCurrency(asset.amount)}</p>
              </div>
            )) : (
              <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4 text-sm text-mist/72">
                No valid positions yet. Add assets in the builder to see money allocation here.
              </div>
            )}
          </div>
        </Panel>

        <Panel title="Future Profit View" eyebrow="Based on your saved horizon">
          {projection ? (
            <div className="space-y-4">
              <RiskCard label="Horizon" value={projection.horizon_label} tone="text-white" />
              <RiskCard
                label="Projected value"
                value={`${formatCurrency(projection.projected_value_low)} to ${formatCurrency(projection.projected_value_high)}`}
                tone="text-pulse"
              />
              <RiskCard
                label="Projected profit"
                value={`${formatCurrency(projection.projected_profit_low)} to ${formatCurrency(projection.projected_profit_high)}`}
                tone="text-frost"
              />
              <RiskCard label="Annualized return" value={`${Number(projection.annualized_return_pct || 0).toFixed(2)}%`} tone="text-white" />
            </div>
          ) : (
            <div className="rounded-[1.5rem] border border-white/8 bg-white/5 p-5 text-sm leading-7 text-mist/72">
              Run the backtest and the system will translate historical performance into an estimated value and profit range for your selected horizon.
            </div>
          )}
        </Panel>
      </section>

      <section className="grid gap-6 xl:grid-cols-[0.95fr_1.45fr]">
        <Panel title="Portfolio Builder" eyebrow="Interactive Controls">
          <form onSubmit={submitBacktest} className="space-y-4">
            <div className="rounded-[1.35rem] border border-frost/18 bg-frost/8 px-4 py-3 text-sm text-frost">
              Enter planned rupee amounts here. The system will convert them into portfolio weights automatically and keep any unallocated money as cash.
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <BuilderStat label="Rows" value={`${assets.length}`} />
              <BuilderStat label="Allocated" value={formatCurrency(allocatedCapital)} />
              <BuilderStat label="Remaining" value={formatCurrency(remainingCapital)} />
            </div>
            {assets.length ? assets.map((asset, index) => {
              const localAmount = Number(asset.amount || 0);
              const derivedPercent = capitalBase > 0 ? (localAmount / capitalBase) * 100 : 0;
              return (
                <div key={asset.id} className="rounded-[1.45rem] border border-white/8 bg-white/5 p-3">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <p className="text-xs font-semibold uppercase tracking-[0.24em] text-mist/55">Position {index + 1}</p>
                    <button
                      type="button"
                      onClick={() => removeAsset(asset.id)}
                      className="rounded-full border border-ember/30 bg-ember/10 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] text-ember transition hover:bg-ember/18"
                    >
                      Delete
                    </button>
                  </div>
                  <div className="grid gap-3 md:grid-cols-[1.1fr_1fr]">
                    <label className="space-y-2">
                      <span className="text-[0.7rem] uppercase tracking-[0.2em] text-mist/55">Asset</span>
                      <input
                        value={asset.symbol}
                        onChange={(event) => updateAsset(asset.id, "symbol", event.target.value.toUpperCase())}
                        className="glass-input w-full px-4 py-3"
                        placeholder="AAPL"
                      />
                    </label>
                    <label className="space-y-2">
                      <span className="text-[0.7rem] uppercase tracking-[0.2em] text-mist/55">Planned rupees</span>
                      <input
                        type="number"
                        inputMode="numeric"
                        step="100"
                        min="0"
                        value={asset.amount}
                        onChange={(event) => updateAsset(asset.id, "amount", normalizeAmount(event.target.value))}
                        className="glass-input w-full px-4 py-3"
                        placeholder="50000"
                      />
                    </label>
                  </div>
                  <p className="mt-3 text-sm text-mist/62">This row currently represents {derivedPercent.toFixed(2)}% of your total capital.</p>
                </div>
              );
            }) : (
              <div className="rounded-[1.45rem] border border-white/8 bg-white/5 p-4 text-sm text-mist/72">
                Your builder is empty. Add an asset to start creating a saved allocation plan.
              </div>
            )}
            <div className="flex flex-wrap gap-3">
              <button type="button" onClick={addAsset} className="rounded-full bg-white/8 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/12">
                Add Asset
              </button>
              <button type="button" onClick={useSelectedAsset} className="rounded-full border border-white/12 px-4 py-2 text-sm font-semibold text-mist/80 transition hover:bg-white/8 hover:text-white">
                Use {selectedAsset}
              </button>
              <button type="button" onClick={allocateRemainingToSelected} className="rounded-full border border-frost/20 bg-frost/10 px-4 py-2 text-sm font-semibold text-frost transition hover:bg-frost/16">
                Put Remaining Into {selectedAsset}
              </button>
              <button type="button" onClick={resetDraft} className="rounded-full border border-white/12 px-4 py-2 text-sm font-semibold text-mist/80 transition hover:bg-white/8 hover:text-white">
                Reset Draft
              </button>
              <button type="button" onClick={() => setActivePage("allocation")} className="rounded-full border border-frost/20 bg-frost/10 px-4 py-2 text-sm font-semibold text-frost transition hover:bg-frost/16">
                Open Allocation Page
              </button>
              <button type="button" onClick={() => setActivePage("guide")} className="rounded-full border border-pulse/20 bg-pulse/10 px-4 py-2 text-sm font-semibold text-pulse transition hover:bg-pulse/16">
                Explain Alpha + Metrics
              </button>
              <label className="space-y-2 rounded-[1.2rem] border border-white/10 bg-white/5 px-4 py-2 text-sm text-mist/80">
                <span className="block text-[0.7rem] uppercase tracking-[0.2em] text-mist/55">Lookback days</span>
                <input
                  type="number"
                  min="30"
                  max="3650"
                  step="1"
                  value={lookbackDays}
                  onChange={(event) => setLookbackDays(Number(event.target.value) || 30)}
                  className="w-28 bg-transparent text-base font-semibold text-white outline-none"
                />
              </label>
              <button type="submit" className="rounded-full bg-pulse px-5 py-3 font-semibold text-ink transition hover:brightness-105">
                Run Backtest
              </button>
            </div>
            <p className="text-sm text-mist/58">This user’s portfolio draft is saved to the backend and also cached locally while you edit.</p>
            {builderError ? <p className="rounded-2xl bg-ember/15 px-4 py-3 text-sm text-ember">{builderError}</p> : null}
            {allocationState.error ? <p className="rounded-2xl bg-ember/15 px-4 py-3 text-sm text-ember">{allocationState.error}</p> : null}
            {strategyState.error ? <p className="rounded-2xl bg-ember/15 px-4 py-3 text-sm text-ember">{strategyState.error}</p> : null}
          </form>
        </Panel>

        <Panel title="Performance" eyebrow="Backtest Output">
          {strategyState.loading ? <div className="skeleton h-[410px] rounded-[1.6rem]" /> : null}
          {strategyState.results ? (
            <div className="space-y-5">
              <MetricsStrip metrics={strategyState.results.metrics} baseline={strategyState.results.baseline_metrics} />
              <div className="grid gap-3 sm:grid-cols-3">
                <BuilderStat label="Invested fraction" value={`${((strategyState.results.invested_fraction || 0) * 100).toFixed(2)}%`} />
                <BuilderStat label="Cash fraction" value={`${((strategyState.results.cash_fraction || 0) * 100).toFixed(2)}%`} />
                <BuilderStat label="Annualized return" value={`${((strategyState.results.metrics?.annualized_return || 0) * 100).toFixed(2)}%`} />
              </div>
              <div className="h-[360px] w-full">
                <ResponsiveContainer>
                  <LineChart data={chartData} margin={{ top: 10, right: 16, left: -18, bottom: 0 }}>
                    <XAxis dataKey="timestamp" tick={{ fill: "#d9e8f5", fontSize: 11 }} minTickGap={24} />
                    <YAxis tick={{ fill: "#d9e8f5", fontSize: 11 }} domain={["auto", "auto"]} />
                    <Tooltip contentStyle={{ backgroundColor: "#08111d", border: "1px solid rgba(255,255,255,0.12)", borderRadius: "16px" }} />
                    <Line type="monotone" dataKey="strategy" stroke="#6df0c2" strokeWidth={2.3} dot={false} />
                    <Line type="monotone" dataKey="baseline" stroke="#ff9f5a" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ) : (
            <p className="text-sm text-mist/70">Run a backtest to inspect return, Sharpe, drawdown, volatility, and the equity curve.</p>
          )}
        </Panel>
      </section>
    </main>
  );

  function updateAsset(assetId, key, value) {
    setAssets((current) =>
      current.map((asset) => {
        if (asset.id !== assetId) return asset;
        if (key === "amount") {
          return { ...asset, amount: normalizeAmount(value) };
        }
        if (key === "symbol") {
          return { ...asset, symbol: String(value || "").trim().toUpperCase() };
        }
        return { ...asset, [key]: value };
      }),
    );
  }

  function getSuggestedAmount() {
    if (remainingCapital > 0) {
      return normalizeAmount(Math.min(remainingCapital, Math.max(capitalBase * 0.1, 10000)));
    }
    return normalizeAmount(Math.max(capitalBase * 0.05, 10000));
  }

  function addAsset() {
    setAssets((current) => [...current, createStrategyAsset(selectedAsset || "", getSuggestedAmount(), 0)]);
  }

  function useSelectedAsset() {
    setAssets((current) => [...current, createStrategyAsset(selectedAsset, getSuggestedAmount(), 0)]);
  }

  function allocateRemainingToSelected() {
    if (remainingCapital <= 0) {
      setBuilderError("No remaining cash is left to allocate.");
      return;
    }
    setBuilderError("");
    setAssets((current) => [...current, createStrategyAsset(selectedAsset, remainingCapital, 0)]);
  }

  function resetDraft() {
    const defaults = defaultStrategyAssets(selectedAsset).map((asset) => createStrategyAsset(asset.symbol, normalizeAmount(capitalBase * asset.weight), asset.weight));
    setAssets(defaults);
    setBuilderError("");
  }

  function removeAsset(assetId) {
    setAssets((current) => current.filter((asset) => asset.id !== assetId));
  }
}

function RiskCard({ label, value, tone }) {
  return (
    <div className="rounded-[1.45rem] border border-white/8 bg-white/5 px-4 py-4">
      <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">{label}</p>
      <p className={`mt-3 text-xl font-semibold capitalize ${tone}`}>{value}</p>
    </div>
  );
}

function BuilderStat({ label, value }) {
  return (
    <div className="rounded-[1.25rem] border border-white/8 bg-white/5 px-4 py-4">
      <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/55">{label}</p>
      <p className="mt-3 text-lg font-semibold text-white">{value}</p>
    </div>
  );
}

function MetricTile({ label, value }) {
  return (
    <div className="metric-chip">
      <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">{label}</p>
      <p className="mt-3 text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}
