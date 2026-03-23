import { useEffect, useMemo, useState } from "react";
import Panel from "../components/Panel";
import { usePlatform } from "../context/PlatformContext";

const signalOptions = ["all", "BUY", "HOLD", "AVOID"];
const confidenceOptions = ["all", "high", "medium", "low"];

function confidenceBand(value) {
  if (value >= 0.75) return "high";
  if (value >= 0.45) return "medium";
  return "low";
}

export default function Scanner() {
  const { scannerState, loadScanner, selectScannerAsset, openAsset } = usePlatform();
  const [topN, setTopN] = useState(scannerState.filters.top_n);
  const [assetType, setAssetType] = useState(scannerState.filters.asset_type);
  const [signalFilter, setSignalFilter] = useState("all");
  const [confidenceFilter, setConfidenceFilter] = useState("all");

  useEffect(() => {
    setTopN(scannerState.filters.top_n);
    setAssetType(scannerState.filters.asset_type);
  }, [scannerState.filters]);

  useEffect(() => {
    if (!scannerState.results.length && !scannerState.loading) {
      loadScanner().catch(() => {});
    }
  }, []);

  const handleRefresh = async (event) => {
    event.preventDefault();
    await loadScanner({ top_n: Number(topN), asset_type: assetType }).catch(() => {});
  };

  const filteredResults = useMemo(() => {
    return scannerState.results.filter((item) => {
      const signalOk = signalFilter === "all" || item.recommendation === signalFilter;
      const confidenceOk = confidenceFilter === "all" || confidenceBand(item.confidence) === confidenceFilter;
      return signalOk && confidenceOk;
    });
  }, [scannerState.results, signalFilter, confidenceFilter]);

  const quickView = scannerState.selected || filteredResults[0] || null;
  const bestOpportunity = filteredResults[0] || null;
  const riskyOpportunity = [...filteredResults].reverse().find((item) => item.confidence < 0.5) || filteredResults.at(-1) || null;

  return (
    <main className="space-y-6">
      <section className="panel p-6 sm:p-8">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-[0.7rem] uppercase tracking-[0.35em] text-frost/70">Market Scanner</p>
            <h2 className="mt-3 text-4xl font-semibold text-white">Scan the live universe, then turn ranked signals into an actual watchlist.</h2>
            <p className="mt-4 max-w-2xl text-base leading-8 text-mist/72">
              Filter by asset type, signal quality, and confidence band to surface what actually deserves your attention.
            </p>
          </div>
          <form onSubmit={handleRefresh} className="flex flex-wrap items-end gap-3">
            <label className="scanner-filter">
              <span>Top N</span>
              <select value={topN} onChange={(event) => setTopN(event.target.value)}>
                <option value={10}>10</option>
                <option value={15}>15</option>
                <option value={20}>20</option>
              </select>
            </label>
            <label className="scanner-filter">
              <span>Asset Type</span>
              <select value={assetType} onChange={(event) => setAssetType(event.target.value)}>
                <option value="">All</option>
                <option value="stock">Stock</option>
                <option value="crypto">Crypto</option>
              </select>
            </label>
            <button type="submit" className="rounded-full bg-frost px-5 py-3 font-semibold text-ink transition hover:brightness-105">
              Refresh Scan
            </button>
          </form>
        </div>
      </section>

      {scannerState.error ? (
        <div className="rounded-[1.5rem] border border-ember/30 bg-ember/12 px-5 py-4 text-sm text-ember">{scannerState.error}</div>
      ) : null}

      {scannerState.partial ? (
        <div className="rounded-[1.5rem] border border-frost/25 bg-frost/10 px-5 py-4 text-sm text-frost">
          Partial scan returned in {scannerState.elapsedSeconds.toFixed(2)}s after evaluating {scannerState.evaluated} assets to keep the UI responsive.
        </div>
      ) : null}

      <section className="grid gap-6 xl:grid-cols-3">
        <OpportunityCard
          title="Best Opportunity"
          tone="pulse"
          asset={bestOpportunity}
          description="Highest-ranked live setup right now based on alpha and confidence."
          onOpen={openAsset}
        />
        <OpportunityCard
          title="Need More Caution"
          tone="ember"
          asset={riskyOpportunity}
          description="Signal exists, but confidence or stability is weaker than the leaders."
          onOpen={openAsset}
        />
        <Panel title="Filter Stack" eyebrow="Interactive Control">
          <div className="space-y-4">
            <FilterGroup label="Signal">
              {signalOptions.map((option) => (
                <FilterButton key={option} active={signalFilter === option} onClick={() => setSignalFilter(option)}>
                  {option}
                </FilterButton>
              ))}
            </FilterGroup>
            <FilterGroup label="Confidence">
              {confidenceOptions.map((option) => (
                <FilterButton key={option} active={confidenceFilter === option} onClick={() => setConfidenceFilter(option)}>
                  {option}
                </FilterButton>
              ))}
            </FilterGroup>
          </div>
        </Panel>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.55fr_0.85fr]">
        <Panel title="Ranked Assets" eyebrow="Bulk Intelligence">
          {scannerState.loading ? <div className="skeleton h-72 rounded-[1.6rem]" /> : null}
          {!scannerState.loading ? (
            <div className="overflow-hidden rounded-[1.4rem] border border-white/8 bg-black/20">
              <table className="w-full text-left text-sm text-mist/80">
                <thead className="bg-white/5 text-[0.7rem] uppercase tracking-[0.25em] text-mist/55">
                  <tr>
                    <th className="px-4 py-3">Rank</th>
                    <th className="px-4 py-3">Symbol</th>
                    <th className="px-4 py-3">Alpha</th>
                    <th className="px-4 py-3">Signal</th>
                    <th className="px-4 py-3">Confidence</th>
                    <th className="px-4 py-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredResults.map((item, index) => {
                    const active = quickView?.symbol === item.symbol;
                    return (
                      <tr key={item.symbol} className={active ? "bg-frost/10" : "border-t border-white/6"}>
                        <td className="px-4 py-3">{index + 1}</td>
                        <td className="px-4 py-3">
                          <button type="button" className="font-semibold text-white" onClick={() => selectScannerAsset(item)}>
                            {item.symbol}
                          </button>
                        </td>
                        <td className="px-4 py-3">{item.alpha.toFixed(4)}</td>
                        <td className="px-4 py-3">{item.recommendation}</td>
                        <td className="px-4 py-3">{Math.round(item.confidence * 100)}%</td>
                        <td className="px-4 py-3 text-right">
                          <button type="button" onClick={() => openAsset(item.symbol)} className="rounded-full bg-pulse/90 px-3 py-1 text-xs font-semibold text-ink transition hover:brightness-105">
                            Deep Dive
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : null}
        </Panel>

        <Panel title="Selected Asset" eyebrow="Quick View">
          {quickView ? (
            <div className="space-y-4">
              <div className="rounded-[1.5rem] border border-white/8 bg-black/25 p-5">
                <p className="text-[0.65rem] uppercase tracking-[0.3em] text-mist/55">Selected</p>
                <h3 className="mt-2 text-3xl font-semibold text-white">{quickView.symbol}</h3>
                <p className="mt-2 text-sm text-mist/70">{quickView.name}</p>
              </div>
              <Mini label="Recommendation" value={quickView.recommendation} />
              <Mini label="Alpha" value={quickView.alpha.toFixed(4)} />
              <Mini label="Confidence" value={`${Math.round(quickView.confidence * 100)}%`} />
              <Mini label="Type" value={quickView.asset_type} />
              <Mini
                label="Interpretation"
                value={
                  quickView.recommendation === "BUY"
                    ? "High-priority watchlist candidate"
                    : quickView.recommendation === "AVOID"
                      ? "Low-priority setup for now"
                      : "Worth monitoring, not forcing"
                }
              />
              <button type="button" onClick={() => openAsset(quickView.symbol)} className="w-full rounded-full bg-pulse px-5 py-3 font-semibold text-ink transition hover:brightness-105">
                Open in Dashboard
              </button>
            </div>
          ) : (
            <p className="text-sm text-mist/70">Run a scan to preview the top-ranked asset.</p>
          )}
        </Panel>
      </section>
    </main>
  );
}

function OpportunityCard({ title, tone, asset, description, onOpen }) {
  const toneClasses = tone === "ember" ? "from-ember/18 to-rose/10" : "from-pulse/18 to-frost/10";
  return (
    <Panel title={title} eyebrow="Opportunity Card" className={`bg-gradient-to-br ${toneClasses}`}>
      {asset ? (
        <div className="space-y-4">
          <div>
            <p className="text-3xl font-semibold text-white">{asset.symbol}</p>
            <p className="mt-2 text-sm text-mist/72">{description}</p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <Mini label="Signal" value={asset.recommendation} />
            <Mini label="Confidence" value={`${Math.round(asset.confidence * 100)}%`} />
          </div>
          <button type="button" onClick={() => onOpen(asset.symbol)} className="rounded-full border border-white/12 px-4 py-3 text-sm font-semibold text-white transition hover:bg-white/8">
            Analyze {asset.symbol}
          </button>
        </div>
      ) : (
        <p className="text-sm text-mist/72">Run a scanner refresh to populate this card.</p>
      )}
    </Panel>
  );
}

function FilterGroup({ label, children }) {
  return (
    <div>
      <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">{label}</p>
      <div className="mt-3 flex flex-wrap gap-2">{children}</div>
    </div>
  );
}

function FilterButton({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full px-4 py-2 text-xs font-semibold uppercase tracking-[0.18em] transition ${
        active ? "bg-white text-ink" : "bg-white/6 text-mist/70 hover:bg-white/10"
      }`}
    >
      {children}
    </button>
  );
}

function Mini({ label, value }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/5 px-4 py-4">
      <p className="text-[0.65rem] uppercase tracking-[0.3em] text-mist/55">{label}</p>
      <p className="mt-3 text-lg font-semibold capitalize text-white">{value}</p>
    </div>
  );
}
