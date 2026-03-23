import Panel from "../components/Panel";
import { usePlatform } from "../context/PlatformContext";

function formatCurrency(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value || 0);
}

export default function Allocation() {
  const { allocationState, session, setActivePage } = usePlatform();
  const allocation = allocationState.data;

  return (
    <main className="space-y-6">
      <section className="panel p-6 sm:p-8">
        <p className="text-[0.7rem] uppercase tracking-[0.35em] text-frost/72">Allocation</p>
        <h2 className="mt-3 text-4xl font-semibold text-white">See exactly how this user’s capital is deployed, what is still in cash, and how the saved plan is structured.</h2>
      </section>

      {allocationState.error ? (
        <div className="rounded-[1.5rem] border border-ember/30 bg-ember/12 px-5 py-4 text-sm text-ember">{allocationState.error}</div>
      ) : null}

      <section className="grid gap-6 xl:grid-cols-5">
        <MetricCard label="Capital" value={formatCurrency(allocation?.capital || session.user?.capital || 0)} />
        <MetricCard label="Invested" value={formatCurrency(allocation?.invested_amount || 0)} />
        <MetricCard label="Cash free" value={formatCurrency(allocation?.available_cash_amount || 0)} />
        <MetricCard label="Equity" value={`${Number(allocation?.equity_pct || 0).toFixed(2)}%`} />
        <MetricCard label="Holdings" value={`${allocation?.allocations?.length || 0}`} />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.35fr_0.95fr]">
        <Panel title="User Allocation Plan" eyebrow="Money Distribution">
          {allocationState.loading ? <div className="skeleton h-72 rounded-[1.6rem]" /> : null}
          {!allocationState.loading && allocation?.allocations?.length ? (
            <div className="overflow-hidden rounded-[1.4rem] border border-white/8 bg-black/20">
              <table className="w-full text-left text-sm text-mist/80">
                <thead className="bg-white/5 text-[0.7rem] uppercase tracking-[0.25em] text-mist/55">
                  <tr>
                    <th className="px-4 py-3">Asset</th>
                    <th className="px-4 py-3">Weight</th>
                    <th className="px-4 py-3">Amount</th>
                  </tr>
                </thead>
                <tbody>
                  {allocation.allocations.map((asset) => (
                    <tr key={asset.symbol} className="border-t border-white/6">
                      <td className="px-4 py-3 font-semibold text-white">{asset.symbol}</td>
                      <td className="px-4 py-3">{Number(asset.percent).toFixed(2)}%</td>
                      <td className="px-4 py-3">{formatCurrency(asset.amount)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : !allocationState.loading ? (
            <div className="rounded-[1.5rem] border border-white/8 bg-white/5 p-5 text-sm text-mist/72">
              No saved portfolio allocation yet. Build one in Strategy Lab and it will appear here.
            </div>
          ) : null}
        </Panel>

        <Panel title="Plan Details" eyebrow="Stored For This User">
          <div className="space-y-4">
            <InfoCard label="Strategy" value={allocation?.strategy || "No strategy saved yet"} />
            <InfoCard label="Lookback" value={`${allocation?.lookback_days || 180} days`} />
            <InfoCard label="Cash reserve" value={formatCurrency(allocation?.available_cash_amount || 0)} />
            <InfoCard label="Updated" value={allocation?.updated_at || "Not saved yet"} />
            <button
              type="button"
              onClick={() => setActivePage("strategy")}
              className="w-full rounded-full bg-pulse px-5 py-3 font-semibold text-ink transition hover:brightness-105"
            >
              Open Strategy Lab
            </button>
          </div>
        </Panel>
      </section>
    </main>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="metric-chip">
      <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">{label}</p>
      <p className="mt-3 text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}

function InfoCard({ label, value }) {
  return (
    <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">
      <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/55">{label}</p>
      <p className="mt-3 text-sm leading-6 text-mist/82">{value}</p>
    </div>
  );
}
