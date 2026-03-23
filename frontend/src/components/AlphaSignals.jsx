import { useEffect, useState } from "react";
import { getAlphaSignals } from "../api/api";

const emptyState = { loading: true, error: "", data: [] };

export default function AlphaSignals() {
  const [{ loading, error, data }, setState] = useState(emptyState);

  useEffect(() => {
    let active = true;
    getAlphaSignals()
      .then((res) => {
        if (!active) return;
        const sorted = [...res.data].sort((a, b) => b.alpha - a.alpha);
        setState({ loading: false, error: "", data: sorted });
      })
      .catch(() => {
        if (!active) return;
        setState({ loading: false, error: "Unable to load alpha signals.", data: [] });
      });

    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="panel p-5">
      <div className="mb-5 flex items-center justify-between">
        <h2 className="panel-title">Alpha Signals</h2>
        <span className="rounded-full bg-ember/15 px-3 py-1 text-xs uppercase tracking-[0.25em] text-ember">
          Ranked
        </span>
      </div>

      {loading ? <p className="text-sm text-mist/70">Loading alpha...</p> : null}
      {error ? <p className="text-sm text-ember">{error}</p> : null}

      {!loading && !error ? (
        <div className="space-y-3">
          {data.map((item) => (
            <div key={item.asset} className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
              <div className="flex items-center justify-between">
                <span className="text-sm uppercase tracking-[0.2em] text-mist/60">{item.asset}</span>
                <span className="font-semibold text-pulse">{item.alpha.toFixed(4)}</span>
              </div>
              <div className="mt-3 h-2 rounded-full bg-white/10">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-frost to-pulse"
                  style={{ width: `${Math.max(10, Math.min(100, item.alpha * 500))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
