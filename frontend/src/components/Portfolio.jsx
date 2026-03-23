import { useEffect, useState } from "react";
import { getPortfolio } from "../api/api";

const emptyState = { loading: true, error: "", data: [] };

export default function Portfolio() {
  const [{ loading, error, data }, setState] = useState(emptyState);

  useEffect(() => {
    let active = true;
    getPortfolio()
      .then((res) => {
        if (!active) return;
        setState({ loading: false, error: "", data: res.data });
      })
      .catch(() => {
        if (!active) return;
        setState({ loading: false, error: "Unable to load portfolio.", data: [] });
      });

    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="panel p-5">
      <div className="mb-5 flex items-center justify-between">
        <h2 className="panel-title">Portfolio</h2>
        <span className="rounded-full bg-pulse/15 px-3 py-1 text-xs uppercase tracking-[0.25em] text-pulse">
          Live Weights
        </span>
      </div>

      {loading ? <p className="text-sm text-mist/70">Loading portfolio...</p> : null}
      {error ? <p className="text-sm text-ember">{error}</p> : null}

      {!loading && !error ? (
        <div className="overflow-hidden rounded-2xl border border-white/10">
          <table className="w-full overflow-hidden text-left text-sm text-mist">
            <thead className="bg-white/5 text-xs uppercase tracking-[0.25em] text-mist/60">
              <tr>
                <th className="px-4 py-3">Asset</th>
                <th className="px-4 py-3 text-right">Weight</th>
              </tr>
            </thead>
            <tbody>
              {data.map((item) => (
                <tr key={item.asset} className="border-t border-white/10">
                  <td className="px-4 py-3 text-white">{item.asset}</td>
                  <td className="px-4 py-3 text-right font-medium text-pulse">
                    {(item.weight * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
