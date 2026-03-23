import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getMetrics } from "../api/api";

const emptyState = { loading: true, error: "", data: [] };

export default function PerformanceChart() {
  const [{ loading, error, data }, setState] = useState(emptyState);

  useEffect(() => {
    let active = true;
    getMetrics()
      .then((res) => {
        if (!active) return;
        setState({ loading: false, error: "", data: res.data.equity_curve ?? [] });
      })
      .catch(() => {
        if (!active) return;
        setState({ loading: false, error: "Unable to load performance.", data: [] });
      });

    return () => {
      active = false;
    };
  }, []);

  return (
    <section className="panel p-5">
      <div className="mb-5 flex items-center justify-between">
        <h2 className="panel-title">Performance</h2>
        <span className="rounded-full bg-frost/15 px-3 py-1 text-xs uppercase tracking-[0.25em] text-frost">
          Equity Curve
        </span>
      </div>

      {loading ? <p className="text-sm text-mist/70">Loading performance...</p> : null}
      {error ? <p className="text-sm text-ember">{error}</p> : null}

      {!loading && !error ? (
        <div className="h-72 w-full">
          <ResponsiveContainer>
            <LineChart data={data} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
              <XAxis dataKey="timestamp" stroke="#a6bfd4" tickLine={false} axisLine={false} />
              <YAxis stroke="#a6bfd4" tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#091726",
                  border: "1px solid rgba(255,255,255,0.12)",
                  borderRadius: "16px",
                  color: "#d9e8f5",
                }}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#6df0c2"
                strokeWidth={3}
                dot={{ r: 0 }}
                activeDot={{ r: 5, fill: "#6df0c2", stroke: "#07111d", strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : null}
    </section>
  );
}
