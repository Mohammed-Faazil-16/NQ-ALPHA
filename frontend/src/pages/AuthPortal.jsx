import { useMemo, useState } from "react";
import { usePlatform } from "../context/PlatformContext";

const BENEFITS = [
  "Profile-led recommendations instead of one-size-fits-all advice",
  "Live alpha analysis across stocks, crypto, and Indian equities",
  "Strategy continuity so the advisor evolves your plan instead of resetting it",
];

export default function AuthPortal() {
  const { authState, registerUserAction, loginUserAction } = usePlatform();
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ full_name: "", email: "", password: "" });

  const title = useMemo(
    () =>
      mode === "login"
        ? "Return to your AI quant workspace"
        : "Create your personalized AI investing workspace",
    [mode],
  );

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (mode === "login") {
      await loginUserAction({ email: form.email, password: form.password }).catch(() => {});
      return;
    }
    await registerUserAction(form).catch(() => {});
  };

  return (
    <main className="mx-auto flex min-h-screen max-w-7xl items-center px-4 py-8 sm:px-6 lg:px-8">
      <div className="grid w-full gap-6 lg:grid-cols-[1.15fr_0.95fr]">
        <section className="panel relative overflow-hidden p-8 sm:p-10">
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-pulse/80 to-transparent" />
          <p className="text-[0.7rem] uppercase tracking-[0.38em] text-pulse">NQ ALPHA</p>
          <h1 className="mt-4 font-sans text-5xl font-semibold leading-tight text-white sm:text-6xl">
            Your personal AI quant terminal for better investing decisions.
          </h1>
          <p className="mt-5 max-w-2xl text-base leading-8 text-mist/72">
            Sign in once, tell the platform how much capital you have, what markets you care about, and what level of risk you can handle. Everything after that becomes personal.
          </p>

          <div className="mt-10 grid gap-4 sm:grid-cols-3">
            <div className="metric-chip">
              <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">Live engine</p>
              <p className="mt-3 text-3xl font-semibold text-white">Stocks + Crypto</p>
            </div>
            <div className="metric-chip">
              <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">Advisor mode</p>
              <p className="mt-3 text-3xl font-semibold text-white">Profile-aware</p>
            </div>
            <div className="metric-chip">
              <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">Decision layer</p>
              <p className="mt-3 text-3xl font-semibold text-white">Actionable</p>
            </div>
          </div>

          <div className="mt-8 space-y-3">
            {BENEFITS.map((benefit) => (
              <div key={benefit} className="flex items-start gap-3 rounded-2xl border border-white/8 bg-white/4 px-4 py-3 text-sm text-mist/78">
                <span className="mt-0.5 h-2.5 w-2.5 rounded-full bg-pulse shadow-[0_0_18px_rgba(109,240,194,0.7)]" />
                <span>{benefit}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="panel p-6 sm:p-8">
          <div className="flex rounded-full border border-white/10 bg-black/20 p-1">
            {[
              ["login", "Sign In"],
              ["register", "Create Account"],
            ].map(([id, label]) => {
              const active = id === mode;
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setMode(id)}
                  className={`flex-1 rounded-full px-4 py-3 text-sm font-semibold transition ${
                    active ? "bg-white text-ink" : "text-mist/72 hover:text-white"
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </div>

          <div className="mt-8">
            <p className="text-[0.7rem] uppercase tracking-[0.32em] text-frost/72">Access</p>
            <h2 className="mt-3 text-3xl font-semibold text-white">{title}</h2>
          </div>

          <form onSubmit={handleSubmit} className="mt-8 space-y-4">
            {mode === "register" ? (
              <label className="block">
                <span className="mb-2 block text-sm text-mist/72">Full name</span>
                <input
                  value={form.full_name}
                  onChange={(event) => setForm((current) => ({ ...current, full_name: event.target.value }))}
                  className="w-full rounded-[1.35rem] border border-white/12 bg-white/6 px-4 py-3 text-white outline-none transition focus:border-pulse"
                  placeholder="Faazil Mohammed"
                />
              </label>
            ) : null}

            <label className="block">
              <span className="mb-2 block text-sm text-mist/72">Email</span>
              <input
                type="email"
                value={form.email}
                onChange={(event) => setForm((current) => ({ ...current, email: event.target.value }))}
                className="w-full rounded-[1.35rem] border border-white/12 bg-white/6 px-4 py-3 text-white outline-none transition focus:border-pulse"
                placeholder="you@example.com"
                required
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm text-mist/72">Password</span>
              <input
                type="password"
                value={form.password}
                onChange={(event) => setForm((current) => ({ ...current, password: event.target.value }))}
                className="w-full rounded-[1.35rem] border border-white/12 bg-white/6 px-4 py-3 text-white outline-none transition focus:border-pulse"
                placeholder="At least 8 characters"
                minLength={8}
                required
              />
            </label>

            {authState.error ? <p className="rounded-2xl border border-ember/30 bg-ember/10 px-4 py-3 text-sm text-ember">{authState.error}</p> : null}

            <button
              type="submit"
              className="w-full rounded-full bg-pulse px-5 py-3 text-sm font-semibold text-ink transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-70"
              disabled={authState.loading}
            >
              {authState.loading ? "Working..." : mode === "login" ? "Enter Terminal" : "Create Workspace"}
            </button>
          </form>
        </section>
      </div>
    </main>
  );
}
