import { useMemo, useState } from "react";
import { usePlatform } from "../context/PlatformContext";

const CAPITAL_OPTIONS = [
  { id: "small", label: "Below Rs 10,000", value: 10000 },
  { id: "starter", label: "Rs 10,000 - Rs 1L", value: 100000 },
  { id: "growth", label: "Rs 1L - Rs 10L", value: 500000 },
  { id: "serious", label: "Above Rs 10L", value: 1500000 },
];
const RISK_OPTIONS = ["conservative", "balanced", "aggressive"];
const INTEREST_OPTIONS = ["stocks", "crypto", "commodities", "forex"];
const GOAL_OPTIONS = ["long-term wealth", "short-term trading", "passive income", "learning"];
const HORIZON_OPTIONS = ["< 6 months", "1-2 years", "3-5 years", "5+ years"];

export default function Onboarding() {
  const { session, authState, saveProfile } = usePlatform();
  const [step, setStep] = useState(0);
  const [form, setForm] = useState({
    capital: session.user?.capital || 100000,
    risk_level: session.user?.risk_level || "balanced",
    interests: session.user?.interests?.length ? session.user.interests : ["stocks"],
    goals: session.user?.goals || "long-term wealth",
    investment_horizon: session.user?.investment_horizon || "3-5 years",
  });

  const steps = useMemo(
    () => [
      {
        title: "Set your starting capital",
        description: "This helps the advisor turn ideas into allocation guidance that fits your size.",
        content: (
          <div className="grid gap-3 md:grid-cols-2">
            {CAPITAL_OPTIONS.map((option) => {
              const active = form.capital === option.value;
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => setForm((current) => ({ ...current, capital: option.value }))}
                  className={`rounded-[1.6rem] border px-5 py-5 text-left transition ${
                    active ? "border-pulse bg-pulse/10 text-white" : "border-white/10 bg-white/5 text-mist/80 hover:border-white/20"
                  }`}
                >
                  <p className="text-sm font-semibold">{option.label}</p>
                </button>
              );
            })}
            <label className="rounded-[1.6rem] border border-white/10 bg-white/5 px-5 py-5 text-left text-sm text-mist/78 md:col-span-2">
              <span className="mb-2 block uppercase tracking-[0.2em] text-mist/50">Custom capital</span>
              <input
                type="number"
                min="0"
                value={form.capital}
                onChange={(event) => setForm((current) => ({ ...current, capital: Number(event.target.value) || 0 }))}
                className="mt-2 w-full rounded-2xl border border-white/12 bg-black/20 px-4 py-3 text-white outline-none focus:border-pulse"
              />
            </label>
          </div>
        ),
      },
      {
        title: "Choose your risk style",
        description: "The platform should know whether to prioritize capital protection or upside capture.",
        content: (
          <div className="grid gap-3 md:grid-cols-3">
            {RISK_OPTIONS.map((risk) => {
              const active = form.risk_level === risk;
              return (
                <button
                  key={risk}
                  type="button"
                  onClick={() => setForm((current) => ({ ...current, risk_level: risk }))}
                  className={`rounded-[1.6rem] border px-5 py-6 text-left capitalize transition ${
                    active ? "border-pulse bg-pulse/10 text-white" : "border-white/10 bg-white/5 text-mist/80 hover:border-white/20"
                  }`}
                >
                  <p className="text-lg font-semibold">{risk}</p>
                </button>
              );
            })}
          </div>
        ),
      },
      {
        title: "Pick the markets you care about",
        description: "The scanner and advisor will prioritize these asset classes in your workflow.",
        content: (
          <div className="grid gap-3 md:grid-cols-2">
            {INTEREST_OPTIONS.map((interest) => {
              const active = form.interests.includes(interest);
              return (
                <button
                  key={interest}
                  type="button"
                  onClick={() =>
                    setForm((current) => ({
                      ...current,
                      interests: active
                        ? current.interests.filter((item) => item !== interest)
                        : [...current.interests, interest],
                    }))
                  }
                  className={`rounded-[1.6rem] border px-5 py-5 text-left capitalize transition ${
                    active ? "border-frost bg-frost/10 text-white" : "border-white/10 bg-white/5 text-mist/80 hover:border-white/20"
                  }`}
                >
                  <p className="text-base font-semibold">{interest}</p>
                </button>
              );
            })}
          </div>
        ),
      },
      {
        title: "Define the outcome you want",
        description: "This shapes how aggressive or defensive the system should sound when it guides you.",
        content: (
          <div className="grid gap-3 md:grid-cols-2">
            {GOAL_OPTIONS.map((goal) => {
              const active = form.goals === goal;
              return (
                <button
                  key={goal}
                  type="button"
                  onClick={() => setForm((current) => ({ ...current, goals: goal }))}
                  className={`rounded-[1.6rem] border px-5 py-5 text-left transition ${
                    active ? "border-ember bg-ember/10 text-white" : "border-white/10 bg-white/5 text-mist/80 hover:border-white/20"
                  }`}
                >
                  <p className="text-base font-semibold capitalize">{goal}</p>
                </button>
              );
            })}
          </div>
        ),
      },
      {
        title: "Choose your investment horizon",
        description: "Short-term traders and long-term compounding investors should never see the same advice.",
        content: (
          <div className="grid gap-3 md:grid-cols-2">
            {HORIZON_OPTIONS.map((option) => {
              const active = form.investment_horizon === option;
              return (
                <button
                  key={option}
                  type="button"
                  onClick={() => setForm((current) => ({ ...current, investment_horizon: option }))}
                  className={`rounded-[1.6rem] border px-5 py-5 text-left transition ${
                    active ? "border-white bg-white text-ink" : "border-white/10 bg-white/5 text-mist/80 hover:border-white/20"
                  }`}
                >
                  <p className="text-base font-semibold">{option}</p>
                </button>
              );
            })}
          </div>
        ),
      },
      {
        title: "Review your investing profile",
        description: "Once you confirm this, the dashboard, scanner, and advisor become personalized.",
        content: (
          <div className="grid gap-4 md:grid-cols-2">
            <SummaryCard label="Capital" value={`Rs ${Number(form.capital || 0).toLocaleString("en-IN")}`} />
            <SummaryCard label="Risk" value={form.risk_level} />
            <SummaryCard label="Interests" value={form.interests.join(", ") || "Stocks"} />
            <SummaryCard label="Goal" value={form.goals} />
            <SummaryCard label="Horizon" value={form.investment_horizon} />
            <SummaryCard label="Profile" value={session.user?.full_name || session.user?.email || "Investor"} />
          </div>
        ),
      },
    ],
    [form, session.user],
  );

  const isLastStep = step === steps.length - 1;

  const handleNext = async () => {
    if (!isLastStep) {
      setStep((current) => Math.min(current + 1, steps.length - 1));
      return;
    }
    await saveProfile(form).catch(() => {});
  };

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl items-center px-4 py-8 sm:px-6 lg:px-8">
      <section className="panel w-full overflow-hidden p-6 sm:p-8 lg:p-10">
        <div className="flex flex-col gap-8 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-2xl">
            <p className="text-[0.7rem] uppercase tracking-[0.35em] text-pulse">Onboarding</p>
            <h1 className="mt-4 text-4xl font-semibold text-white sm:text-5xl">Let NQ ALPHA learn how you invest before it starts advising you.</h1>
            <p className="mt-4 text-base leading-8 text-mist/72">
              We use this profile to personalize scanner rankings, portfolio guidance, and the AI advisor's strategy language.
            </p>
          </div>
          <div className="min-w-[220px] rounded-[1.6rem] border border-white/10 bg-black/20 p-5">
            <p className="text-[0.65rem] uppercase tracking-[0.25em] text-frost/70">Progress</p>
            <p className="mt-3 text-3xl font-semibold text-white">{step + 1} / {steps.length}</p>
            <div className="mt-4 h-2 rounded-full bg-white/8">
              <div
                className="h-full rounded-full bg-gradient-to-r from-pulse via-frost to-ember transition-all"
                style={{ width: `${((step + 1) / steps.length) * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="mt-10 rounded-[2rem] border border-white/10 bg-white/[0.035] p-6 sm:p-8">
          <p className="text-[0.65rem] uppercase tracking-[0.32em] text-mist/55">Step {step + 1}</p>
          <h2 className="mt-3 text-3xl font-semibold text-white">{steps[step].title}</h2>
          <p className="mt-3 max-w-3xl text-base leading-7 text-mist/72">{steps[step].description}</p>
          <div className="mt-8">{steps[step].content}</div>
        </div>

        {authState.error ? <p className="mt-6 rounded-2xl border border-ember/30 bg-ember/10 px-4 py-3 text-sm text-ember">{authState.error}</p> : null}

        <div className="mt-8 flex flex-wrap items-center justify-between gap-4">
          <button
            type="button"
            onClick={() => setStep((current) => Math.max(current - 1, 0))}
            className="rounded-full border border-white/12 px-5 py-3 text-sm font-semibold text-mist/78 transition hover:bg-white/6 disabled:cursor-not-allowed disabled:opacity-40"
            disabled={step === 0 || authState.loading}
          >
            Back
          </button>
          <button
            type="button"
            onClick={handleNext}
            className="rounded-full bg-pulse px-6 py-3 text-sm font-semibold text-ink transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-70"
            disabled={authState.loading || !form.interests.length}
          >
            {authState.loading ? "Saving..." : isLastStep ? "Finish Setup" : "Continue"}
          </button>
        </div>
      </section>
    </main>
  );
}

function SummaryCard({ label, value }) {
  return (
    <div className="rounded-[1.45rem] border border-white/10 bg-black/20 px-5 py-4">
      <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/55">{label}</p>
      <p className="mt-3 text-lg font-semibold capitalize text-white">{value}</p>
    </div>
  );
}
