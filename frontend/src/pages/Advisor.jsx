import { useMemo, useState } from "react";
import Panel from "../components/Panel";
import { usePlatform } from "../context/PlatformContext";

const QUICK_PROMPTS = [
  "Analyze Apple for my current profile",
  "Optimize my current portfolio mix",
  "Show the best opportunities for my risk level",
  "What should I improve in my strategy this week?",
];

function sanitizeHeadingLine(line) {
  return String(line || "").replace(/\*\*/g, "").replace(/__/g, "").trim();
}

function parseStructuredResponse(response) {
  const blocks = {};
  let currentKey = "summary";
  const lines = String(response || "").split(/\r?\n/).filter((line) => line.trim());

  for (const rawLine of lines) {
    const line = sanitizeHeadingLine(rawLine);
    const match = line.match(/^(Strategy|Risk Level|Allocation|Reasoning)(?:\s*:)?\s*(.*)$/i);
    if (match) {
      currentKey = match[1].toLowerCase().replace(/\s+/g, "_");
      blocks[currentKey] = match[2].trim();
    } else {
      blocks[currentKey] = `${blocks[currentKey] ? `${blocks[currentKey]} ` : ""}${line.trim()}`.trim();
    }
  }

  return blocks;
}

export default function Advisor() {
  const {
    advisorState,
    financialPlanState,
    askAdvisor,
    selectedAsset,
    session,
    refreshAdvisorStatus,
    loadFinancialPlan,
  } = usePlatform();
  const [message, setMessage] = useState(`Given ${selectedAsset}, what should I improve in my current strategy?`);
  const [messages, setMessages] = useState([]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!message.trim()) return;

    const userMessage = message.trim();
    setMessages((current) => [...current, { role: "user", text: userMessage }]);
    setMessage("");

    const response = await askAdvisor(userMessage).catch(() => null);
    if (response?.response) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: response.response,
          source: response.source,
          model: response.model,
          latency: response.latency_seconds,
        },
      ]);
    }
  };

  const latestStructured = useMemo(() => parseStructuredResponse(advisorState.response), [advisorState.response]);
  const currentPlan = financialPlanState.data;
  const liveModelConnected = Boolean(advisorState.status?.connected && advisorState.status?.using_live_model);

  return (
    <main className="space-y-6">
      <section className="panel p-6 sm:p-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[0.7rem] uppercase tracking-[0.35em] text-pulse">AI Advisor</p>
            <h2 className="mt-3 text-4xl font-semibold text-white">Talk to the advisor like a real investor, then verify the engine behind the answer.</h2>
            <p className="mt-4 max-w-3xl text-base leading-8 text-mist/72">
              The advisor uses your profile, separate memory, saved financial plan, and live Ollama runtime. You can see whether the answer came from the real model or a fallback.
            </p>
          </div>
          <button
            type="button"
            onClick={() => {
              refreshAdvisorStatus().catch(() => {});
              loadFinancialPlan().catch(() => {});
            }}
            className="rounded-full border border-white/12 bg-white/5 px-4 py-2 text-sm font-semibold text-mist transition hover:bg-white/10 hover:text-white"
          >
            Refresh Advisor Status
          </button>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-4">
        <StatusCard
          label="Runtime"
          value={liveModelConnected ? "Live Ollama" : "Fallback mode"}
          detail={advisorState.status?.configured_model || "unknown"}
          tone={liveModelConnected ? "text-pulse" : "text-ember"}
        />
        <StatusCard
          label="Latest source"
          value={advisorState.source || "No request yet"}
          detail={advisorState.model || "No model metadata yet"}
          tone={advisorState.source === "ollama" ? "text-pulse" : "text-frost"}
        />
        <StatusCard
          label="Latency"
          value={advisorState.latencySeconds ? `${advisorState.latencySeconds.toFixed(2)}s` : "-"}
          detail="Time spent on the latest advisor call"
          tone="text-white"
        />
        <StatusCard
          label="Saved plan"
          value={currentPlan?.strategy || "No saved plan yet"}
          detail={currentPlan?.updated_at ? `Updated ${currentPlan.updated_at}` : "A live advisor answer will store one."}
          tone="text-white"
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[0.92fr_1.08fr]">
        <Panel title="Ask the Advisor" eyebrow="Conversation Layer">
          <div className="mb-4 flex flex-wrap gap-2">
            {QUICK_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                onClick={() => setMessage(prompt)}
                className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold text-mist/75 transition hover:bg-white/10 hover:text-white"
              >
                {prompt}
              </button>
            ))}
          </div>
          <form onSubmit={handleSubmit} className="space-y-4">
            <textarea
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              rows={7}
              className="glass-input w-full px-4 py-4"
              placeholder={`Ask as ${session.user?.full_name || "your investor profile"}...`}
            />
            <div className="flex flex-wrap items-center gap-3">
              <button type="submit" className="rounded-full bg-frost px-5 py-3 font-semibold text-ink transition hover:brightness-105">
                Generate Advice
              </button>
              {advisorState.source ? (
                <span className={`rounded-full px-3 py-2 text-xs font-semibold uppercase tracking-[0.2em] ${advisorState.source === "ollama" ? "bg-pulse/20 text-pulse" : "bg-ember/16 text-ember"}`}>
                  {advisorState.source === "ollama" ? "Using live Ollama" : advisorState.source}
                </span>
              ) : null}
            </div>
            {advisorState.error ? <p className="rounded-2xl bg-ember/15 px-4 py-3 text-sm text-ember">{advisorState.error}</p> : null}
            {!liveModelConnected ? (
              <p className="rounded-2xl border border-ember/25 bg-ember/10 px-4 py-3 text-sm text-ember">
                Ollama is not reporting a live ready state right now. The system will fall back to the last saved financial plan instead of inventing advice.
              </p>
            ) : null}
          </form>
        </Panel>

        <Panel title="Advisor Response" eyebrow="Structured Output">
          {advisorState.loading ? <div className="skeleton h-80 rounded-[1.6rem]" /> : null}
          {!advisorState.loading && messages.length ? (
            <div className="space-y-4">
              {messages.map((entry, index) => (
                <div
                  key={`${entry.role}-${index}`}
                  className={`rounded-[1.5rem] px-4 py-4 ${entry.role === "user" ? "ml-6 bg-frost/12 text-white" : "mr-6 border border-white/8 bg-white/5 text-mist/84"}`}
                >
                  <div className="flex flex-wrap items-center gap-3">
                    <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/55">{entry.role === "user" ? "You" : "Advisor"}</p>
                    {entry.role === "assistant" && entry.source ? (
                      <span className={`rounded-full px-2.5 py-1 text-[0.62rem] font-semibold uppercase tracking-[0.16em] ${entry.source === "ollama" ? "bg-pulse/15 text-pulse" : "bg-ember/15 text-ember"}`}>
                        {entry.source}
                      </span>
                    ) : null}
                    {entry.role === "assistant" && entry.model ? (
                      <span className="text-[0.7rem] text-mist/55">{entry.model}</span>
                    ) : null}
                    {entry.role === "assistant" && entry.latency ? (
                      <span className="text-[0.7rem] text-mist/55">{Number(entry.latency).toFixed(2)}s</span>
                    ) : null}
                  </div>
                  <p className="mt-3 whitespace-pre-wrap text-sm leading-7">{entry.text}</p>
                </div>
              ))}
            </div>
          ) : !advisorState.loading ? (
            <p className="text-sm text-mist/70">Submit a question to pull memory, portfolio state, and a profile-aware recommendation into one place.</p>
          ) : null}
        </Panel>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="grid gap-6 xl:grid-cols-4">
          <StructuredCard label="Strategy" value={latestStructured.strategy || currentPlan?.strategy || "Waiting for advisor response"} />
          <StructuredCard label="Risk Level" value={latestStructured.risk_level || currentPlan?.risk_level || session.user?.risk_level || "balanced"} />
          <StructuredCard label="Allocation" value={latestStructured.allocation || currentPlan?.allocation_summary || "The portfolio engine will fill this after strategy refinement."} />
          <StructuredCard label="Reasoning" value={latestStructured.reasoning || currentPlan?.reasoning || latestStructured.summary || "Ask a question to get the full rationale."} />
        </div>

        <Panel title="Saved Financial Plan" eyebrow="Stored Separately Per User">
          {financialPlanState.loading ? <div className="skeleton h-56 rounded-[1.6rem]" /> : null}
          {!financialPlanState.loading && currentPlan ? (
            <div className="space-y-4">
              <PlanRow label="Strategy" value={currentPlan.strategy} />
              <PlanRow label="Risk level" value={currentPlan.risk_level} />
              <PlanRow label="Allocation" value={currentPlan.allocation_summary} />
              <PlanRow label="Reasoning" value={currentPlan.reasoning} />
              <PlanRow label="Source" value={`${currentPlan.source || "unknown"}${currentPlan.model ? ` / ${currentPlan.model}` : ""}`} />
            </div>
          ) : !financialPlanState.loading ? (
            <p className="text-sm text-mist/70">No financial plan has been saved yet. The first successful advisor response will create one for this user.</p>
          ) : null}
        </Panel>
      </section>
    </main>
  );
}

function StatusCard({ label, value, detail, tone }) {
  return (
    <div className="panel p-5">
      <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">{label}</p>
      <p className={`mt-4 text-xl font-semibold ${tone}`}>{value}</p>
      <p className="mt-2 text-sm leading-6 text-mist/62">{detail}</p>
    </div>
  );
}

function StructuredCard({ label, value }) {
  return (
    <div className="panel p-5">
      <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">{label}</p>
      <p className="mt-4 text-sm leading-7 text-mist/82">{value}</p>
    </div>
  );
}

function PlanRow({ label, value }) {
  return (
    <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">
      <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/55">{label}</p>
      <p className="mt-3 text-sm leading-7 text-mist/82">{value || "-"}</p>
    </div>
  );
}
