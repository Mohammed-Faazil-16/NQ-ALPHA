import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import Panel from "../components/Panel";
import { getSystemGuide } from "../api/api";

function CountCard({ label, value, tone = "text-white" }) {
  return (
    <div className="metric-chip">
      <p className="text-[0.65rem] uppercase tracking-[0.25em] text-mist/55">{label}</p>
      <p className={`mt-3 text-2xl font-semibold ${tone}`}>{value}</p>
    </div>
  );
}

function TogglePill({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
        active ? "bg-pulse text-ink shadow-[0_10px_28px_rgba(109,240,194,0.28)]" : "bg-white/5 text-mist hover:bg-white/10 hover:text-white"
      }`}
    >
      {children}
    </button>
  );
}

function EquationBlock({ item, paper = false }) {
  return (
    <article className={paper ? "paper-equation" : "rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4"}>
      <p className={paper ? "paper-equation-label" : "text-[0.7rem] uppercase tracking-[0.24em] text-frost/70"}>{item.label}</p>
      <div className={paper ? "paper-equation-expression" : "mt-3 rounded-2xl border border-white/8 bg-black/20 px-4 py-3 font-mono text-sm text-pulse"}>
        {item.expression}
      </div>
      <p className={paper ? "paper-equation-explanation" : "mt-3 text-sm leading-7 text-mist/74"}>{item.explanation}</p>
    </article>
  );
}

function ArchitectureFlow({ architecture, animated, paper = false }) {
  const stages = architecture?.stages || [];
  const flows = architecture?.flows || [];

  return (
    <div className={`architecture-flow ${animated ? "architecture-flow--animated" : ""} ${paper ? "architecture-flow--paper" : ""}`.trim()}>
      {stages.map((stage, index) => (
        <Fragment key={stage.id}>
          <article className={`architecture-stage architecture-stage--${stage.type} ${paper ? "architecture-stage--paper" : ""}`.trim()}>
            <div className="flex items-start justify-between gap-3">
              <span className="architecture-stage-step">{stage.step}</span>
              <span className={`architecture-stage-kind ${paper ? "architecture-stage-kind--paper" : ""}`.trim()}>{stage.type}</span>
            </div>
            <h3 className={paper ? "paper-architecture-title" : "mt-4 text-xl font-semibold text-white"}>{stage.title}</h3>
            <p className={paper ? "paper-architecture-copy" : "mt-3 text-sm leading-7 text-mist/74"}>{stage.description}</p>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <div className={paper ? "paper-mini-block" : "rounded-[1.2rem] border border-white/8 bg-black/20 px-4 py-3"}>
                <p className={paper ? "paper-mini-label" : "text-[0.65rem] uppercase tracking-[0.22em] text-mist/55"}>Inputs</p>
                <p className={paper ? "paper-mini-value" : "mt-2 text-sm leading-7 text-white"}>{(stage.inputs || []).join(" | ")}</p>
              </div>
              <div className={paper ? "paper-mini-block" : "rounded-[1.2rem] border border-white/8 bg-black/20 px-4 py-3"}>
                <p className={paper ? "paper-mini-label" : "text-[0.65rem] uppercase tracking-[0.22em] text-mist/55"}>Outputs</p>
                <p className={paper ? "paper-mini-value" : "mt-2 text-sm leading-7 text-white"}>{(stage.outputs || []).join(" | ")}</p>
              </div>
            </div>
          </article>
          {index < stages.length - 1 ? (
            <div className="architecture-link">
              <div className="architecture-link-rail">{animated ? <span className="architecture-link-dot" /> : null}</div>
              <p className={`architecture-link-label ${paper ? "architecture-link-label--paper" : ""}`.trim()}>{flows[index]?.label || "data handoff"}</p>
            </div>
          ) : null}
        </Fragment>
      ))}
    </div>
  );
}

const ACCENT_CLASS = {
  blue: "paper-block--blue",
  pink: "paper-block--pink",
  green: "paper-block--green",
  gold: "paper-block--gold",
  indigo: "paper-block--indigo",
  emerald: "paper-block--emerald",
  violet: "paper-block--violet",
};

function PaperArchitectureBoard({ architecture, animated }) {
  const layout = architecture?.paper_layout;
  const blocks = layout?.blocks || [];
  const links = layout?.links || [];
  const containerRef = useRef(null);
  const blockRefs = useRef({});
  const [paths, setPaths] = useState([]);

  useEffect(() => {
    if (!containerRef.current || !blocks.length) return undefined;

    const measure = () => {
      const container = containerRef.current;
      if (!container) return;
      const containerRect = container.getBoundingClientRect();
      const rects = {};

      blocks.forEach((block) => {
        const node = blockRefs.current[block.id];
        if (!node) return;
        const rect = node.getBoundingClientRect();
        rects[block.id] = {
          left: rect.left - containerRect.left,
          top: rect.top - containerRect.top,
          right: rect.right - containerRect.left,
          bottom: rect.bottom - containerRect.top,
          width: rect.width,
          height: rect.height,
          centerX: rect.left - containerRect.left + rect.width / 2,
          centerY: rect.top - containerRect.top + rect.height / 2,
        };
      });

      const nextPaths = links.flatMap((link, index) => {
        const source = rects[link.from];
        const target = rects[link.to];
        if (!source || !target) return [];

        let start = { x: source.right, y: source.centerY };
        let end = { x: target.left, y: target.centerY };

        if (Math.abs(target.centerX - source.centerX) < 60) {
          start = { x: source.centerX, y: source.bottom };
          end = { x: target.centerX, y: target.top };
        } else if (target.centerX < source.centerX) {
          start = { x: source.left, y: source.centerY };
          end = { x: target.right, y: target.centerY };
        }

        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const offsetY = link.offsetY || 0;
        const offsetX = link.offsetX || 0;
        const controlX = Math.max(Math.abs(dx) * 0.4, 28);
        const controlY = Math.max(Math.abs(dy) * 0.25, 20);
        const path = Math.abs(dx) >= Math.abs(dy)
          ? `M ${start.x} ${start.y} C ${start.x + Math.sign(dx || 1) * controlX} ${start.y + offsetY}, ${end.x - Math.sign(dx || 1) * controlX} ${end.y + offsetY}, ${end.x} ${end.y}`
          : `M ${start.x} ${start.y} C ${start.x + offsetX} ${start.y + Math.sign(dy || 1) * controlY}, ${end.x + offsetX} ${end.y - Math.sign(dy || 1) * controlY}, ${end.x} ${end.y}`;

        return [{
          id: `${link.from}-${link.to}-${index}`,
          path,
          label: link.label,
          style: link.style || "solid",
          labelX: (start.x + end.x) / 2,
          labelY: (start.y + end.y) / 2 - 10 + offsetY * 0.32,
        }];
      });

      setPaths(nextPaths);
    };

    const frame = requestAnimationFrame(measure);
    window.addEventListener("resize", measure);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", measure);
    };
  }, [blocks, links]);

  return (
    <figure className={`paper-architecture-board ${animated ? "paper-architecture-board--animated" : ""}`.trim()}>
      <div className="paper-architecture-canvas" ref={containerRef}>
        <svg className="paper-architecture-svg" viewBox="0 0 1000 520" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <marker id="paperArrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(15, 23, 42, 0.45)" />
            </marker>
          </defs>
          {paths.map((item) => (
            <g key={item.id}>
              <path d={item.path} className={`paper-architecture-path paper-architecture-path--${item.style || "solid"} ${animated ? "paper-architecture-path--animated" : ""}`.trim()} markerEnd="url(#paperArrow)" />
              {animated ? (
                <circle r="5" className="paper-architecture-node-dot">
                  <animateMotion dur="3.8s" repeatCount="indefinite" path={item.path} />
                </circle>
              ) : null}
              <text x={item.labelX} y={item.labelY} textAnchor="middle" className="paper-architecture-link-text">{item.label}</text>
            </g>
          ))}
        </svg>
        {blocks.map((block) => (
          <article
            key={block.id}
            ref={(node) => {
              if (node) blockRefs.current[block.id] = node;
            }}
            className={`paper-architecture-block ${ACCENT_CLASS[block.accent] || "paper-block--blue"}`}
            style={{ gridColumn: block.grid?.column, gridRow: block.grid?.row }}
          >
            <div className="paper-architecture-block-tag">{block.title}</div>
            <ul className="paper-architecture-items">
              {(block.items || []).map((item) => (
                <li key={`${block.id}-${item}`}>{item}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>
      <figcaption className="paper-figure-caption">{layout?.caption}</figcaption>
      <p className="paper-figure-copy">{layout?.description}</p>
    </figure>
  );
}

function PaperSubsection({ subsection }) {
  return (
    <section className="paper-subsection">
      <h4 className="paper-subsection-title">{subsection.heading}</h4>
      <div className="space-y-4">
        {(subsection.paragraphs || []).map((paragraph) => (
          <p key={paragraph} className="paper-paragraph">{paragraph}</p>
        ))}
        {(subsection.bullets || []).length ? (
          <ul className="paper-list">
            {subsection.bullets.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : null}
        {(subsection.equations || []).length ? (
          <div className="space-y-4">
            {subsection.equations.map((equation) => (
              <EquationBlock key={`${subsection.id}-${equation.label}`} item={equation} paper />
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function PaperSection({ section, children }) {
  return (
    <section className="paper-section">
      <div className="paper-section-header">
        <h3 className="paper-section-title">{section.heading}</h3>
      </div>
      <div className="space-y-4">
        {(section.paragraphs || []).map((paragraph) => (
          <p key={paragraph} className="paper-paragraph">{paragraph}</p>
        ))}
        {(section.bullets || []).length ? (
          <ul className="paper-list">
            {section.bullets.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : null}
        {children}
        {(section.equations || []).length ? (
          <div className="space-y-4">
            {section.equations.map((equation) => (
              <EquationBlock key={`${section.id}-${equation.label}`} item={equation} paper />
            ))}
          </div>
        ) : null}
        {(section.subsections || []).length ? (
          <div className="space-y-5 pt-2">
            {section.subsections.map((subsection) => (
              <PaperSubsection key={subsection.id} subsection={subsection} />
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function ReferenceList({ references }) {
  return (
    <section className="paper-section">
      <div className="paper-section-header">
        <h3 className="paper-section-title">10. References</h3>
      </div>
      <ol className="paper-references">
        {(references || []).map((reference) => (
          <li key={reference}>{reference}</li>
        ))}
      </ol>
    </section>
  );
}

export default function SystemGuide() {
  const [password, setPassword] = useState("");
  const [guide, setGuide] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState("product");
  const [architectureMode, setArchitectureMode] = useState("animated");
  const [openSections, setOpenSections] = useState({
    "platform-overview": true,
    "alpha-engine": true,
  });

  const counts = guide?.counts || {};
  const quickFacts = useMemo(() => {
    if (!guide) return [];
    return [
      { label: "Assets reachable now", value: `${counts.all_assets_count || 0}`, tone: "text-pulse" },
      { label: "Training universe", value: `${counts.selected_universe_count || 0}`, tone: "text-frost" },
      { label: "Alpha features live", value: `${guide.alpha?.feature_count_active || 0}`, tone: "text-white" },
      { label: "Saved user plans", value: `${counts.plan_count || 0}`, tone: "text-ember" },
    ];
  }, [guide, counts]);

  const submitPassword = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const payload = await getSystemGuide(password);
      setGuide(payload);
      setOpenSections((current) => ({
        ...current,
        [payload.sections?.[0]?.id || "platform-overview"]: true,
      }));
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || requestError?.message || "Unable to unlock the system guide.");
    } finally {
      setLoading(false);
    }
  };
  const toggleSection = (sectionId) => {
    setOpenSections((current) => ({
      ...current,
      [sectionId]: !current[sectionId],
    }));
  };

  const animated = architectureMode === "animated";
  const paper = guide?.paper_material;
  const math = guide?.alpha?.math || [];

  return (
    <main className="space-y-6">
      <section className="panel p-6 sm:p-8">
        <p className="text-[0.7rem] uppercase tracking-[0.35em] text-frost/70">System Guide</p>
        <h2 className="mt-3 text-4xl font-semibold text-white">Everything behind NQ ALPHA, explained as both a product system and a paper-ready research story.</h2>
        <p className="mt-5 max-w-4xl text-base leading-8 text-mist/74">
          This guide now has two modes. Product View explains how the live system works page by page. Paper View turns the same codebase into a structured technical document with aligned research-paper sections, deeper alpha mathematics, architecture, training scope, and research positioning.
        </p>
      </section>

      {!guide ? (
        <section className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
          <Panel title="Unlock The Internal Guide" eyebrow="Protected Access">
            <form onSubmit={submitPassword} className="space-y-4">
              <p className="text-sm leading-7 text-mist/72">
                This view is intentionally gated because it exposes the internal model logic, mathematical formulation, architecture, storage layers, asset coverage, and paper-style documentation of the platform.
              </p>
              <label className="block space-y-2">
                <span className="text-[0.7rem] uppercase tracking-[0.2em] text-mist/55">Guide password</span>
                <input
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Enter the project guide password"
                  className="glass-input w-full px-4 py-3"
                />
              </label>
              <button
                type="submit"
                disabled={loading}
                className="rounded-full bg-pulse px-5 py-3 font-semibold text-ink transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {loading ? "Unlocking..." : "Unlock System Guide"}
              </button>
              {error ? <p className="rounded-2xl bg-ember/15 px-4 py-3 text-sm text-ember">{error}</p> : null}
            </form>
          </Panel>

          <Panel title="What This Guide Covers" eyebrow="Included Coverage">
            <div className="space-y-3 text-sm leading-7 text-mist/74">
              <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">Deep alpha explanation, transformer math, regimes, feature groups, and why each layer exists.</div>
              <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">A publication-style architecture figure with animated and static modes for product use or paper screenshots.</div>
              <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">A paper-style technical document that mirrors the section structure of the Agri-Mantra-style format while keeping NQ ALPHA's own content.</div>
              <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">Training scope, accessible assets, portfolio metrics, advisor logic, memory layers, and future roadmap.</div>
            </div>
          </Panel>
        </section>
      ) : null}

      {guide ? (
        <>
          <section className="grid gap-4 xl:grid-cols-4">
            {quickFacts.map((fact) => (
              <CountCard key={fact.label} label={fact.label} value={fact.value} tone={fact.tone} />
            ))}
          </section>

          <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <Panel title={guide.title} eyebrow="Guide Overview">
              <div className="space-y-4">
                <p className="text-sm leading-7 text-mist/74">{guide.subtitle}</p>
                <div className="flex flex-wrap gap-2">
                  <TogglePill active={viewMode === "product"} onClick={() => setViewMode("product")}>Product View</TogglePill>
                  <TogglePill active={viewMode === "paper"} onClick={() => setViewMode("paper")}>Paper View</TogglePill>
                </div>
                <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4 text-sm leading-7 text-mist/74">
                  <span className="font-semibold text-white">Coverage summary:</span> {guide.coverage?.summary}
                </div>
              </div>
            </Panel>

            <Panel title="Architecture Mode" eyebrow="Figure Controls">
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  <TogglePill active={architectureMode === "animated"} onClick={() => setArchitectureMode("animated")}>Animated</TogglePill>
                  <TogglePill active={architectureMode === "static"} onClick={() => setArchitectureMode("static")}>Static</TogglePill>
                </div>
                <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4 text-sm leading-7 text-mist/74">
                  {guide.architecture?.static_mode_note}
                </div>
                <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4 text-sm leading-7 text-mist/74">
                  <span className="font-semibold text-white">Updated:</span> {new Date(guide.updated_at).toLocaleString()}
                </div>
              </div>
            </Panel>
          </section>

          {viewMode === "product" ? (
            <>
              <section className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
                <Panel title={guide.architecture?.title} eyebrow="Live System Flow">
                  <p className="mb-5 text-sm leading-7 text-mist/74">{guide.architecture?.summary}</p>
                  <ArchitectureFlow architecture={guide.architecture} animated={animated} />
                </Panel>

                <Panel title="Alpha At A Glance" eyebrow="Model Snapshot">
                  <div className="space-y-4">
                    <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">
                      <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/55">Transformer setup</p>
                      <p className="mt-3 text-sm leading-7 text-white">
                        Hidden dim {guide.alpha?.feature_projection_hidden_dim}, {guide.alpha?.attention_heads} heads, {guide.alpha?.encoder_layers} encoder layers, dropout {guide.alpha?.dropout}.
                      </p>
                    </div>
                    <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">
                      <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/55">Regimes</p>
                      <p className="mt-3 text-sm leading-7 text-white">{(guide.alpha?.regime_states || []).join(", ")}</p>
                    </div>
                    <div className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">
                      <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/55">Active live features</p>
                      <p className="mt-3 text-sm leading-7 text-white">{(guide.alpha?.model_active_features || []).join(", ")}</p>
                    </div>
                  </div>
                </Panel>
              </section>

              <Panel title="Alpha Mathematics" eyebrow="Explain The Model">
                <div className="grid gap-4 xl:grid-cols-2">
                  {math.map((item) => (
                    <EquationBlock key={item.label} item={item} />
                  ))}
                </div>
              </Panel>

              <section className="space-y-4">
                {guide.sections?.map((section) => {
                  const isOpen = Boolean(openSections[section.id]);
                  return (
                    <section key={section.id} className="panel overflow-hidden p-0">
                      <button
                        type="button"
                        onClick={() => toggleSection(section.id)}
                        className="flex w-full items-center justify-between gap-4 px-6 py-5 text-left transition hover:bg-white/[0.03]"
                      >
                        <div>
                          <p className="text-[0.65rem] uppercase tracking-[0.3em] text-frost/70">Section</p>
                          <h3 className="mt-2 text-2xl font-semibold text-white">{section.title}</h3>
                          <p className="mt-3 max-w-4xl text-sm leading-7 text-mist/72">{section.summary}</p>
                        </div>
                        <div className="rounded-full border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-white">{isOpen ? "Hide" : "Open"}</div>
                      </button>
                      {isOpen ? (
                        <div className="border-t border-white/8 px-6 py-5">
                          <div className="grid gap-4 xl:grid-cols-2">
                            {section.entries?.map((entry) => (
                              <article key={`${section.id}-${entry.label}`} className="rounded-[1.35rem] border border-white/8 bg-white/5 px-4 py-4">
                                <p className="text-[0.7rem] uppercase tracking-[0.24em] text-mist/55">{entry.label}</p>
                                <p className="mt-3 text-sm leading-7 text-white">{entry.detail}</p>
                              </article>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </section>
                  );
                })}
              </section>
            </>
          ) : null}
          {viewMode === "paper" && paper ? (
            <section className="paper-surface">
              <div className="paper-topline">Research Paper Material</div>
              <h2 className="paper-title">{paper.title}</h2>
              <p className="paper-subtitle">{paper.subtitle}</p>
              <p className="paper-reference">{paper.reference_format}</p>

              <section className="paper-section paper-section--hero">
                <h3 className="paper-section-title">Abstract</h3>
                <p className="paper-paragraph">{paper.abstract}</p>
                <div className="paper-keywords">
                  <span className="paper-keywords-label">Keywords</span>
                  <div className="paper-keywords-list">
                    {(paper.keywords || []).map((keyword) => (
                      <span key={keyword} className="paper-keyword-pill">{keyword}</span>
                    ))}
                  </div>
                </div>
              </section>

              <div className="space-y-6">
                {(paper.sections || []).map((section) => (
                  <PaperSection key={section.id} section={section}>
                    {section.id === "architecture-diagram" ? <PaperArchitectureBoard architecture={guide.architecture} animated={animated} /> : null}
                  </PaperSection>
                ))}
                <ReferenceList references={paper.references} />
              </div>
            </section>
          ) : null}
        </>
      ) : null}
    </main>
  );
}
