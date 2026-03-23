export default function Panel({ title, eyebrow, action, className = "", children }) {
  return (
    <section className={`panel p-5 sm:p-6 ${className}`.trim()}>
      {(title || eyebrow || action) && (
        <header className="mb-5 flex items-start justify-between gap-4">
          <div>
            {eyebrow ? <p className="text-[0.65rem] uppercase tracking-[0.35em] text-frost/70">{eyebrow}</p> : null}
            {title ? <h2 className="panel-title mt-2">{title}</h2> : null}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}
