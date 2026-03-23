import { usePlatform } from "../context/PlatformContext";

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard" },
  { id: "scanner", label: "Scanner" },
  { id: "strategy", label: "Strategy Lab" },
  { id: "allocation", label: "Allocation" },
  { id: "advisor", label: "AI Advisor" },
  { id: "guide", label: "System Guide" },
];

export default function TopNav() {
  const { activePage, setActivePage, session, logoutUser } = usePlatform();
  const userName = session.user?.full_name || session.user?.email?.split("@", 1)[0] || "Investor";

  return (
    <nav className="sticky top-4 z-20 mb-8">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 rounded-[1.75rem] border border-white/12 bg-ink/72 px-4 py-3 shadow-glow backdrop-blur-2xl sm:px-6">
        <div>
          <p className="text-[0.65rem] uppercase tracking-[0.35em] text-frost/70">NQ ALPHA</p>
          <h1 className="mt-1 text-xl font-semibold text-white">NQ ALPHA Platform</h1>
        </div>
        <div className="flex flex-1 flex-wrap justify-center gap-2">
          {NAV_ITEMS.map((item) => {
            const active = item.id === activePage;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActivePage(item.id)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  active
                    ? "bg-pulse text-ink shadow-[0_12px_35px_rgba(109,240,194,0.35)]"
                    : "bg-white/5 text-mist hover:bg-white/10"
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-3 rounded-full border border-white/10 bg-black/20 px-3 py-2">
          <div className="hidden text-right sm:block">
            <p className="text-sm font-semibold text-white">{userName}</p>
            <p className="text-[0.65rem] uppercase tracking-[0.22em] text-mist/52">{session.user?.risk_level || "balanced"}</p>
          </div>
          <button
            type="button"
            onClick={logoutUser}
            className="rounded-full border border-white/10 px-3 py-2 text-xs font-semibold uppercase tracking-[0.18em] text-mist/72 transition hover:bg-white/6 hover:text-white"
          >
            Logout
          </button>
        </div>
      </div>
    </nav>
  );
}
