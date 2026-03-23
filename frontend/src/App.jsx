import TopNav from "./components/TopNav";
import { PlatformProvider, usePlatform } from "./context/PlatformContext";
import Allocation from "./pages/Allocation";
import AuthPortal from "./pages/AuthPortal";
import Advisor from "./pages/Advisor";
import Dashboard from "./pages/Dashboard";
import Onboarding from "./pages/Onboarding";
import Scanner from "./pages/Scanner";
import StrategyLab from "./pages/StrategyLab";
import SystemGuide from "./pages/SystemGuide";

function AppShell() {
  const { activePage, authReady, isAuthenticated, needsOnboarding } = usePlatform();

  if (!authReady) {
    return (
      <div className="min-h-screen px-4 py-4 text-mist sm:px-6 lg:px-8">
        <div className="mx-auto max-w-4xl rounded-[2rem] border border-white/10 bg-white/5 px-8 py-16 text-center backdrop-blur-xl">
          <p className="text-[0.7rem] uppercase tracking-[0.35em] text-pulse">NQ ALPHA</p>
          <h1 className="mt-4 text-3xl font-semibold text-white">Restoring your secure session...</h1>
          <p className="mt-4 text-sm leading-7 text-mist/72">We are validating the saved login so your portfolio, allocation, and advisor memory stay in sync.</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <AuthPortal />;
  }

  if (needsOnboarding) {
    return <Onboarding />;
  }

  return (
    <div className="min-h-screen px-4 py-4 text-mist sm:px-6 lg:px-8">
      <TopNav />
      <div className="mx-auto max-w-7xl">
        {activePage === "dashboard" ? <Dashboard /> : null}
        {activePage === "scanner" ? <Scanner /> : null}
        {activePage === "strategy" ? <StrategyLab /> : null}
        {activePage === "allocation" ? <Allocation /> : null}
        {activePage === "advisor" ? <Advisor /> : null}
        {activePage === "guide" ? <SystemGuide /> : null}
      </div>
    </div>
  );
}

export default function App() {
  return (
    <PlatformProvider>
      <AppShell />
    </PlatformProvider>
  );
}
