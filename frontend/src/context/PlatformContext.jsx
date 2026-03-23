import { createContext, useContext, useEffect, useMemo, useState } from "react";
import {
  AUTH_EXPIRED_EVENT,
  getAdvisorStatus,
  getAlphaSeries,
  getFeatures,
  getFinancialPlan,
  getPortfolioAllocation,
  getPrice,
  getScan,
  getUserProfile,
  loginUser,
  postBacktest,
  postChat,
  registerUser,
  requestRecommendation,
  savePortfolioAllocation,
  updateUserProfile,
} from "../api/api";

const PlatformContext = createContext(null);
const TOKEN_KEY = "neuroquant_token";
const USER_KEY = "neuroquant_user";

const defaultStrategyAssets = (selectedAsset) => [
  { symbol: selectedAsset || "AAPL", weight: 0.4 },
  { symbol: "MSFT", weight: 0.35 },
  { symbol: "NVDA", weight: 0.25 },
];

function readStoredUser() {
  try {
    const raw = window.localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function getErrorMessage(error, fallback, unauthorizedFallback = "Your session expired. Please sign in again to sync your saved portfolio and plans.") {
  if (error?.response?.status === 401) {
    return unauthorizedFallback;
  }
  return error?.response?.data?.detail || error?.message || fallback;
}

export function PlatformProvider({ children }) {
  const initialToken = window.localStorage.getItem(TOKEN_KEY) || "";
  const initialUser = readStoredUser();

  const [activePage, setActivePage] = useState("dashboard");
  const [selectedAsset, setSelectedAsset] = useState("AAPL");
  const [authReady, setAuthReady] = useState(!initialToken);
  const [session, setSession] = useState({
    token: initialToken,
    user: initialUser,
  });
  const [authState, setAuthState] = useState({ loading: false, error: "" });
  const [dashboardData, setDashboardData] = useState({
    loading: false,
    error: "",
    recommendation: null,
    priceData: [],
    priceAsOf: "",
    features: null,
    regime: "normal",
    regimeId: 1,
    alphaSeries: [],
    alphaAsOf: "",
  });
  const [scannerState, setScannerState] = useState({
    loading: false,
    error: "",
    filters: { top_n: 20, asset_type: "" },
    results: [],
    selected: null,
    partial: false,
    evaluated: 0,
    elapsedSeconds: 0,
  });
  const [strategyState, setStrategyState] = useState({
    loading: false,
    error: "",
    results: null,
  });
  const [allocationState, setAllocationState] = useState({
    loading: false,
    error: "",
    data: null,
  });
  const [financialPlanState, setFinancialPlanState] = useState({
    loading: false,
    error: "",
    data: null,
  });
  const [advisorState, setAdvisorState] = useState({
    loading: false,
    error: "",
    response: "",
    source: "",
    model: "",
    latencySeconds: 0,
    status: null,
  });

  const persistSession = (token, user) => {
    if (token) {
      window.localStorage.setItem(TOKEN_KEY, token);
    } else {
      window.localStorage.removeItem(TOKEN_KEY);
    }
    if (user) {
      window.localStorage.setItem(USER_KEY, JSON.stringify(user));
    } else {
      window.localStorage.removeItem(USER_KEY);
    }
    setSession({ token: token || "", user: user || null });
  };

  const expireSession = (message = "Your session expired. Please sign in again.") => {
    window.localStorage.removeItem(TOKEN_KEY);
    window.localStorage.removeItem(USER_KEY);
    setSession({ token: "", user: null });
    setAuthReady(true);
    setAuthState({ loading: false, error: message });
    setAllocationState({ loading: false, error: "", data: null });
    setFinancialPlanState({ loading: false, error: "", data: null });
    setStrategyState((current) => ({ ...current, error: "" }));
    setAdvisorState({ loading: false, error: "", response: "", source: "", model: "", latencySeconds: 0, status: null });
    setActivePage("dashboard");
  };

  const refreshProfile = async () => {
    if (!session.token) return null;
    const user = await getUserProfile();
    persistSession(session.token, user);
    return user;
  };

  const refreshAdvisorStatus = async () => {
    try {
      const status = await getAdvisorStatus();
      setAdvisorState((current) => ({ ...current, status }));
      return status;
    } catch (error) {
      setAdvisorState((current) => ({
        ...current,
        status: {
          connected: false,
          configured_model: "unknown",
          using_live_model: false,
          error: error?.response?.data?.detail || error?.message || "Unable to reach advisor runtime.",
        },
      }));
      return null;
    }
  };

  const loadFinancialPlan = async () => {
    if (!session.token) return null;
    setFinancialPlanState((current) => ({ ...current, loading: true, error: "" }));
    try {
      const payload = await getFinancialPlan();
      setFinancialPlanState({ loading: false, error: "", data: payload || null });
      return payload || null;
    } catch (error) {
      setFinancialPlanState({
        loading: false,
        error: getErrorMessage(error, "Unable to load the saved financial plan."),
        data: null,
      });
      throw error;
    }
  };

  const registerUserAction = async (payload) => {
    setAuthState({ loading: true, error: "" });
    try {
      const response = await registerUser(payload);
      persistSession(response.access_token, response.user);
      setAuthReady(true);
      setAuthState({ loading: false, error: "" });
      return response.user;
    } catch (error) {
      setAuthState({
        loading: false,
        error: getErrorMessage(error, "Unable to create your account.", "Please sign in again."),
      });
      throw error;
    }
  };

  const loginUserAction = async (payload) => {
    setAuthState({ loading: true, error: "" });
    try {
      const response = await loginUser(payload);
      persistSession(response.access_token, response.user);
      setAuthReady(true);
      setAuthState({ loading: false, error: "" });
      return response.user;
    } catch (error) {
      setAuthState({
        loading: false,
        error: getErrorMessage(error, "Unable to sign in.", "Please sign in again."),
      });
      throw error;
    }
  };

  const saveProfile = async (profile) => {
    setAuthState({ loading: true, error: "" });
    try {
      const user = await updateUserProfile(profile);
      persistSession(session.token, user);
      setAuthState({ loading: false, error: "" });
      return user;
    } catch (error) {
      setAuthState({
        loading: false,
        error: getErrorMessage(error, "Unable to save your profile."),
      });
      throw error;
    }
  };

  const loadAllocation = async () => {
    if (!session.token) return null;
    setAllocationState((current) => ({ ...current, loading: true, error: "" }));
    try {
      const payload = await getPortfolioAllocation();
      setAllocationState({ loading: false, error: "", data: payload });
      return payload;
    } catch (error) {
      setAllocationState({
        loading: false,
        error: getErrorMessage(error, "Unable to load allocation."),
        data: null,
      });
      throw error;
    }
  };

  const saveAllocationPlan = async ({ assets, lookbackDays }) => {
    if (!session.token) return null;
    try {
      const payload = await savePortfolioAllocation({ assets, lookback_days: lookbackDays });
      setAllocationState({ loading: false, error: "", data: payload });
      return payload;
    } catch (error) {
      setAllocationState((current) => ({
        ...current,
        error: getErrorMessage(error, "Unable to save allocation."),
      }));
      throw error;
    }
  };

  const logoutUser = () => {
    expireSession("");
  };

  const loadAssetAnalysis = async (query) => {
    setDashboardData((current) => ({ ...current, loading: true, error: "" }));
    try {
      const recommendation = await requestRecommendation(query);
      const symbol = recommendation.symbol;
      setSelectedAsset(symbol);

      const [pricePayload, featurePayload, alphaPayload] = await Promise.all([
        getPrice(symbol),
        getFeatures(symbol),
        getAlphaSeries(symbol),
      ]);

      setDashboardData({
        loading: false,
        error: "",
        recommendation,
        priceData: pricePayload.data ?? [],
        priceAsOf: pricePayload.latest_timestamp ?? "",
        features: featurePayload.features ?? null,
        regime: featurePayload.regime ?? "normal",
        regimeId: featurePayload.regime_id ?? 1,
        alphaSeries: alphaPayload.series ?? [],
        alphaAsOf: alphaPayload.as_of ?? "",
      });
      return symbol;
    } catch (error) {
      setDashboardData((current) => ({
        ...current,
        loading: false,
        error: error?.response?.data?.detail || error?.message || "Unable to load live analysis.",
      }));
      throw error;
    }
  };

  const openAsset = async (symbol) => {
    setActivePage("dashboard");
    return loadAssetAnalysis(symbol);
  };

  const loadScanner = async (overrides = {}) => {
    const nextFilters = { ...scannerState.filters, ...overrides };
    setScannerState((current) => ({ ...current, loading: true, error: "", filters: nextFilters }));
    try {
      const payload = await getScan({
        top_n: nextFilters.top_n,
        asset_type: nextFilters.asset_type || undefined,
      });
      const results = Array.isArray(payload) ? payload : payload?.results ?? [];
      setScannerState((current) => ({
        ...current,
        loading: false,
        error: "",
        filters: nextFilters,
        results,
        selected: results[0] ?? null,
        partial: Boolean(payload?.partial),
        evaluated: Number(payload?.evaluated ?? results.length),
        elapsedSeconds: Number(payload?.elapsed_seconds ?? 0),
      }));
      return payload;
    } catch (error) {
      setScannerState((current) => ({
        ...current,
        loading: false,
        error: error?.response?.data?.detail || error?.message || "Unable to scan the market.",
        partial: false,
        evaluated: 0,
        elapsedSeconds: 0,
      }));
      throw error;
    }
  };

  const selectScannerAsset = (asset) => {
    setScannerState((current) => ({ ...current, selected: asset }));
  };

  const runStrategyBacktest = async ({ assets, lookbackDays, capital = 0, investmentHorizon = "" }) => {
    setStrategyState((current) => ({ ...current, loading: true, error: "" }));
    try {
      const results = await postBacktest({
        assets,
        lookback_days: lookbackDays,
        capital,
        investment_horizon: investmentHorizon || undefined,
      });
      setStrategyState({ loading: false, error: "", results });
      return results;
    } catch (error) {
      setStrategyState((current) => ({
        ...current,
        loading: false,
        error: error?.response?.data?.detail || error?.message || "Unable to run backtest.",
      }));
      throw error;
    }
  };

  const askAdvisor = async (message) => {
    if (!session.user?.id) {
      const error = new Error("Please sign in first.");
      setAdvisorState((current) => ({ ...current, loading: false, error: error.message }));
      throw error;
    }

    setAdvisorState((current) => ({ ...current, loading: true, error: "" }));
    try {
      const response = await postChat({ user_id: session.user.id, message });
      setAdvisorState((current) => ({
        ...current,
        loading: false,
        error: "",
        response: response.response || "",
        source: response.source || "unknown",
        model: response.model || "",
        latencySeconds: Number(response.latency_seconds || 0),
      }));
      if (response.plan) {
        setFinancialPlanState({ loading: false, error: "", data: response.plan });
      }
      return response;
    } catch (error) {
      setAdvisorState((current) => ({
        ...current,
        loading: false,
        error: error?.response?.data?.detail || error?.message || "Advisor request failed.",
      }));
      throw error;
    }
  };

  useEffect(() => {
    const handleExpired = () => {
      expireSession("Your session expired. Please sign in again to keep your portfolio, allocation, and advisor state synced.");
    };
    window.addEventListener(AUTH_EXPIRED_EVENT, handleExpired);
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, handleExpired);
  }, []);

  useEffect(() => {
    let cancelled = false;

    if (!session.token) {
      setAuthReady(true);
      return () => {
        cancelled = true;
      };
    }

    setAuthReady(false);
    refreshProfile()
      .then(() => {
        if (!cancelled) {
          setAuthReady(true);
        }
      })
      .catch(() => {
        if (!cancelled) {
          expireSession("Your session expired. Please sign in again to continue.");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [session.token]);

  useEffect(() => {
    if (!authReady || !session.user?.onboarding_complete) return;
    loadAssetAnalysis(selectedAsset).catch(() => {});
    loadAllocation().catch(() => {});
    loadFinancialPlan().catch(() => {});
    refreshAdvisorStatus().catch(() => {});
  }, [authReady, session.user?.onboarding_complete]);

  const value = useMemo(
    () => ({
      activePage,
      setActivePage,
      selectedAsset,
      setSelectedAsset,
      session,
      authReady,
      isAuthenticated: Boolean(authReady && session.token && session.user?.id),
      needsOnboarding: Boolean(authReady && session.token && session.user && !session.user.onboarding_complete),
      authState,
      registerUserAction,
      loginUserAction,
      refreshProfile,
      saveProfile,
      logoutUser,
      dashboardData,
      scannerState,
      strategyState,
      allocationState,
      financialPlanState,
      advisorState,
      loadAssetAnalysis,
      openAsset,
      loadScanner,
      selectScannerAsset,
      runStrategyBacktest,
      loadAllocation,
      saveAllocationPlan,
      loadFinancialPlan,
      refreshAdvisorStatus,
      askAdvisor,
      defaultStrategyAssets,
    }),
    [activePage, selectedAsset, session, authReady, authState, dashboardData, scannerState, strategyState, allocationState, financialPlanState, advisorState],
  );

  return <PlatformContext.Provider value={value}>{children}</PlatformContext.Provider>;
}

export function usePlatform() {
  const context = useContext(PlatformContext);
  if (!context) {
    throw new Error("usePlatform must be used within PlatformProvider");
  }
  return context;
}
