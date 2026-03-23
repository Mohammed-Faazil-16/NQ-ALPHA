import axios from "axios";

export const AUTH_EXPIRED_EVENT = "neuroquant:auth-expired";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 10000,
});

const CHAT_API = axios.create({
  baseURL: "http://127.0.0.1:8000",
  timeout: 45000,
});

const attachToken = (config) => {
  const token = window.localStorage.getItem("neuroquant_token");
  if (token) {
    config.headers = config.headers || {};
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
};

const notifyAuthExpired = (error) => {
  if (typeof window === "undefined") return;
  if (error?.response?.status !== 401) return;
  if (!error?.config?.headers?.Authorization) return;
  window.dispatchEvent(new CustomEvent(AUTH_EXPIRED_EVENT, { detail: error?.response?.data }));
};

const handleResponseError = (error) => {
  notifyAuthExpired(error);
  return Promise.reject(error);
};

API.interceptors.request.use(attachToken);
CHAT_API.interceptors.request.use(attachToken);
API.interceptors.response.use((response) => response, handleResponseError);
CHAT_API.interceptors.response.use((response) => response, handleResponseError);

const unwrap = (promise) => promise.then((response) => response.data);

export const registerUser = (payload) => unwrap(API.post("/auth/register", payload));
export const loginUser = (payload) => unwrap(API.post("/auth/login", payload));
export const getUserProfile = () => unwrap(API.get("/user/profile"));
export const updateUserProfile = (payload) => unwrap(API.post("/user/profile", payload));

export const getPortfolioAllocation = () => unwrap(API.get("/portfolio/allocation"));
export const savePortfolioAllocation = (payload) => unwrap(API.post("/portfolio/allocation", payload));
export const getFinancialPlan = () => unwrap(API.get("/financial-plan/current"));
export const getSystemGuide = (password) => unwrap(API.post("/system/guide", { password }));

export const requestRecommendation = (query) => unwrap(API.post("/recommend", { query }));
export const getPrice = (symbol) => unwrap(API.get("/price", { params: { symbol } }));
export const getFeatures = (symbol) => unwrap(API.get("/features", { params: { symbol } }));
export const getAlphaSeries = (symbol, lookback = 30) =>
  unwrap(API.get("/alpha_series", { params: { symbol, lookback } }));
export const getAssetNews = (symbol, limit = 3) => unwrap(API.get("/news", { params: { symbol, limit } }));
export const searchAssets = (q, limit = 8, assetType = "") =>
  unwrap(API.get("/assets/search", { params: { q, limit, asset_type: assetType || undefined } }));
export const getScan = (params = {}) => unwrap(API.get("/scan", { params }));
export const postBacktest = (payload) => unwrap(API.post("/backtest", payload));
export const postChat = (payload) => unwrap(CHAT_API.post("/chat", payload));
export const getAdvisorStatus = () => unwrap(API.get("/advisor/status"));
