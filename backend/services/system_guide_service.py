from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func

from agents.alpha_agent.dataset_builder import (
    FEATURE_COLUMNS,
    GLOBAL_CONTEXT_FEATURES,
    REGIME_TO_ID,
    SEQUENCE_LENGTH,
    SPLIT_DATE,
    STORED_FEATURE_COLUMNS,
    TARGET_HORIZON,
    get_active_features as get_training_active_features,
    get_feature_groups,
)
from backend.database.models.all_assets import AllAssets
from backend.database.models.asset_universe import AssetUniverse
from backend.db.models import FeaturesLatest, FinancialPlan, MemoryMessage, Portfolio, User
from backend.db.postgres import SessionLocal
from backend.services.model_registry import DEFAULT_ACTIVE_FEATURES, get_active_features as get_model_active_features

SYSTEM_GUIDE_PASSWORD = os.getenv("SYSTEM_GUIDE_PASSWORD", "Marthakimaaka@007")
PAPER_TITLE = "NQ Alpha: A Regime-Aware AI Quant Investment Decision System"
RESEARCH_PAPER_TITLE = "NeuroQuant: A Reinforcement Learning Framework with Stochastic Optimal Transport for Adaptive Portfolio Management Under Market Regime Shifts"
BASE_PAPER_TITLE = "Applications of Markov Decision Process Model and Deep Learning in Quantitative Portfolio Management during the COVID-19 Pandemic"
REFERENCE_FORMAT_NOTE = (
    "Paper view follows the structural cadence of the Agri-Mantra reference: Abstract, Keywords, Introduction, Literature Survey, Existing System Challenges, Proposed System Advantages, Motivation and Goal, Architecture Diagram, Module Explanation with Implementation, Results and Discussion, Conclusion, and References. The content itself remains exclusive to NQ ALPHA."
)


def validate_system_guide_password(password: str) -> bool:
    return str(password or "") == SYSTEM_GUIDE_PASSWORD


def _safe_count(query_fn) -> int:
    try:
        return int(query_fn())
    except Exception:
        return 0


def _collect_runtime_counts() -> dict[str, Any]:
    db = SessionLocal()
    try:
        counts = {
            "all_assets_count": _safe_count(lambda: db.query(AllAssets).count()),
            "selected_universe_count": _safe_count(lambda: db.query(AssetUniverse).count()),
            "features_latest_count": _safe_count(lambda: db.query(FeaturesLatest).count()),
            "user_count": _safe_count(lambda: db.query(User).count()),
            "portfolio_count": _safe_count(lambda: db.query(Portfolio).count()),
            "plan_count": _safe_count(lambda: db.query(FinancialPlan).count()),
            "memory_count": _safe_count(lambda: db.query(MemoryMessage).count()),
        }
        asset_type_counts = Counter()
        try:
            for asset_type, count in db.query(AllAssets.asset_type, func.count(AllAssets.id)).group_by(AllAssets.asset_type).all():
                asset_type_counts[str(asset_type or "stock").lower()] = int(count)
        except Exception:
            pass
        counts["asset_type_counts"] = dict(asset_type_counts)
        return counts
    finally:
        db.close()


def _alpha_math() -> list[dict[str, str]]:
    rows = [
        (
            "Forward return target",
            "r_(i,t->t+H) = close_(i,t+H) / close_(i,t) - 1",
            "The research target begins with a future H-step return for asset i, which ties learning to realized forward performance instead of to a hand-written signal label.",
        ),
        (
            "Cross-sectional rank target",
            "y_(i,t) = rank(r_(i,t->t+H)) / N",
            "The model is trained on relative rank across assets at each timestamp, so alpha is naturally interpreted as a relative opportunity score rather than as a direct price forecast.",
        ),
        (
            "Temporal smoothing",
            "x'_t = 0.7 * x_t + 0.3 * x_(t-1)",
            "Feature smoothing reduces one-bar noise and improves stability across walk-forward splits.",
        ),
        (
            "Target smoothing",
            "y'_t = 0.8 * y_t + 0.2 * y_(t-1)",
            "The target is lightly smoothed so the ranking function is learned more consistently across neighboring timestamps.",
        ),
        (
            "Feature normalization",
            "x~_(i,t,k) = clip((x_(i,t,k) - mu_k) / (sigma_k + epsilon), -c, c)",
            "Each feature channel is standardized and clipped so extreme market shocks do not destabilize training or live inference.",
        ),
        (
            "Feature projection",
            "h_t = W_2 * GELU(W_1 x~_t + b_1) + b_2",
            "A learned projection layer lifts each engineered feature vector into the hidden dimension used by the transformer.",
        ),
        (
            "Self-attention core",
            "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V",
            "Self-attention allows the model to compare all timesteps in the 30-step window simultaneously instead of relying only on local lag structure.",
        ),
        (
            "Temporal encoder",
            "z_1, ..., z_T = TransformerEncoder(h_1 + PE_1, ..., h_T + PE_T)",
            "Positional encoding preserves order while the encoder learns which historical interactions matter most for forward rank performance.",
        ),
        (
            "Regime fusion",
            "g = [z_T ; E(r_t)]",
            "The final temporal state is fused with a learned regime embedding so identical price patterns can be judged differently in bull, normal, volatile, and crisis states.",
        ),
        (
            "Alpha head",
            "alpha = w^T Dropout(GELU(W g + b)) + c",
            "A compact head converts the regime-aware temporal representation into a scalar alpha value used everywhere else in the platform.",
        ),
        (
            "Decision thresholds",
            "BUY if alpha > 0.02; HOLD if -0.02 <= alpha <= 0.02; AVOID if alpha < -0.02",
            "The recommendation layer is deterministic so the analyzer and advisor cannot silently disagree.",
        ),
        (
            "Confidence mapping",
            "confidence = min(20 * |alpha|, 1.0)",
            "Confidence is a bounded conviction score derived from alpha magnitude. It is useful for ranking strength, not as a calibrated probability.",
        ),
        (
            "Portfolio weights",
            "w_i = a_i / C_total",
            "In the live product, rupee amounts a_i are converted into portfolio weights by dividing by the user's total capital base C_total.",
        ),
        (
            "Portfolio return",
            "r_(p,t) = sum_i w_i * r_(i,t)",
            "Backtest return is the weighted sum of asset returns and becomes the basis for compounded wealth and risk metrics.",
        ),
        (
            "Compounded wealth",
            "V_t = V_(t-1) * (1 + r_(p,t))",
            "This produces the equity curve shown in Strategy Lab and makes multi-period performance path-dependent instead of terminal-only.",
        ),
        (
            "Annualized Sharpe",
            "Sharpe = sqrt(252) * mean(r_p) / std(r_p)",
            "Sharpe measures return efficiency per unit of realized volatility.",
        ),
        (
            "Drawdown",
            "DD_t = 1 - V_t / max_(s <= t) V_s",
            "Drawdown measures the depth of loss from a prior peak and is often the most emotionally meaningful risk statistic for users.",
        ),
    ]
    return [{"label": label, "expression": expression, "explanation": explanation} for label, expression, explanation in rows]

def _paper_architecture_blocks() -> dict[str, Any]:
    return {
        "caption": "Figure 1: High-level architecture of the NQ ALPHA investment decision system.",
        "description": "The architecture begins with user state and market context, resolves the requested asset, engineers regime-aware signals, scores alpha in the NQ ALPHA core, then routes that truth into portfolio strategy, advisor grounding, and final system outputs.",
        "blocks": [
            {
                "id": "input-sources",
                "title": "Input Sources",
                "accent": "blue",
                "grid": {"column": "1 / span 2", "row": "1 / span 3"},
                "items": ["Portfolio State", "User Profile", "Live + Historical", "User Query", "Macro Context"],
            },
            {
                "id": "advisor-memory",
                "title": "Advisor Memory",
                "accent": "pink",
                "grid": {"column": "7 / span 2", "row": "1 / span 2"},
                "items": ["User Memory", "Financial Plan Memory", "Portfolio Grounding", "Ollama Advisor", "Analyzer Alignment", "Explainable Output"],
            },
            {
                "id": "portfolio-strategy",
                "title": "Portfolio Strategy",
                "accent": "blue",
                "grid": {"column": "10 / span 2", "row": "1 / span 2"},
                "items": ["Portfolio Builder", "Future Projection", "Allocation Engine", "Risk Metrics", "Strategy Lab", "Backtest Engine"],
            },
            {
                "id": "symbol-intelligence",
                "title": "Symbol Intelligence",
                "accent": "violet",
                "grid": {"column": "1 / span 2", "row": "4 / span 2"},
                "items": ["Asset Discovery", "Symbol Resolver", "IN / US / Crypto", "Asset Registry"],
            },
            {
                "id": "feature-engine",
                "title": "Feature Engine",
                "accent": "green",
                "grid": {"column": "3 / span 2", "row": "3 / span 3"},
                "items": ["OHLCV Features", "Cross Sectional", "Momentum & Trend", "Volatility Signals", "Regime Features", "Normalization + Sequence"],
            },
            {
                "id": "regime-intelligence",
                "title": "Regime Intelligence",
                "accent": "gold",
                "grid": {"column": "5 / span 2", "row": "2 / span 2"},
                "items": ["Distribution Shift", "Regime Detection", "Volatility State", "Market Conditioning"],
            },
            {
                "id": "nq-alpha-core",
                "title": "NQ Alpha Core",
                "accent": "green",
                "grid": {"column": "7 / span 2", "row": "4 / span 3"},
                "items": ["Recommendation Engine", "Alpha Score", "Regime-Aware Inference", "Confidence Score", "Transformer Alpha"],
            },
            {
                "id": "system-outputs",
                "title": "System Outputs",
                "accent": "emerald",
                "grid": {"column": "10 / span 2", "row": "4 / span 2"},
                "items": ["Buy / Hold / Avoid", "Alpha Timeline", "Allocation View", "Risk Signals", "User Advice", "Dashboard"],
            },
        ],
        "links": [
            {"from": "input-sources", "to": "advisor-memory", "label": "profile data", "style": "dashed", "offsetY": -56},
            {"from": "input-sources", "to": "advisor-memory", "label": "user state", "style": "dashed-secondary", "offsetY": 48},
            {"from": "input-sources", "to": "symbol-intelligence", "label": "raw data", "style": "solid"},
            {"from": "symbol-intelligence", "to": "feature-engine", "label": "resolved symbols", "style": "solid"},
            {"from": "feature-engine", "to": "regime-intelligence", "label": "feature context", "style": "solid"},
            {"from": "feature-engine", "to": "nq-alpha-core", "label": "engineered features", "style": "solid"},
            {"from": "regime-intelligence", "to": "nq-alpha-core", "label": "regime context", "style": "solid"},
            {"from": "regime-intelligence", "to": "advisor-memory", "label": "alpha signal", "style": "solid"},
            {"from": "nq-alpha-core", "to": "portfolio-strategy", "label": "alpha signal", "style": "solid"},
            {"from": "nq-alpha-core", "to": "system-outputs", "label": "signal output", "style": "solid"},
            {"from": "advisor-memory", "to": "system-outputs", "label": "grounded advice", "style": "solid"},
            {"from": "portfolio-strategy", "to": "system-outputs", "label": "allocation logic", "style": "solid"},
            {"from": "portfolio-strategy", "to": "advisor-memory", "label": "strategy alignment", "style": "solid-secondary", "offsetY": 34},
        ],
    }


def _architecture() -> dict[str, Any]:
    stages = [
        ("ui", "01", "User Interfaces", "experience", "Dashboard, Scanner, Strategy Lab, Allocation, AI Advisor, and the protected guide collect user intent.", ["search terms", "portfolio edits", "advisor prompts"], ["validated UI actions"]),
        ("api", "02", "FastAPI Gateway", "orchestration", "Routes validate payloads, apply authentication, and orchestrate lightweight inference services.", ["HTTP requests", "JWT session"], ["service calls"]),
        ("symbol-data", "03", "Symbol + Data Intelligence", "data", "Natural language names are resolved into market symbols and then used for live OHLCV retrieval.", ["asset query"], ["clean symbol", "market data"]),
        ("features", "04", "Feature + Regime Layer", "features", "Raw price series become engineered features and a regime label, then get cached for fast reuse.", ["OHLCV"], ["feature tensor", "regime id"]),
        ("alpha", "05", "Alpha Transformer", "model", "A transformer encoder plus regime embedding outputs one scalar alpha score for the asset.", ["feature sequence", "regime"], ["alpha score"]),
        ("decision", "06", "Decision + Portfolio Layer", "decision", "Alpha is converted into BUY, HOLD, or AVOID and combined with portfolio state, backtest metrics, and future value projections.", ["alpha", "capital plan"], ["signal", "metrics"]),
        ("advisor", "07", "Memory + Advisor Layer", "advisor", "Portfolio truth, memory, saved plans, and Ollama reasoning are fused so explanations stay personalized and aligned.", ["signal", "memory", "portfolio state"], ["advisor response", "saved plan"]),
    ]
    return {
        "title": "NQ ALPHA architecture",
        "summary": "The platform is a chained decision system: user intent flows into FastAPI, then through symbol intelligence, data fetching, feature engineering, alpha inference, decision logic, portfolio analytics, and finally the advisor layer.",
        "recommended_mode": "animated",
        "static_mode_note": "Static mode is recommended for paper screenshots and figure export.",
        "paper_caption": "Figure 1: End-to-end architecture of NQ ALPHA from user interaction to aligned analyzer and advisor output.",
        "flows": [
            {"from": "ui", "to": "api", "label": "search, auth, and portfolio requests"},
            {"from": "api", "to": "symbol-data", "label": "symbol normalization and market fetch"},
            {"from": "symbol-data", "to": "features", "label": "OHLCV to engineered state"},
            {"from": "features", "to": "alpha", "label": "feature tensor plus regime"},
            {"from": "alpha", "to": "decision", "label": "alpha to actionable signal"},
            {"from": "decision", "to": "advisor", "label": "signal plus portfolio truth"},
        ],
        "stages": [
            {"id": sid, "step": step, "title": title, "type": stage_type, "description": description, "inputs": inputs, "outputs": outputs}
            for sid, step, title, stage_type, description, inputs, outputs in stages
        ],
        "paper_layout": _paper_architecture_blocks(),
    }

def _product_sections(counts: dict[str, Any], feature_groups: dict[str, list[str]], training_active_features: list[str], model_active_features: list[str], asset_type_text: str) -> list[dict[str, Any]]:
    math_entries = [{"label": item["label"], "detail": f"{item['expression']} | {item['explanation']}"} for item in _alpha_math()]
    return [
        {
            "id": "platform-overview",
            "title": "Platform Overview",
            "summary": "What the product does from raw market data to user-facing decisions.",
            "entries": [
                {"label": "What NQ ALPHA is", "detail": "NQ ALPHA is a live AI quant decision system that combines symbol resolution, live data retrieval, feature engineering, transformer alpha inference, recommendation logic, portfolio planning, backtesting, memory, authentication, and an Ollama advisor."},
                {"label": "Aligned backend flow", "detail": "Asset-specific advisor questions reuse the same analyzer path as the dashboard, which keeps recommendation, alpha, confidence, and regime aligned across the platform."},
                {"label": "Practical pipeline", "detail": "Resolve symbol -> fetch data -> build features -> infer alpha and regime -> map to recommendation -> combine with saved portfolio truth -> render UI and advisor output."},
            ],
        },
        {
            "id": "main-algorithms",
            "title": "Main Algorithms And Why They Exist",
            "summary": "The major algorithms in practical language.",
            "entries": [
                {"label": "Symbol resolver", "detail": "A layered resolver checks aliases, live assets, curated training assets, and discovery logic so users can type names like ICICI, Reliance, Apple, or BTC without knowing the ticker suffix."},
                {"label": "Feature engine", "detail": "Raw OHLCV becomes returns, momentum, volatility, mean-reversion, RSI, MACD-style signals, and regime-aware interaction features so the model sees normalized structure instead of naked prices."},
                {"label": "Regime detector", "detail": "A regime layer classifies bull, normal, volatile, or crisis states so the same momentum pattern can be interpreted differently under different market conditions."},
                {"label": "Alpha transformer", "detail": "The live alpha model is a transformer encoder with hidden size 96, 4 attention heads, 2 encoder layers, positional encoding, and a regime embedding because financial signals are temporal and path-dependent."},
                {"label": "Recommendation layer", "detail": "The final signal layer is deterministic: alpha is thresholded into BUY, HOLD, or AVOID and scaled into a bounded confidence score. This keeps the decision engine explainable."},
                {"label": "Portfolio layer", "detail": "The user-facing portfolio engine is rupee-first. It converts planned rupee amounts into weights, keeps unused capital in cash, and computes backtest metrics and future value bands."},
                {"label": "Hybrid advisor", "detail": "Money questions are answered from saved state, asset questions are answered from the analyzer, and broader planning questions go to Ollama. This reduces hallucination and contradiction."},
            ],
        },
        {
            "id": "alpha-engine",
            "title": "Alpha Model Construction",
            "summary": "What alpha means and what the model is built from.",
            "entries": [
                {"label": "Meaning of alpha", "detail": "Alpha is a relative opportunity score. Positive alpha means the asset is expected to be stronger than peers over the target horizon. It is not an exact future price target."},
                {"label": "Model class", "detail": "The deployed class is NeuroQuantAlphaModel in transformer mode with positional encoding, a 4-state regime embedding, and a small multilayer alpha head with dropout 0.15."},
                {"label": "Why a transformer", "detail": "A transformer can compare all timesteps in the 30-step window through self-attention, which is more expressive than a flat table model or a single hand-coded momentum rule."},
                {"label": "Active deployment features", "detail": f"The live deployment uses {len(model_active_features)} selected inference features: {', '.join(model_active_features)}."},
                {"label": "Training-active features", "detail": f"The persisted training-active set is: {', '.join(training_active_features)}."},
                {"label": "Full feature universe", "detail": f"The broader research pipeline contains {len(FEATURE_COLUMNS)} engineered features. Stored base feature columns are: {', '.join(STORED_FEATURE_COLUMNS)}. Feature groups are: " + "; ".join(f"{group}: {', '.join(values)}" for group, values in feature_groups.items())},
                {"label": "Target design", "detail": f"The training target is rank-based future performance over a {TARGET_HORIZON}-step horizon, which makes the model naturally suited to cross-sectional selection."},
            ],
        },
        {"id": "alpha-mathematics", "title": "Alpha Mathematics", "summary": "Core mathematical blocks behind the alpha engine and decision layer.", "entries": math_entries},
        {
            "id": "portfolio-layer",
            "title": "Portfolio Layer And Metrics",
            "summary": "What Strategy Lab and Allocation are really measuring.",
            "entries": [
                {"label": "Portfolio meaning", "detail": "Portfolio means the saved user-specific capital deployment plan: asset amounts, remaining cash, lookback choice, and the resulting historical performance path."},
                {"label": "Return", "detail": "Return is the total compounded gain or loss over the selected historical window."},
                {"label": "Sharpe", "detail": "Sharpe is annualized reward per unit of volatility, so it measures efficiency rather than raw upside alone."},
                {"label": "Drawdown", "detail": "Drawdown is the deepest fall from a previous peak and is often the most emotionally meaningful risk metric for real users."},
                {"label": "Volatility", "detail": "Volatility is the annualized standard deviation of returns and captures how violently the portfolio path moves."},
                {"label": "Baseline", "detail": "Baseline is the comparison portfolio used to judge whether the chosen allocation improved on a simpler reference path."},
                {"label": "Future value projection", "detail": "Strategy Lab converts annualized return into horizon-aware value and profit bands for 1-2 years, 3-5 years, or 5+ years."},
            ],
        },
        {
            "id": "asset-access",
            "title": "Training Universe vs Accessible Universe",
            "summary": "How much the model was trained on, and how much the live system can analyze now.",
            "entries": [
                {"label": "Curated training universe", "detail": f"The curated training universe currently contains {counts['selected_universe_count']} assets in asset_universe."},
                {"label": "Live accessible universe", "detail": f"The live system currently has {counts['all_assets_count']} assets in all_assets. Asset type mix right now: {asset_type_text}."},
                {"label": "Operational reach", "detail": "The live platform can analyze Indian stocks, US stocks, crypto pairs, and other reachable assets if the symbol layer and provider can resolve them correctly."},
                {"label": "Modeling caution", "detail": "Operational access is broader than the strict training universe, so the strongest confidence remains closest to the training distribution even when the system can technically analyze farther markets."},
                {"label": "Global context", "detail": f"The broader research builder exposes global context features such as: {', '.join(GLOBAL_CONTEXT_FEATURES)}."},
            ],
        },
        {
            "id": "advisor-memory",
            "title": "Advisor, Memory, And User State",
            "summary": "How the system stays personalized without losing numerical truth.",
            "entries": [
                {"label": "State separation", "detail": f"The database currently contains {counts['user_count']} users, {counts['portfolio_count']} portfolio records, {counts['plan_count']} saved financial plans, and {counts['memory_count']} text memory records. These are intentionally stored separately."},
                {"label": "Exact money answers", "detail": "Budget, invested amount, and remaining cash are answered from saved portfolio state instead of from the language model."},
                {"label": "Consistent asset answers", "detail": "Asset-specific advisor queries call the shared analyzer path first, so the advisor inherits the same recommendation, alpha, regime, and confidence as the dashboard."},
                {"label": "Where memory lives", "detail": "Profiles live in users, allocations in portfolios, plan summaries in financial_plans, text history in memory_messages, and vector memory in ChromaDB."},
            ],
        },
        {
            "id": "research-positioning",
            "title": "Research Positioning And Paper Comparison",
            "summary": "How the current platform relates to your existing paper and the base reference paper.",
            "entries": [
                {"label": "Your paper", "detail": f"Your paper, {RESEARCH_PAPER_TITLE}, emphasizes reinforcement learning, stochastic optimal transport, and regime-aware policy shifting."},
                {"label": "Base paper", "detail": f"The base paper, {BASE_PAPER_TITLE}, focuses on SSDAE and LSTM-AE feature extraction with A2C and an omega-ratio objective."},
                {"label": "What is newer here", "detail": "The current production platform adds a transformer alpha model, fast inference cache, live symbol discovery, rupee-first portfolio planning, per-user saved state, and an advisor constrained by hard portfolio and analyzer truth."},
                {"label": "Academic caution", "detail": "The platform is more integrated and more practical than either paper alone, but full research superiority still requires matched benchmarking on controlled datasets."},
            ],
        },
        {
            "id": "future-plan",
            "title": "Future Plan",
            "summary": "Where NQ ALPHA can become stronger next.",
            "entries": [
                {"label": "Engineering roadmap", "detail": "Expand exchange-aware data coverage, keep improving freshness metadata, and make symbol discovery even more robust for Indian and global assets."},
                {"label": "Modeling roadmap", "detail": "Retrain on a broader India-plus-global universe, improve macro conditioning, and benchmark transformer alpha against simpler baselines and policy-based layers."},
                {"label": "Research roadmap", "detail": "Reconnect more tightly to the original research thesis by evaluating stronger regime-transition logic, transport-based state comparison, and matched experiments against the reference papers."},
            ],
        },
    ]


def _paper_material(counts: dict[str, Any], training_active_features: list[str], model_active_features: list[str]) -> dict[str, Any]:
    asset_type_text = ", ".join(f"{name}: {count}" for name, count in counts["asset_type_counts"].items()) or "No grouped asset types reported"
    math = _alpha_math()
    return {
        "title": PAPER_TITLE,
        "subtitle": "A publication-style technical narrative generated from the current NQ ALPHA codebase, model configuration, and live platform design.",
        "reference_format": REFERENCE_FORMAT_NOTE,
        "abstract": "NQ ALPHA is a production-grade AI quant investment decision system that unifies natural-language symbol resolution, live market data retrieval, transformer-based alpha inference, regime-aware contextualization, rupee-first portfolio construction, backtesting, persistent user state, and a constrained local-LLM advisor. The system is designed to close the gap between quantitative research prototypes and deployable decision products by ensuring that analysis, allocation, and explanation all inherit the same underlying signal logic.",
        "keywords": ["quantitative finance", "alpha modeling", "transformer encoder", "regime-aware inference", "portfolio analytics", "financial AI advisor", "decision systems"],
        "figure_caption": "Figure 1: High-level system architecture of the NQ ALPHA platform.",
        "sections": [
            {
                "id": "introduction",
                "heading": "1. Introduction",
                "paragraphs": [
                    "Many financial AI systems become either research-only prototypes or polished interfaces with weak quantitative grounding. NQ ALPHA is designed to bridge those two worlds by turning signal extraction, allocation, and explanation into one coherent system.",
                    "The platform converts natural asset queries into symbols, live features, alpha scores, deterministic recommendations, portfolio analytics, and advisor output inside one aligned backend stack.",
                ],
                "bullets": ["Live analyzer, scanner, strategy lab, allocation, and advisor", "Transformer-based alpha engine with regime conditioning", "Rupee-first portfolio planning", "Shared logic between analyzer and advisor"],
                "equations": [],
            },
            {
                "id": "literature-survey",
                "heading": "2. Literature Survey",
                "paragraphs": [
                    f"Your paper, {RESEARCH_PAPER_TITLE}, emphasizes reinforcement learning, stochastic optimal transport, and regime-aware portfolio adaptation. That work is valuable because it frames portfolio management as a sequential decision problem.",
                    f"The base paper, {BASE_PAPER_TITLE}, demonstrates the importance of representation learning and policy design under stressed market conditions. NQ ALPHA diverges from that stack by using a transformer-based alpha layer for live analysis and by placing stronger emphasis on deployable alignment, explainability, and user-state persistence.",
                    "The resulting system therefore sits at the intersection of sequence modeling, portfolio analytics, and production decision support rather than belonging to only one of those categories.",
                ],
                "bullets": ["Builds on regime-aware quantitative finance research", "Extends beyond isolated model papers into a deployable system", "Treats explanation and portfolio truth as first-class design goals"],
                "equations": [],
            },
            {
                "id": "existing-challenges",
                "heading": "3. Existing System Challenges",
                "paragraphs": [
                    "Financial AI products often suffer from fragmentation: symbol search, model inference, portfolio planning, and advisor explanation are implemented as disconnected layers. This leads to contradictory decisions, stale context, and user distrust.",
                    "A second challenge is inference architecture. Training pipelines are commonly too heavy for live APIs, while a third challenge is usability because end users think in names and money amounts rather than in suffix-heavy tickers and abstract decimal weights.",
                ],
                "bullets": ["Symbol ambiguity across exchanges and regions", "Slow live inference when training code leaks into APIs", "Hallucination risk for money questions", "Weak interpretability of alpha and metrics"],
                "equations": [],
            },
            {
                "id": "proposed-advantages",
                "heading": "4. Proposed System Advantages",
                "paragraphs": [
                    "NQ ALPHA addresses those issues through a modular but tightly aligned design. Asset-specific advisor queries reuse the same analyzer path as the dashboard, which prevents silent recommendation conflicts.",
                    "The platform also translates user capital directly into rupee-based planning, stores user-specific allocation and memory state, and separates heavy research pipelines from lightweight inference services.",
                ],
                "bullets": ["Fast live inference with cached features", "Advisor constrained by analyzer and portfolio truth", "Rupee-first planning and future value bands", "Protected guide for transparent system communication"],
                "equations": [],
            },
            {
                "id": "motivation",
                "heading": "5. Motivation and Goal",
                "paragraphs": [
                    "The central motivation of NQ ALPHA is to convert exclusive quant research into a usable investment decision product without discarding mathematical rigor. The system is therefore designed as an AI quant decision platform rather than as a chart viewer or a generic chatbot.",
                    "Its goal is to let a user search an asset, inspect the model's belief, understand the regime, simulate allocation, and ask follow-up advisor questions without leaving one internally consistent stack.",
                ],
                "bullets": ["Bring research-grade logic into a practical product", "Keep explanation and action tightly coupled", "Express allocation in real money terms", "Support both product use and paper communication"],
                "equations": [],
            },
            {
                "id": "architecture-diagram",
                "heading": "6. Architecture Diagram",
                "paragraphs": [
                    "The overall architecture of NQ ALPHA is shown in Figure 1. The design begins with a data layer that receives market data and user portfolio inputs, transforms them into processed features, passes them into the NQ ALPHA core for alpha inference and decision logic, then branches into regime intelligence, intelligence modules, and user-facing outputs.",
                    "This figure is designed to be publication-friendly while still mapping directly to the real implementation used by the web application.",
                ],
                "bullets": ["Data Layer -> Feature Engineering -> NQ ALPHA Core", "Regime Intelligence and Intelligence Modules enrich the decision path", "Outputs and Visualization translate internal state into user-facing action"],
                "equations": [],
            },
            {
                "id": "module-explanation",
                "heading": "7. Module Explanation with Implementation",
                "paragraphs": [
                    "This section explains the functional modules that make NQ ALPHA operate as a production system. Each module is independently meaningful, but the system's main strength appears when the modules operate together under one aligned logic stack.",
                ],
                "bullets": [],
                "equations": [],
                "subsections": [
                    {
                        "id": "symbol-data",
                        "heading": "7.1 Symbol Intelligence and Data Layer",
                        "paragraphs": [
                            "The platform first resolves natural input such as Apple, ICICI, HDFC, or BTC into a tradable symbol using aliases, searchable asset tables, fuzzy logic, and provider-aware fallbacks. Once resolved, the data layer retrieves live OHLCV history for feature construction.",
                            f"At runtime, the curated training universe contains {counts['selected_universe_count']} assets while the broader live all_assets table contains {counts['all_assets_count']} reachable assets. The current asset-type mix is {asset_type_text}.",
                        ],
                        "bullets": ["Natural-language asset search", "Exchange-aware and alias-aware resolution", "Live market retrieval", "Persistent growth of accessible universe"],
                        "equations": [],
                    },
                    {
                        "id": "feature-regime",
                        "heading": "7.2 Feature Engineering and Regime Modeling",
                        "paragraphs": [
                            f"The feature layer builds from a research universe of {len(FEATURE_COLUMNS)} engineered signals, with stored base features such as {', '.join(STORED_FEATURE_COLUMNS)}. It also constructs cross-asset, interaction, correlation, and regime-transition signals so the model sees context, not just the raw asset path.",
                            f"Global context channels such as {', '.join(GLOBAL_CONTEXT_FEATURES)} help the model understand market breadth and stress, while regime states in {{{', '.join(REGIME_TO_ID.keys())}}} condition downstream interpretation.",
                        ],
                        "bullets": ["Temporal smoothing for stability", "Cross-sectional rank-aware signals", "Global breadth and context channels", "Discrete regime conditioning"],
                        "equations": math[:5],
                    },
                    {
                        "id": "alpha-model",
                        "heading": "7.3 Transformer-Based Alpha Model",
                        "paragraphs": [
                            f"The live alpha model consumes a {SEQUENCE_LENGTH}-step window of selected features. The deployment currently uses {len(model_active_features)} active inference features, while the persisted training-active set contains {len(training_active_features)} features.",
                            "A transformer is used because important financial relationships can exist between non-adjacent timesteps, such as delayed reversals, volatility transitions, or regime-conditioned trend persistence. Self-attention is therefore more expressive than a flat rule stack or a purely local recurrent summary.",
                        ],
                        "bullets": ["Hidden dimension: 96", "Attention heads: 4", "Encoder layers: 2", "Dropout: 0.15", "Rank-based future target"],
                        "equations": math[5:12],
                    },
                    {
                        "id": "portfolio-layer",
                        "heading": "7.4 Recommendation, Portfolio, and Backtest Layer",
                        "paragraphs": [
                            "The analyzer converts alpha into BUY, HOLD, or AVOID through a deterministic threshold rule. Strategy Lab allows the user to specify planned rupee allocations, which are converted into portfolio weights while unused money is preserved as cash.",
                            "Historical portfolio behavior is then measured using return, Sharpe, drawdown, volatility, and baseline comparison. This makes the allocation layer both practical for the user and interpretable in quantitative terms.",
                        ],
                        "bullets": ["Deterministic recommendation mapping", "Rupee-first planning", "Backtest metrics and future value projections", "Allocation page for capital, invested amount, and cash reserve"],
                        "equations": math[12:],
                    },
                    {
                        "id": "advisor-memory",
                        "heading": "7.5 Advisor, Memory, and User State",
                        "paragraphs": [
                            f"NQ ALPHA stores state explicitly. There are currently {counts['user_count']} users, {counts['portfolio_count']} portfolio rows, {counts['plan_count']} saved financial plans, and {counts['memory_count']} memory rows in the relational layer, alongside vector memory in ChromaDB.",
                            "The advisor behaves as a constrained reasoning layer: money questions are answered from saved portfolio state, asset-specific questions inherit the analyzer view, and only broader planning language is delegated to Ollama.",
                        ],
                        "bullets": ["Per-user profiles, plans, and memory", "State-backed answers for money questions", "Analyzer-backed answers for asset questions", "LLM used as a reasoning layer, not as financial truth"],
                        "equations": [],
                    },
                ],
            },
            {
                "id": "results",
                "heading": "8. Results and Discussion",
                "paragraphs": [
                    "The current system demonstrates a strong end-to-end alignment between analyzer logic, portfolio state, and advisor explanation. This is an important systems result because many financial AI products fail once their modules begin contradicting one another.",
                    "The platform also shows that a broader operational asset universe can coexist with a more curated training universe, provided the system communicates that distinction honestly and uses the advisor as a constrained reasoning layer rather than as a free-form prediction engine.",
                ],
                "bullets": [],
                "equations": [],
                "subsections": [
                    {
                        "id": "results-ops",
                        "heading": "8.1 Operational Findings",
                        "paragraphs": [
                            f"The live platform currently reports {counts['features_latest_count']} cached feature snapshots, {counts['portfolio_count']} stored portfolios, and {counts['plan_count']} saved plan records. These counts demonstrate that the system is already maintaining live state rather than operating as a purely illustrative prototype.",
                            "The most practical systems result is analyzer-advisor agreement: asset questions are now routed through the shared analyzer path and money questions are routed through saved portfolio truth.",
                        ],
                        "bullets": ["Aligned analyzer and advisor decisions", "Persistent per-user state", "Separated heavy research pipeline from live inference"],
                        "equations": [],
                    },
                    {
                        "id": "results-research",
                        "heading": "8.2 Research Interpretation",
                        "paragraphs": [
                            f"Compared with {RESEARCH_PAPER_TITLE} and {BASE_PAPER_TITLE}, the present system is more integrated at the product layer because it combines signal extraction, allocation, and explanation inside one live environment.",
                            "However, an academically rigorous superiority claim still requires matched benchmarking on common datasets and evaluation protocols. The honest contribution today is therefore a strong deployable systems contribution with a research-grade quantitative core.",
                        ],
                        "bullets": ["Stronger production integration than the reference systems", "Broader user-facing explainability", "Matched benchmarking remains future work"],
                        "equations": [],
                    },
                ],
            },
            {
                "id": "conclusion",
                "heading": "9. Conclusion",
                "paragraphs": [
                    "NQ ALPHA demonstrates that an AI quant product can be both explainable and structurally rigorous when the analyzer, allocation engine, and advisor are all built around one shared logic stack. The project is strongest when viewed not as a single model, but as a coordinated decision architecture.",
                    "Its contribution is therefore twofold: a transformer-based alpha layer with regime-aware conditioning, and a production-grade system that translates that signal into allocation, metrics, and explanation without losing internal consistency.",
                ],
                "bullets": ["Unified signal, allocation, and explanation stack", "Research-grade structure with product-grade execution", "Clear path toward broader retraining and benchmark studies"],
                "equations": [],
            },
        ],
        "references": [
            "[1] A. Mohammed Faazil, NeuroQuant: A Reinforcement Learning Framework with Stochastic Optimal Transport for Adaptive Portfolio Management Under Market Regime Shifts.",
            "[2] Han Yue, Jiapeng Liu, and Qin Zhang, Applications of Markov Decision Process Model and Deep Learning in Quantitative Portfolio Management during the COVID-19 Pandemic, Systems, 2022.",
            "[3] Harry Markowitz, Portfolio Selection, The Journal of Finance, 1952.",
            "[4] Ashish Vaswani et al., Attention Is All You Need, NeurIPS, 2017.",
            "[5] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, Layer Normalization, arXiv:1607.06450, 2016.",
            "[6] John Moody and Matthew Saffell, Learning to Trade via Direct Reinforcement, IEEE Transactions on Neural Networks, 2001.",
            "[7] Zhengyao Jiang, Dixing Xu, and Jinjun Liang, A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem, arXiv:1706.10059, 2017.",
            "[8] Xiao-Yang Liu et al., FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, 2020.",
            "[9] Matthew F. Dixon, Igor Halperin, and Paul Bilokon, Machine Learning in Finance: From Theory to Practice, 2020.",
            "[10] Marco Cuturi and Gabriel Peyre, Computational Optimal Transport, Foundations and Trends in Machine Learning, 2019.",
            "[11] James D. Hamilton, A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle, Econometrica, 1989.",
            "[12] Bryan Lim et al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting, International Journal of Forecasting, 2021.",
            "[13] John Hull, Risk Management and Financial Institutions, Wiley, 2018.",
            "[14] Contemporary quantitative trading and evaluation literature on return, drawdown, Sharpe, and regime-sensitive allocation benchmarking.",
            "[15] Internal NQ ALPHA engineering repository, protected system guide, live analyzer, portfolio layer, and advisor alignment stack documentation.",
        ],
    }

def build_system_guide_payload() -> dict[str, Any]:
    counts = _collect_runtime_counts()
    feature_groups = get_feature_groups()
    training_active_features = list(get_training_active_features())
    model_active_features = list(get_model_active_features())
    asset_type_text = ", ".join(f"{name}: {count}" for name, count in counts["asset_type_counts"].items()) or "No grouped asset types reported"
    return {
        "brand": "NQ ALPHA",
        "title": "NQ ALPHA System Guide",
        "subtitle": "A code-aligned explainer of the full platform, from alpha mathematics and portfolio logic to live asset coverage, memory, advisor alignment, and research positioning.",
        "password_protected": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "alpha": {
            "sequence_length": SEQUENCE_LENGTH,
            "training_active_features": training_active_features,
            "model_active_features": model_active_features,
            "feature_count_total": len(FEATURE_COLUMNS),
            "feature_count_active": len(model_active_features),
            "feature_projection_hidden_dim": 96,
            "attention_heads": 4,
            "encoder_layers": 2,
            "feedforward_dim": 192,
            "dropout": 0.15,
            "regime_states": list(REGIME_TO_ID.keys()),
            "target_horizon": TARGET_HORIZON,
            "default_active_features": DEFAULT_ACTIVE_FEATURES,
            "math": _alpha_math(),
            "feature_groups": feature_groups,
        },
        "coverage": {
            "training_universe_count": counts["selected_universe_count"],
            "accessible_universe_count": counts["all_assets_count"],
            "asset_types": counts["asset_type_counts"],
            "summary": f"Curated training universe: {counts['selected_universe_count']} assets. Live accessible universe: {counts['all_assets_count']} assets. Asset type mix: {asset_type_text}.",
        },
        "papers": {
            "paper_title": PAPER_TITLE,
            "research_paper": RESEARCH_PAPER_TITLE,
            "base_paper": BASE_PAPER_TITLE,
            "reference_format": REFERENCE_FORMAT_NOTE,
        },
        "architecture": _architecture(),
        "paper_material": _paper_material(counts, training_active_features, model_active_features),
        "sections": _product_sections(counts, feature_groups, training_active_features, model_active_features, asset_type_text),
    }
