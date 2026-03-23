from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from backend.services.asset_intelligence_service import extract_asset_intelligence
from backend.services.financial_plan_service import (
    get_latest_financial_plan,
    plan_to_response_text,
    save_financial_plan,
)
from backend.services.forecast_service import build_asset_forecast, extract_forecast_horizon_years
from backend.services.llm_service import generate_advisor_response_with_metadata
from backend.services.memory_service import get_user_profile, retrieve_context
from backend.services.portfolio_service import build_portfolio_from_strategy, get_user_portfolio
from backend.services.runtime_cache import runtime_cache


ADVISOR_CACHE_TTL_SECONDS = 300
ADVISOR_TIMEOUT_SECONDS = 20
FORECAST_LLM_TIMEOUT_SECONDS = 6
MONEY_QUERY_HINTS = {
    "how much money",
    "how much do i have",
    "how much have i invested",
    "how much i invested",
    "how much currently to invest",
    "budget",
    "capital",
    "invested",
    "available cash",
    "remaining cash",
    "remaining amount",
    "remaining balance",
    "remaining funds",
    "cash left",
    "cash in bank",
    "amount i hold",
    "cash i hold",
    "to invest",
    "portfolio value",
}
ASSET_PRICE_QUERY_HINTS = {
    "current stock price",
    "current price",
    "price",
    "quote",
    "trading at",
    "stock price",
}
ASSET_FORECAST_QUERY_HINTS = {
    "predict",
    "predicted",
    "forecast",
    "future price",
    "price in",
    "price after",
    "3 year",
    "3 years",
    "2 year",
    "2 years",
    "5 year",
    "5 years",
    "target price",
    "what will",
}


def _currency_symbol(currency: str) -> str:
    normalized = str(currency or "").strip().upper()
    if normalized == "INR":
        return "₹"
    if normalized == "USD":
        return "$"
    return normalized or "$"


def _format_currency(value: float, currency: str) -> str:
    return f"{_currency_symbol(currency)}{value:,.0f}"


def _format_price(value: float, currency: str) -> str:
    return f"{_currency_symbol(currency)}{value:,.2f}"


def _asset_price_currency(symbol: str) -> str:
    normalized = str(symbol or "").upper()
    if normalized.endswith(".NS") or normalized.endswith(".BO"):
        return "INR"
    return "USD"


def _portfolio_money_summary(profile: dict[str, object], portfolio_data: dict[str, object]) -> str:
    currency = str(portfolio_data.get("currency") or "INR")

    capital_raw = portfolio_data.get("capital")
    invested_raw = portfolio_data.get("invested_amount")
    available_cash_raw = portfolio_data.get("available_cash_amount")

    capital = float(capital_raw if capital_raw is not None else profile.get("capital") or 0.0)
    invested_amount = float(invested_raw if invested_raw is not None else 0.0)
    available_cash_amount = float(
        available_cash_raw if available_cash_raw is not None else max(capital - invested_amount, 0.0)
    )
    allocations = portfolio_data.get("allocations") or []
    risk_level = str(profile.get("risk_level") or "balanced")

    if allocations:
        lines = [
            f"- {item['symbol']}: {_format_currency(float(item.get('amount') or 0.0), currency)} ({float(item.get('percent') or 0.0):.2f}%)"
            for item in allocations[:8]
        ]
        allocation_line = "\n".join(lines)
        top_symbols = ", ".join(item["symbol"] for item in allocations[:2])
        allocation_text = (
            f"Capital base: {_format_currency(capital, currency)}. Currently invested: {_format_currency(invested_amount, currency)}. "
            f"Still available as cash: {_format_currency(available_cash_amount, currency)}.\n{allocation_line}"
        )
        if available_cash_amount > 0.0:
            next_step = (
                f"You still have {_format_currency(available_cash_amount, currency)} free. The cleanest next move is to stage that cash into your strongest existing holdings, led by {top_symbols}, rather than spreading it too thin."
            )
        else:
            next_step = "Your portfolio is already fully deployed, so the next decision is rebalancing existing positions instead of adding fresh cash."
    else:
        allocation_text = (
            f"Capital base: {_format_currency(capital, currency)}. No specific asset allocation is saved yet. "
            f"Available to invest now: {_format_currency(available_cash_amount, currency)}."
        )
        next_step = "No saved allocation exists yet, so the next move is to build a first rupee allocation in Strategy Lab before the advisor suggests deployment changes."

    return (
        "Strategy: Use your saved allocation as the baseline and adjust only when a new plan is confirmed.\n"
        f"Risk Level: {risk_level}\n"
        f"Allocation: {allocation_text}\n"
        f"Reasoning: These numbers come directly from your saved NQ ALPHA portfolio state, so this answer is exact and not an AI estimate. {next_step}"
    )


def _is_money_query(user_query: str) -> bool:
    normalized = str(user_query or "").strip().lower()
    return any(hint in normalized for hint in MONEY_QUERY_HINTS)


def _is_asset_price_query(user_query: str) -> bool:
    normalized = str(user_query or "").strip().lower()
    return any(hint in normalized for hint in ASSET_PRICE_QUERY_HINTS)


def _is_asset_forecast_query(user_query: str) -> bool:
    normalized = str(user_query or "").strip().lower()
    return any(hint in normalized for hint in ASSET_FORECAST_QUERY_HINTS)


def _find_portfolio_position(portfolio_data: dict[str, object], symbol: str) -> dict[str, object] | None:
    for item in portfolio_data.get("allocations") or []:
        if str(item.get("symbol") or "").upper() == symbol.upper():
            return item
    return None


def _risk_budget_fraction(risk_level: str) -> float:
    normalized = str(risk_level or "balanced").strip().lower()
    if normalized == "aggressive":
        return 0.15
    if normalized == "conservative":
        return 0.05
    return 0.1


def _asset_specific_summary(profile: dict[str, object], portfolio_data: dict[str, object], asset_context: dict[str, object]) -> str:
    symbol = str(asset_context.get("symbol") or "").upper()
    asset_name = str(asset_context.get("asset_name") or symbol)
    recommendation = str(asset_context.get("recommendation") or "HOLD").upper()
    regime = str(asset_context.get("regime") or "normal")
    alpha = float(asset_context.get("alpha") or 0.0)
    confidence = float(asset_context.get("confidence") or 0.0)
    risk_level = str(profile.get("risk_level") or "balanced")
    currency = str(portfolio_data.get("currency") or "INR")
    price_currency = _asset_price_currency(symbol)
    capital = float(portfolio_data.get("capital") or profile.get("capital") or 0.0)
    available_cash = float(portfolio_data.get("available_cash_amount") or 0.0)
    latest_price = asset_context.get("latest_price")
    latest_timestamp = str(asset_context.get("latest_timestamp") or "")
    matched_query = str(asset_context.get("matched_query") or "")
    is_price_query = _is_asset_price_query(matched_query)
    position = _find_portfolio_position(portfolio_data, symbol)

    if position:
        current_position_text = (
            f"You already hold {_format_currency(float(position.get('amount') or 0.0), currency)} in {symbol} "
            f"({float(position.get('percent') or 0.0):.2f}% of capital)."
        )
    else:
        current_position_text = f"You do not currently have a saved position in {symbol}."

    latest_price_text = ""
    if latest_price is not None:
        latest_price_text = f" Latest observed close was {_format_price(float(latest_price), price_currency)} on {latest_timestamp}."

    if recommendation == "BUY":
        target_amount = min(available_cash, capital * _risk_budget_fraction(risk_level) * max(confidence, 0.5))
        if target_amount > 0.0:
            allocation = (
                f"System analyzer verdict for {asset_name} ({symbol}) is BUY with {confidence * 100:.0f}% confidence in a {regime} regime. "
                f"A sensible next deployment is {_format_currency(target_amount, currency)} from available cash, while keeping the rest unallocated until the signal is refreshed. {current_position_text}"
            )
        else:
            allocation = (
                f"System analyzer verdict for {asset_name} ({symbol}) is BUY with {confidence * 100:.0f}% confidence in a {regime} regime, but there is no free cash left. "
                f"If you want to act on it, rebalance from weaker holdings instead of forcing leverage. {current_position_text}"
            )
        strategy = f"Stay consistent with the analyzer and only add exposure to {symbol} while its signal remains BUY."
    elif recommendation == "AVOID":
        allocation = (
            f"System analyzer verdict for {asset_name} ({symbol}) is AVOID with {confidence * 100:.0f}% confidence in a {regime} regime. "
            f"Do not allocate fresh cash to this asset right now. {current_position_text}"
        )
        strategy = f"Do not add fresh exposure to {symbol} until the analyzer improves from AVOID to HOLD or BUY."
    else:
        allocation = (
            f"System analyzer verdict for {asset_name} ({symbol}) is HOLD with {confidence * 100:.0f}% confidence in a {regime} regime. "
            f"Keep it on watch or maintain the current size, but do not commit meaningful new capital yet. {current_position_text}"
        )
        strategy = f"Keep {symbol} as a monitored candidate and wait for a stronger analyzer signal before adding new money."

    if is_price_query and latest_price is not None:
        strategy = (
            f"{asset_name} ({symbol}) is currently trading around {_format_price(float(latest_price), price_currency)} based on the latest tracked close from {latest_timestamp}, "
            f"and the shared analyzer stance is {recommendation}."
        )

    reasoning = (
        f"This answer is locked to the same analyzer that powers the dashboard, so the advisor cannot contradict the live recommendation. "
        f"Current alpha is {alpha:.4f}, regime is {regime}, and confidence is {confidence * 100:.0f}%.{latest_price_text}"
    )

    return (
        f"Strategy: {strategy}\n"
        f"Risk Level: {risk_level}\n"
        f"Allocation: {allocation}\n"
        f"Reasoning: {reasoning}"
    )


def _build_forecast_prompt(
    profile: dict[str, object],
    portfolio_data: dict[str, object],
    asset_context: dict[str, object],
    forecast: dict[str, float | str],
    user_query: str,
) -> str:
    symbol = str(asset_context.get("symbol") or "").upper()
    asset_name = str(asset_context.get("asset_name") or symbol)
    recommendation = str(asset_context.get("recommendation") or "HOLD").upper()
    price_currency = _asset_price_currency(symbol)
    position = _find_portfolio_position(portfolio_data, symbol)
    position_text = (
        f"Current saved position: {float(position.get('percent') or 0.0):.2f}% of capital"
        if position
        else "Current saved position: none"
    )

    return f"""
SYSTEM:
You are NQ ALPHA's forecast explainer.
- Use only the facts provided below.
- Do not invent hidden catalysts or fake certainty.
- Give a scenario-based answer, not a guaranteed target.
- Keep the tone practical and investor-friendly.
- Use exactly these headings:
Current Price:
3-Year Base Case:
3-Year Range:
Take:
Reasoning:

FACTS:
Asset: {asset_name}
Symbol: {symbol}
Current price: {_format_price(float(forecast['current_price']), price_currency)}
Price as of: {forecast['current_timestamp']}
Analyzer recommendation: {recommendation}
Alpha score: {float(asset_context.get('alpha') or 0.0):.4f}
Confidence: {float(asset_context.get('confidence') or 0.0) * 100:.0f}%
Regime: {asset_context.get('regime')}
Trailing annualized return: {float(forecast['trailing_cagr']) * 100:.2f}%
Annualized volatility: {float(forecast['annualized_volatility']) * 100:.2f}%
Adjusted annual return used in forecast: {float(forecast['adjusted_annual_return']) * 100:.2f}%
Forecast horizon: {float(forecast['years']):.1f} years
Base-case projected price: {_format_price(float(forecast['base_price']), price_currency)}
Downside projected price: {_format_price(float(forecast['downside_price']), price_currency)}
Upside projected price: {_format_price(float(forecast['upside_price']), price_currency)}
User risk level: {profile.get('risk_level')}
User capital: {_format_currency(float(portfolio_data.get('capital') or profile.get('capital') or 0.0), str(portfolio_data.get('currency') or 'INR'))}
Available cash: {_format_currency(float(portfolio_data.get('available_cash_amount') or 0.0), str(portfolio_data.get('currency') or 'INR'))}
{position_text}

USER QUESTION:
{user_query}
""".strip()


def _deterministic_forecast_response(
    profile: dict[str, object],
    portfolio_data: dict[str, object],
    asset_context: dict[str, object],
    forecast: dict[str, float | str],
) -> str:
    symbol = str(asset_context.get("symbol") or "").upper()
    asset_name = str(asset_context.get("asset_name") or symbol)
    recommendation = str(asset_context.get("recommendation") or "HOLD").upper()
    regime = str(asset_context.get("regime") or "normal")
    confidence = float(asset_context.get("confidence") or 0.0)
    risk_level = str(profile.get("risk_level") or "balanced")
    currency = str(portfolio_data.get("currency") or "INR")
    price_currency = _asset_price_currency(symbol)
    available_cash = float(portfolio_data.get("available_cash_amount") or 0.0)
    target_fraction = _risk_budget_fraction(risk_level)
    staged_amount = min(available_cash, float(portfolio_data.get("capital") or profile.get("capital") or 0.0) * target_fraction)

    if recommendation == "BUY" and staged_amount > 0.0:
        take = f"The signal is constructive, so a staged entry of about {_format_currency(staged_amount, currency)} is reasonable instead of a full-size allocation."
    elif recommendation == "BUY":
        take = "The signal is constructive, but you do not have free cash right now, so any move would need a rebalance from weaker holdings."
    elif recommendation == "AVOID":
        take = "The model still flags this as an avoid, so the forecast should be treated as a scenario map rather than a buy trigger."
    else:
        take = "The model is neutral, so the forecast is more useful as a watchlist range than as an immediate buy signal."

    reasoning = (
        f"This 3-year view starts from the latest tracked close, then blends trailing return behavior, annualized volatility, current alpha, regime, and confidence into a bounded scenario range. "
        f"That keeps the answer grounded in the same engine that powers the analyzer instead of asking the LLM to invent a target."
    )

    return (
        f"Current Price: {asset_name} ({symbol}) is currently around {_format_price(float(forecast['current_price']), price_currency)} as of {forecast['current_timestamp']}.\n"
        f"3-Year Base Case: If the current return profile, analyzer signal, and {regime} regime stay broadly similar, the base-case path lands near {_format_price(float(forecast['base_price']), price_currency)} in about {float(forecast['years']):.1f} years.\n"
        f"3-Year Range: A practical downside-to-upside range is {_format_price(float(forecast['downside_price']), price_currency)} to {_format_price(float(forecast['upside_price']), price_currency)}.\n"
        f"Take: {take} Current analyzer stance is {recommendation} with {confidence * 100:.0f}% confidence and user risk level is {risk_level}.\n"
        f"Reasoning: {reasoning}"
    )


def generate_financial_advice(user_id: str, user_query: str) -> dict[str, object]:
    cache_key = ("advisor", user_id, user_query.strip().lower())
    cached = runtime_cache.get(cache_key)
    if cached is not None:
        return dict(cached)

    latest_plan = get_latest_financial_plan(user_id)

    try:
        memory = retrieve_context(user_id, user_query, top_k=5)
        profile = get_user_profile(user_id)
        last_strategy = (latest_plan or {}).get("strategy") if latest_plan else None
        portfolio_data = get_user_portfolio(user_id)

        if _is_money_query(user_query):
            payload = {
                "text": _portfolio_money_summary(profile, portfolio_data),
                "source": "portfolio-state",
                "model": None,
                "latency_seconds": 0.0,
                "plan": latest_plan,
            }
            runtime_cache.set(cache_key, payload, ttl_seconds=60)
            return payload

        try:
            asset_context = extract_asset_intelligence(user_query)
        except Exception:
            asset_context = None

        if asset_context is not None and _is_asset_forecast_query(user_query):
            try:
                forecast = build_asset_forecast(
                    symbol=str(asset_context.get("symbol") or ""),
                    years=extract_forecast_horizon_years(user_query),
                    alpha=float(asset_context.get("alpha") or 0.0),
                    regime=str(asset_context.get("regime") or "normal"),
                    confidence=float(asset_context.get("confidence") or 0.0),
                )
                deterministic_text = _deterministic_forecast_response(profile, portfolio_data, asset_context, forecast)
                prompt = _build_forecast_prompt(profile, portfolio_data, asset_context, forecast, user_query)
                result = generate_advisor_response_with_metadata(
                    prompt,
                    timeout_seconds=FORECAST_LLM_TIMEOUT_SECONDS,
                    forecast_mode=True,
                )
                if str(result.get("source") or "") == "ollama":
                    payload = {
                        "text": str(result.get("text") or deterministic_text),
                        "source": "ollama-forecast",
                        "model": result.get("model"),
                        "latency_seconds": result.get("latency_seconds", 0.0),
                        "plan": latest_plan,
                    }
                else:
                    payload = {
                        "text": deterministic_text,
                        "source": "forecast-engine",
                        "model": result.get("model"),
                        "latency_seconds": result.get("latency_seconds", 0.0),
                        "plan": latest_plan,
                    }
                runtime_cache.set(cache_key, payload, ttl_seconds=120)
                return payload
            except Exception:
                pass

        if asset_context is not None:
            payload = {
                "text": _asset_specific_summary(profile, portfolio_data, asset_context),
                "source": "shared-analyzer",
                "model": None,
                "latency_seconds": 0.0,
                "plan": latest_plan,
            }
            runtime_cache.set(cache_key, payload, ttl_seconds=60)
            return payload

        memory_block = "\n".join(memory[:5]) if memory else "None"
        prompt = f"""
SYSTEM:
You are an elite quantitative hedge fund AI.

STRICT RULES:
- USER PROFILE is ABSOLUTE truth
- MEMORY defines past decisions
- YOU MUST stay CONSISTENT with previous strategies
- Do NOT randomly switch strategies
- If a strategy was previously suggested, refine it instead of replacing it
- Use direct user language, not abstract quant jargon
- Return plain text only, with no markdown bolding
- Keep each output section concise and immediately actionable
- Respect the user's base currency and saved capital exactly
- When a specific asset is discussed, you must stay consistent with the shared system analyzer rather than inventing a different recommendation

USER PROFILE:
{profile}

PREVIOUS STRATEGY:
{last_strategy if last_strategy else 'None'}

CURRENT PORTFOLIO:
{portfolio_data}

MEMORY:
{memory_block}

USER QUESTION:
{user_query}

INSTRUCTIONS:
- Maintain STRATEGY CONSISTENCY
- Align with risk profile, capital, horizon, and interests
- Explain what the user should do next in plain language
- Avoid generic disclaimers
- Use exactly these headings with a colon after each one, and keep each one to one short paragraph:
Strategy:
Risk Level:
Allocation:
Reasoning:
""".strip()

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(generate_advisor_response_with_metadata, prompt, ADVISOR_TIMEOUT_SECONDS)
        try:
            result = future.result(timeout=ADVISOR_TIMEOUT_SECONDS + 2)
        except FutureTimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            fallback = {
                "text": plan_to_response_text(
                    latest_plan,
                    fallback_reason="The advisor timed out before Ollama finished, so the latest saved plan is being shown instead.",
                ),
                "source": "saved-plan-fallback" if latest_plan else "timeout-fallback",
                "model": None,
                "latency_seconds": ADVISOR_TIMEOUT_SECONDS,
                "plan": latest_plan,
            }
            return fallback
        finally:
            if not future.cancelled():
                executor.shutdown(wait=False, cancel_futures=True)

        source = str(result.get("source") or "")
        if source == "ollama":
            saved_plan = save_financial_plan(
                user_id=user_id,
                response_text=str(result.get("text") or ""),
                source=source,
                model=str(result.get("model") or ""),
            )
            strategy = (saved_plan or {}).get("strategy") or (latest_plan or {}).get("strategy")
            if strategy:
                build_portfolio_from_strategy(user_id, str(strategy))
            result["plan"] = saved_plan
            runtime_cache.set(cache_key, result, ttl_seconds=ADVISOR_CACHE_TTL_SECONDS)
            return result

        fallback = {
            "text": plan_to_response_text(
                latest_plan,
                fallback_reason="The live model could not complete this request, so the latest saved plan is shown for continuity.",
            ),
            "source": "saved-plan-fallback" if latest_plan else source or "fallback",
            "model": result.get("model"),
            "latency_seconds": result.get("latency_seconds", 0.0),
            "plan": latest_plan,
        }
        return fallback
    except Exception as exc:
        return {
            "text": plan_to_response_text(
                latest_plan,
                fallback_reason=f"Advisor service fallback activated because {exc}.",
            ),
            "source": "error-fallback",
            "model": None,
            "latency_seconds": 0.0,
            "plan": latest_plan,
        }
