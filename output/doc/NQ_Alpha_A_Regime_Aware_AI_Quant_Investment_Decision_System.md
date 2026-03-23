# NQ Alpha: A Regime-Aware AI Quant Investment Decision System

A Mohammed Faazil
Computer Science Engineer and AI Researcher, Chennai, Tamil Nadu, India
mohammed.faazil.16@gmail.com



## Abstract

NQ ALPHA is a regime-aware AI quant investment decision system designed to unify live market analysis, portfolio construction, and advisor reasoning inside one coherent decision architecture. The platform resolves natural-language asset requests into market symbols, retrieves reachable OHLCV histories, engineers time-series and cross-sectional signals, infers an alpha score through a transformer-based model, converts that score into deterministic investment actions, and then contextualizes the result through user-specific portfolio state, backtesting metrics, and an aligned local-LLM advisor. The main design principle is alignment: the analyzer, scanner, strategy lab, allocation layer, and advisor are not allowed to operate as disconnected modules. Instead, all of them inherit the same signal path, the same recommendation thresholds, and the same saved portfolio truth. This makes NQ ALPHA structurally different from generic advisory chatbots and from research prototypes that stop at model evaluation.

The live platform is trained on a curated universe of 30 selected assets while remaining operationally capable of reaching a broader universe of 636 assets through symbol intelligence and market-data access. The deployed alpha model consumes a 30-step window of engineered features, uses 4 self-attention heads across 2 transformer encoder layers, and conditions its final alpha output on market regime states (bull, normal, volatile, crisis). Strategy Lab converts planned rupee amounts into weights, preserves cash explicitly, and evaluates historical return, Sharpe ratio, drawdown, volatility, and baseline behavior. The advisor layer answers money questions from saved state rather than from free-form generation and answers asset questions through the same analyzer logic used by the dashboard. The result is a product-oriented quant system whose novelty lies not in one isolated algorithm, but in the aligned integration of signal extraction, decision logic, capital planning, and explainable advisory output.

## Keywords

Quantitative finance, transformer alpha model, regime-aware inference, portfolio analytics, investment decision system, market intelligence, reinforcement learning lineage, optimal transport context, financial AI advisor, explainable allocation.

## 1. Introduction

Modern financial decision systems are expected to do more than classify price direction. A usable platform must identify what asset the user means, process live market information, infer a signal that is mathematically defensible, convert that signal into a recommendation, estimate the portfolio impact of acting on that signal, and then explain the decision in language that does not contradict the numerical engine. In practice, many systems fail because these layers are built independently. A scanner might say one thing, an optimization engine another, and a large-language-model assistant something completely different. NQ ALPHA was developed to solve that systems-level problem rather than only the model-level problem.

The project originates from the NeuroQuant research direction authored by A Mohammed Faazil, where regime-aware reinforcement learning and stochastic optimal transport were used to study adaptive portfolio behavior under market regime shifts. That original work showed that market-state awareness and transition sensitivity matter deeply in finance, especially when static policies are exposed to non-stationary returns. NQ ALPHA extends that line of thought into a broader investment decision architecture. Instead of limiting the contribution to a reinforcement learning agent, the current system integrates transformer-based alpha inference, symbol intelligence, live asset coverage, user-specific portfolio memory, and a constrained local advisor. The resulting system is still research-grounded, but it is far closer to an operational decision product.

The central argument of this paper is that the next generation of quant products should not be evaluated only by raw predictive metrics. They should also be evaluated by whether their internal modules remain logically aligned once exposed to real users. NQ ALPHA therefore treats alignment as a first-class research and engineering goal. Asset-specific advisory responses must agree with the analyzer. Budget answers must come from saved state, not language-model improvisation. Portfolio metrics must be explained in ways a user can actually act on. By embedding these constraints directly into the architecture, the system becomes more trustworthy, more extensible, and more relevant for both product deployment and future publication.

## 2. Literature Survey

The research lineage of NQ ALPHA begins with the NeuroQuant paper, A. Mohammed Faazil, NeuroQuant: A Reinforcement Learning Framework with Stochastic Optimal Transport for Adaptive Portfolio Management Under Market Regime Shifts., which focuses on reinforcement learning, market-regime awareness, and stochastic optimal transport as a mechanism for identifying distributional market shifts. That work is important because it emphasizes adaptation as the central challenge in quantitative portfolio management. A static policy may succeed in one market state and fail catastrophically in another. By using Wasserstein-distance-based regime reasoning, the NeuroQuant line of work contributes a principled way to identify transitions instead of relying only on fixed indicator thresholds.

A second major reference point is the base paper, Han Yue, Jiapeng Liu, and Qin Zhang, Applications of Markov Decision Process Model and Deep Learning in Quantitative Portfolio Management during the COVID-19 Pandemic, Systems, 2022., which uses a Markov decision process formulation and deep representation learning to improve portfolio management under the COVID-19 period. Its contribution is especially valuable because it highlights two persistent problems in financial AI: the difficulty of state representation and the difficulty of designing stable reward structures in highly noisy markets. That paper also reinforces the importance of risk-adjusted metrics such as Sharpe and related ratios, which remain useful in NQ ALPHA even though the current system moves away from a pure actor-critic live decision layer.

Beyond these two direct anchors, several broader literatures shape NQ ALPHA. Transformer models introduced by Vaswani et al. changed sequential modeling by allowing each timestep to attend to all others, making them well-suited to financial patterns that are path-dependent but not strictly local. Computational optimal transport provides a mathematically grounded way to compare distributions under shifting support, which is valuable when market return regimes change abruptly. Classical portfolio theory beginning with Markowitz remains relevant because allocation still requires reasoning about return, risk, and the trade-off between them. Reinforcement-learning-for-trading research shows the potential of sequential action models, but also demonstrates how brittle they can become without careful feature design, reward shaping, and market-state awareness.

NQ ALPHA differs from much of this literature in one key respect: its innovation is architectural as much as algorithmic. Many prior works optimize one model or one reward function, but do not solve the production problem of keeping analyzer, allocation, and advisor logic consistent. In NQ ALPHA, the novelty is the combination of transformer alpha inference, deterministic signal mapping, rupee-first portfolio planning, persistent user state, and a constrained advisor that is forced to inherit analyzer truth and portfolio truth whenever those truths already exist. That systems-level alignment is the out-of-the-box contribution that most similar papers do not operationalize.

A further distinction is research communication. Most finance systems either present equations without interaction or interaction without equations. NQ ALPHA explicitly attempts to support both. The protected System Guide, the publication-style paper mode, the modular architecture view, and the allocation-intelligence pages together create a rare bridge between a publishable technical narrative and a working product interface. This dual-use design is important because it allows the system to be studied academically, demonstrated operationally, and improved iteratively without splitting into disconnected research and product branches.

## 3. Existing System Challenges

Financial AI systems face three structural challenges. The first is symbol ambiguity. Real users do not speak in exchange-qualified tickers. They say ICICI, HDFC, Bank of Baroda, Apple, Bitcoin, or crude oil. A system that cannot reliably translate natural asset intent into a valid market symbol immediately becomes brittle, especially across Indian and international markets. The second challenge is architectural mismatch. Many production prototypes leak training-time pipelines into inference-time APIs, causing slow responses, stale snapshots, and conflicting logic across routes. The third challenge is explanation. Even when a model works, the surrounding system often fails to express risk, conviction, allocation, and remaining cash in a way that maps to actual user decisions.

These challenges are intensified in multi-layer systems where different modules are built at different times. A recommendation engine may use one rule set, a portfolio page another, and an advisor still another. Once that happens, the user sees an avoid signal in one panel and a buy suggestion in another. Likewise, an allocation page may show capital in one format while the advisor answers with hallucinated numbers in another currency. These are not just user-experience bugs; they are violations of quantitative integrity. In a serious investment system, contradictory outputs are a research problem because they imply hidden inconsistencies in the decision architecture itself.

NQ ALPHA treats these issues as first-order design constraints. The project therefore asks a different research question from many traditional model papers: not only can a signal be learned, but can the full decision stack remain internally truthful when that signal is exposed to live user workflows?

## 4. Proposed System Advantages

NQ ALPHA addresses the challenges above through a modular but strongly coupled architecture. At the bottom of the stack, symbol intelligence translates user intent into searchable, resolvable market symbols and can persist newly discovered assets into the broader all-assets universe. The feature layer then transforms raw OHLCV into engineered state representations that include price behavior, volatility behavior, cross-sectional context, and regime-sensitive interaction features. These representations feed a transformer alpha model that outputs a scalar alpha score rather than an opaque action sequence. The recommendation layer uses deterministic thresholds so asset-specific decisions remain explainable and reusable across all user interfaces.

The second major advantage is the rupee-first portfolio layer. Instead of forcing users to think in decimal allocations, the system allows capital planning in money terms and converts those planned rupee amounts into weights internally. This reverses the usual quant-product burden: the system adapts to the user rather than forcing the user to adapt to the system. Strategy Lab, Allocation, and the Advisor therefore all work with the same concept of capital base, invested amount, cash reserve, holding weights, and projected future value. This is especially important for retail-facing quant systems, where user comprehension is part of model usability.

The out-of-the-box novelty of NQ ALPHA lies in its alignment layer. Asset-specific advisor queries are routed through the same analyzer path as the dashboard. Money questions are answered from the saved portfolio state, not from the LLM. Saved financial plans persist separately from raw text memory. The result is a system in which explanation does not sit outside the quantitative engine; it sits on top of the same truth state. That is the project's main departure from many research papers and many modern AI products.

## 5. Motivation and Goal

The motivation behind NQ ALPHA is to close the gap between research-grade quant logic and an actually usable investment interface. Many strong papers stop at backtests, and many strong products stop at UI polish. The present system aims to combine the discipline of the first with the accessibility of the second. In other words, the goal is not merely to score alpha, and not merely to display analysis, but to create an integrated AI quant investment decision system whose outputs are explainable, portfolio-aware, and operationally aligned.

A second motivation is to expand the practical reach of a model beyond its core training universe while still being honest about modeling limits. The live platform can operationally access Indian equities, US equities, crypto pairs, and other assets that the data provider and symbol layer can resolve. Yet the system also preserves the distinction between the curated training universe and the broader operational universe. This honesty is important because a research-grade product should not hide out-of-distribution risk beneath a polished interface.

## 6. Architecture Diagram

The updated NQ ALPHA architecture is organized around eight operational blocks: Input Sources, Symbol Intelligence, Feature Engine, Regime Intelligence, Advisor Memory, NQ Alpha Core, Portfolio Strategy, and System Outputs. Input Sources collect portfolio state, user profile, live and historical market data, user query intent, and macro context. Symbol Intelligence converts natural asset names into resolved market symbols and keeps a persistent registry across Indian, US, and crypto markets. The Feature Engine converts raw OHLCV and contextual signals into normalized engineered features, while Regime Intelligence models distribution shift, market conditioning, volatility state, and regime classification as a contextual branch that influences downstream scoring.

At the center of the architecture, the NQ Alpha Core produces alpha score, confidence score, regime-aware inference, and recommendation logic. Above and beside that core, Advisor Memory grounds LLM responses in saved user memory, financial plan memory, portfolio grounding, analyzer alignment, and explainable output. Portfolio Strategy turns the alpha signal into portfolio builder actions, backtesting logic, future projection, and allocation decisions. All of these modules finally converge inside System Outputs, where buy-hold-avoid signals, allocation views, alpha timelines, risk signals, dashboard analytics, and user advice are exposed. This revised architecture is more original to NQ ALPHA and better expresses the system as a live investment decision stack rather than as a generic pipeline.

## 7. Module Explanation with Implementation

This section explains how the live NQ ALPHA system turns data into an actionable investment decision workflow. The modules are not merely conceptual; they correspond to real code paths and persistent system components in the running platform.

## 7.1 Symbol Intelligence and Data Layer

The platform starts by resolving natural asset names into market symbols. This layer is essential because a financial product that requires the user to know the exact ticker syntax is not truly user-intelligent. NQ ALPHA uses aliases, search indexing, fuzzy matching, provider-aware fallbacks, and persistent asset insertion so that unresolved assets can be discovered and then retained for future use. At the current runtime state, the curated training universe contains 30 assets, while the broader live all-assets table contains 636 assets. The accessible universe mix is currently crypto: 100, stock: 536.

Once a symbol is resolved, the live data service retrieves OHLCV history that can be used for feature construction, alpha inference, analyzer output, price visualization, and timeline generation. In operational terms, this means the system can analyze many Indian and international symbols even when the core model was trained on a more curated subset. The important research caveat is that operational reach does not automatically imply identical statistical confidence across all regions and asset classes.

## 7.2 Feature Engineering and Regime Modeling

The feature engine constructs a research universe of 40 engineered signals. These include price returns, moving-average deviations, volatility descriptors, relative-strength measures, breadth context, lagged signals, interaction terms, and regime-transition features. The live model currently uses 10 active inference features, while the broader training-active set includes: relative_return_1, rank_return_1, vol_adjusted_return_1, return_zscore_5, price_vs_sma20, RSI_14, rolling_rank_mean_10, price_vs_sma50, momentum_x_volatility, momentum_x_regime. The narrower live set exists because production inference benefits from a stable selected feature subset rather than from an unfiltered research feature universe.

Mathematically, the project begins from a forward return target and then converts that target into a cross-sectional ranking objective. Feature smoothing and target smoothing are applied to improve stability across time. Global context channels such as market return, market volatility, breadth-up percentage, breadth-above-SMA percentage, breadth dispersion, cross-sectional volatility, and top-bottom spread are injected so the model does not evaluate an asset in isolation. Regime states are explicitly modeled, which allows the system to interpret the same raw pattern differently depending on whether the market is bull, normal, volatile, or crisis.

This matters because real investment decisions are rarely driven by one indicator family alone. A momentum reading has a different meaning when the market is in crisis than when the market is in a broad bull expansion. Similarly, a relative-strength move during narrow breadth conditions may not carry the same reliability as the same move during strong cross-sectional participation. By embedding global context and regime information directly into the feature space, NQ ALPHA attempts to formalize the kind of conditional reasoning that experienced discretionary portfolio managers often perform implicitly.

## 7.3 Transformer-Based Alpha Model

The alpha model consumes a 30-step temporal window of engineered features. Each timestep is projected into a hidden representation, enriched with positional information, and processed by a transformer encoder using 4 attention heads across 2 encoder layers. The hidden temporal state is then fused with a learned regime embedding, and a compact output head maps the fused representation into a scalar alpha score. This alpha is not a raw probability and not a direct price target. It is a relative opportunity score that measures how attractive the asset appears relative to the target horizon and the model's learned ranking behavior.

The key mathematics behind this layer include: (1) a forward-return target over a fixed horizon, (2) a cross-sectional ranking transformation, (3) normalized and clipped features, (4) self-attention over the temporal window, and (5) regime-conditioned alpha projection. The reason a transformer is used instead of a simpler static model is that financial structure is rarely only local. Trend continuation, reversal, volatility compression, and regime transition all involve relationships between non-adjacent periods. Self-attention provides a compact way to learn those dependencies without relying on manually designed lag logic alone.

## 7.4 Recommendation, Portfolio, and Simulation Layer

The recommendation layer converts alpha into one of three deterministic actions: BUY, HOLD, or AVOID. This choice is deliberate. A deterministic rule ensures that the scanner, analyzer, and advisor remain aligned because the signal mapping is explicit and reusable. Confidence is then derived as a bounded function of the alpha magnitude so the UI can express conviction without pretending to estimate a precise probability of success.

Portfolio planning begins in rupee space. The user enters planned amounts, and the system converts them into weights internally while preserving unallocated capital as cash. Backtesting then computes a historical equity path along with return, Sharpe ratio, drawdown, volatility, and baseline comparisons. This layer is novel in the sense that it translates quant metrics into a user-comprehensible capital-planning workflow rather than leaving them as abstract research statistics. It also enables future value projections by mapping annualized performance into 1 to 2 year, 3 to 5 year, or 5 plus year scenarios.

## 7.5 Advisor, Memory, and User State

NQ ALPHA stores user information in separated layers: profiles, portfolios, financial plans, and memory messages are not collapsed into one text stream. The current runtime contains 17 users, 8 portfolio records, 15 financial plan records, and 114 memory records, with vector memory maintained separately in ChromaDB. This separation is important because a money answer should come from state, not from conversational inference.

The advisor therefore works as a constrained reasoning layer. If the user asks how much money remains, the answer comes from saved portfolio state. If the user asks about a specific stock, the answer is routed through the shared analyzer path so the advisor cannot contradict the dashboard. Only broader strategic language and explanatory synthesis are delegated to the local Ollama model. This design is one of the system's strongest contributions because it directly addresses a major failure mode of many AI-finance products: numerical inconsistency hidden behind polished text.

## 8. Results and Discussion

The most important result of NQ ALPHA is systems-level alignment. The platform now enforces agreement between analysis, allocation, and advisory response. This may sound like an engineering detail, but it is a substantive contribution because financial AI systems often fail at the exact point where multiple modules begin interacting. A signal engine that looks strong in isolation can still become dangerous if its outputs are diluted, contradicted, or hallucinated by the surrounding product stack.

A second result is practical breadth. The live system can analyze a broader operational universe than the strict training universe, which expands product usefulness while preserving explicit caution about modeling confidence. A third result is user intelligibility. The system translates portfolio construction into rupees, exposes future value ranges, explains regime meaning in simple language, and preserves exact money state for the advisor. These properties do not replace quantitative rigor; they operationalize it.

In comparison with the earlier NeuroQuant paper, the current project is broader at the product layer. It keeps regime-awareness and transport-inspired thinking in its research lineage, but introduces a transformer alpha engine, shared analyzer-advisor logic, a live asset-discovery path, and a portfolio state layer. In comparison with the base paper, it moves beyond representation learning plus actor-critic reward design and instead focuses on a full-stack investment decision pipeline. The out-of-the-box contribution is therefore the aligned architecture itself: one truth path for analysis, capital planning, and explanation.

## 8.1 Operational Evaluation

Operationally, the platform already behaves like a real decision system rather than a static research artifact. The live runtime currently reports 35 cached feature snapshots, 8 portfolio records, 15 saved financial-plan records, and 114 memory records. These counts matter because they show that NQ ALPHA is not only a trained model; it is a persistent stateful system that can remember users, store allocations, persist plans, and serve analysis repeatedly under the same logic path. The system also tracks the distinction between the selected training universe and the broader accessible live universe, which is critical for honest deployment in open-ended symbol search conditions.

From a product-performance standpoint, the most important evaluation criterion is not only response speed or isolated alpha behavior, but consistency under user interaction. When a user searches an asset, opens the dashboard, adds it to Strategy Lab, and then asks the advisor about it, the system now preserves one view of that asset across the entire flow. This is equivalent to a systems-level benchmark: analyzer agreement, allocation agreement, and advisor agreement. Many AI products would fail this test because their natural-language layer is not constrained by the signal layer. NQ ALPHA passes it by routing stock-specific questions through shared analyzer truth and money questions through saved portfolio state.

## 8.2 Out-of-the-Box Contribution and Research Distinction

The strongest distinction of NQ ALPHA from many similar papers is that it treats the surrounding product architecture as part of the research contribution. Traditional quantitative papers usually stop once a policy, ranking model, or reward function is demonstrated. In contrast, NQ ALPHA asks what happens after the signal leaves the model. How is it converted into a recommendation? How is that recommendation translated into a capital plan? How does the user know how much money is still free? What prevents the advisor from contradicting the analyzer? These are not superficial interface questions; they determine whether a quantitative system remains correct once humans begin interacting with it.

This is where NQ ALPHA is out-of-the-box relative to the prior work. It combines a transformer alpha model with live symbol intelligence, deterministic recommendation thresholds, rupee-first portfolio construction, persistent user state, exact money answers, a protected research guide, and a local-LLM advisor that is constrained rather than unconstrained. In research terms, this means the novelty is hybrid: algorithmic on the signal side, architectural on the system side, and interaction-aware on the user side. That combination is the paper's core identity and is what justifies presenting NQ ALPHA not merely as another investment model, but as an AI quant investment decision system.

## 9. Conclusion

NQ ALPHA demonstrates that a modern quant system should be evaluated as an integrated decision architecture rather than as a disconnected collection of models. The platform combines signal extraction, regime conditioning, deterministic recommendation logic, rupee-first allocation, persistent user state, and constrained advisor reasoning into one coherent stack. This makes it simultaneously more explainable and more deployable than many isolated research prototypes.

The project's exclusive contribution is not only that it uses alpha modeling, regime awareness, and portfolio analytics, but that it forces these layers to share one consistent interpretation of the market and one consistent representation of the user's capital. That alignment is what makes NQ ALPHA a stronger foundation for future productization and for a more mature publication-grade research program.

## Mathematical Formulation Highlights

- Forward return target: r_(i,t->t+H) = close_(i,t+H) / close_(i,t) - 1
- Cross-sectional rank target: y_(i,t) = rank(r_(i,t->t+H)) / N
- Feature smoothing: x'_t = 0.7 * x_t + 0.3 * x_(t-1)
- Target smoothing: y'_t = 0.8 * y_t + 0.2 * y_(t-1)
- Feature projection: h_t = W_2 * GELU(W_1 x~_t + b_1) + b_2
- Self-attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
- Temporal encoder: z_1,...,z_T = TransformerEncoder(h_1 + PE_1,...,h_T + PE_T)
- Regime fusion: g = [z_T ; E(r_t)]
- Alpha head: alpha = w^T Dropout(GELU(W g + b)) + c
- Decision rule: BUY if alpha > 0.02; HOLD if -0.02 <= alpha <= 0.02; AVOID if alpha < -0.02
- Confidence mapping: confidence = min(20 * |alpha|, 1.0)
- Portfolio weights: w_i = a_i / C_total
- Portfolio return: r_(p,t) = sum_i w_i * r_(i,t)
- Compounded wealth: V_t = V_(t-1) * (1 + r_(p,t))
- Annualized Sharpe: Sharpe = sqrt(252) * mean(r_p) / std(r_p)
- Drawdown: DD_t = 1 - V_t / max_(s <= t) V_s

## 10. References

- [1] A. Mohammed Faazil, NeuroQuant: A Reinforcement Learning Framework with Stochastic Optimal Transport for Adaptive Portfolio Management Under Market Regime Shifts.
- [2] Han Yue, Jiapeng Liu, and Qin Zhang, Applications of Markov Decision Process Model and Deep Learning in Quantitative Portfolio Management during the COVID-19 Pandemic, Systems, 2022.
- [3] Harry Markowitz, Portfolio Selection, The Journal of Finance, 1952.
- [4] Ashish Vaswani et al., Attention Is All You Need, NeurIPS, 2017.
- [5] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton, Layer Normalization, arXiv:1607.06450, 2016.
- [6] John Moody and Matthew Saffell, Learning to Trade via Direct Reinforcement, IEEE Transactions on Neural Networks, 2001.
- [7] Zhengyao Jiang, Dixing Xu, and Jinjun Liang, A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem, arXiv:1706.10059, 2017.
- [8] Xiao-Yang Liu et al., FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance, 2020.
- [9] Matthew F. Dixon, Igor Halperin, and Paul Bilokon, Machine Learning in Finance: From Theory to Practice, 2020.
- [10] Marco Cuturi and Gabriel Peyre, Computational Optimal Transport, Foundations and Trends in Machine Learning, 2019.
- [11] James D. Hamilton, A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle, Econometrica, 1989.
- [12] Bryan Lim et al., Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting, International Journal of Forecasting, 2021.
- [13] John Hull, Risk Management and Financial Institutions, Wiley, 2018.
- [14] Tucker Balch et al., How to Evaluate Trading Strategies: Beyond Return and Volatility, Journal of Trading Practice discussions and academic finance benchmarking literature.
- [15] Internal NQ ALPHA engineering repository, protected system guide, live analyzer, portfolio layer, and advisor alignment stack documentation.
