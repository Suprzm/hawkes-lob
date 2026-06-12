# hawkes-lob

[![CI](https://github.com/Suprzm/hawkes-lob/actions/workflows/ci.yml/badge.svg)](https://github.com/Suprzm/hawkes-lob/actions/workflows/ci.yml)

**Limit Order Book simulator using Hawkes processes** — calibrated on real crypto market data.

Hawkes processes capture the self-exciting nature of order flow: a trade triggers more trades, cancellations cluster around price moves, and activity bursts are followed by calm periods. This project models these dynamics and validates them against BTC/USDT market microstructure data from Binance.

---

## Motivation

Standard models assume order arrivals follow a Poisson process — events are independent and arrive at a constant rate. This is empirically wrong.

In real limit order books, order flow exhibits:
- **Clustering** — trades arrive in bursts, not uniformly
- **Cross-excitation** — a buy market order increases the probability of further buys *and* ask-side cancellations
- **Mean reversion** — activity spikes decay exponentially back to a baseline

Hawkes processes model exactly this: the conditional intensity λ(t) is not constant but jumps after each event and decays exponentially toward a baseline μ.

```
λ(t) = μ + Σᵢ α · exp(−β(t − tᵢ))   for all tᵢ < t
```

Where:
- **μ** — baseline intensity (order arrival rate in quiet market)
- **α** — jump size after each event (how much one trade excites the next)
- **β** — decay speed (how fast the excitement fades)
- **η = α/β** — branching ratio (average number of child events per parent event, must be < 1 for stationarity)

---

## Project Structure

```
hawkes-lob/
├── src/
│   └── hawkes_lob/
│       ├── __init__.py
│       └── hawkes.py          # Core: Poisson, Hawkes simulation & MLE
├── tests/
│   └── test_hawkes.py         # 13 unit tests
├── notebooks/
│   ├── 01_lob_understanding.ipynb    # LOB structure & event types
│   └── 02_hawkes_development.ipynb   # Poisson vs Hawkes visualisation & MLE
├── data/
│   ├── raw/                   # Raw Binance tick data (gitignored)
│   └── processed/             # Cleaned event sequences
├── Makefile
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/Suprzm/hawkes-lob.git
cd hawkes-lob
python -m venv .venv && source .venv/bin/activate
make install
make test
```

---

## What's implemented

### 1. Limit Order Book — core structure

A LOB is represented as two dictionaries mapping price levels to volumes:

```python
bids = {99.5: 100, 99.0: 250}   # buyers
asks = {100.0: 150, 100.5: 300}  # sellers
```

Four event types are modelled:

| Event | Description |
|---|---|
| Limit order | Add volume at a price level |
| Market order | Execute immediately at best bid/ask |
| Cancellation | Remove volume from the book |
| Execution | Match a bid and an ask |

Key quantities computed: mid-price, spread, market order execution (buy & sell).

### 2. Poisson process baseline

A homogeneous Poisson process with constant intensity λ serves as the null model. Events are simulated via exponential inter-arrival times:

```python
events = simulate_poisson(lambda_=10.0, T=50.0, seed=42)
# → 499 events uniformly distributed over [0, 50]
```

**Limitation**: Poisson assumes independence — no clustering, no memory. Empirically false for order flow.

### 3. Hawkes process

Simulated via the **Ogata thinning algorithm** (1981):

1. Propose a candidate event from a Poisson process with majorising intensity λ̄
2. Accept with probability λ(t) / λ̄
3. Update λ̄ after each accepted event

```python
events = simulate_hawkes(mu=2.0, alpha=0.5, beta=1.0, T=50.0, seed=42)
# → 183 events with visible clustering (vs ~100 for equivalent Poisson)
```

The clustering is immediately visible — zones of dense activity separated by quiet periods, matching the empirical signature of real order flow.

### 4. MLE calibration

Parameters (μ, α, β) are estimated from observed event sequences by maximising the Hawkes log-likelihood:

```
log L(μ,α,β) = −∫₀ᵀ λ(t)dt + Σᵢ log λ(tᵢ)
```

The integral has a closed-form solution for exponential kernels, making optimisation efficient. A recursive formula computes Σᵢ log λ(tᵢ) in O(n).

```python
result = fit_hawkes(events, T=500.0)
# → {'mu': 1.837, 'alpha': 0.348, 'beta': 0.681, 'branching_ratio': 0.512}
```

**Known limitation**: μ, α, β are individually hard to separate due to the elongated likelihood surface along the η = α/β = const manifold. The branching ratio is recovered with ~2% error on synthetic data; individual parameters exhibit higher variance (~17–30%). This is a well-documented identifiability issue in Hawkes MLE — see Bacry et al. (2015).

---

## Results on synthetic data

Simulation with true parameters (μ=2.0, α=0.5, β=1.0, T=2000):

| Parameter | True | Estimated | Error |
|---|---|---|---|
| μ | 2.000 | 1.837 | 8.2% |
| α | 0.500 | 0.348 | 30.4% — known identifiability issue |
| β | 1.000 | 0.681 | 31.9% — known identifiability issue |
| **η = α/β** | **0.500** | **0.512** | **2.3%** |
| **E[λ] = μ/(1−η)** | **4.000** | **3.876** | **3.1%** |

The branching ratio is the key quantity for LOB analysis — it measures the average number of child events per parent event. η close to 1 signals an unstable, self-reinforcing market (flash crash risk).

---

## Results on real ETH/BTC data (Binance)

Calibrated on 1000 trades over 114 minutes:

| Parameter | Value | Interpretation |
|---|---|---|
| μ | 0.0812 trades/s | ~1 exogenous trade every 12s |
| α | 25.46 | large jump size per trade |
| β | 57.19 | excitation halves in ~12ms |
| **η** | **0.445** | each trade generates 0.44 child trades |
| E[λ] | 0.1463 trades/s | model vs empirical error: **0.1%** |

Goodness of fit: model-implied mean intensity matches empirical 
rate with 0.1% error — confirming stationarity of ETH/BTC 
over the 114-minute window.

**Why ETH/BTC and not BTC/USDT?**  
BTC/USDT on Binance processes ~16 trades/second — the stationary  
Hawkes model is misspecified on such high-frequency data  
(88% intensity error observed). ETH/BTC at 0.15 trades/second  
satisfies the stationarity assumption required for reliable MLE.

---

## Poisson vs Hawkes — visual comparison

| | Poisson | Hawkes |
|---|---|---|
| λ(t) | Constant | Time-varying, jumps at each event |
| Events | Uniformly distributed | Clustered bursts + quiet periods |
| Memory | None | Each event influences future intensity |
| LOB realism | Low | High |
| Events generated (μ=2, T=50) | ~100 | ~183 (+83%) |

---

## Tests

18 unit tests covering the full pipeline — from process simulation
to strategy execution:

```bash
make test
```

```
tests/test_hawkes.py::test_poisson_returns_list                    PASSED
tests/test_hawkes.py::test_poisson_events_in_range                 PASSED
tests/test_hawkes.py::test_poisson_events_sorted                   PASSED
tests/test_hawkes.py::test_poisson_mean_close_to_lambda            PASSED
tests/test_hawkes.py::test_hawkes_intensity_base                   PASSED
tests/test_hawkes.py::test_hawkes_intensity_increases_after_event  PASSED
tests/test_hawkes.py::test_hawkes_intensity_decays                 PASSED
tests/test_hawkes.py::test_hawkes_branching_ratio                  PASSED
tests/test_hawkes.py::test_hawkes_events_in_range                  PASSED
tests/test_hawkes.py::test_hawkes_more_events_than_poisson         PASSED
tests/test_hawkes.py::test_mle_recovers_parameters                 PASSED
tests/test_hawkes.py::test_mle_branching_ratio_valid               PASSED
tests/test_hawkes.py::test_mle_returns_expected_keys               PASSED
tests/test_simulator.py::test_simulation_returns_states            PASSED
tests/test_simulator.py::test_simulation_price_positive            PASSED
tests/test_simulator.py::test_simulation_times_sorted              PASSED
tests/test_simulator.py::test_strategy_runs                        PASSED
tests/test_simulator.py::test_strategy_metrics_keys                PASSED
```

**Hawkes process (10)** — simulation correctness, intensity properties
(base level, post-event jump, exponential decay), stationarity constraint
(η < 1 required for convergence), clustering validation (Hawkes generates
more events than equivalent Poisson), and MLE parameter recovery on
synthetic data with 2.3% branching ratio error.

**LOB Simulator (3)** — market state generation, price positivity
throughout simulation, chronological ordering of events.

**Strategy (2)** — end-to-end execution of Avellaneda-Stoikov on a
simulated LOB, and strategy metrics output validation.

Note: MLE tests cover branching ratio η and implied mean intensity
E[λ] = μ/(1−η) rather than individual α/β — known identifiability
limitation documented in Known Limitations.

---

## Backtesting — Avellaneda-Stoikov on simulated ETH/BTC LOB

Simulation: T=3600s, 554 trades, calibrated Hawkes parameters.

| Metric | Value |
|---|---|
| Final P&L | +0.0130 BTC |
| Realized P&L (spread capture) | +0.0399 BTC |
| Inventory P&L | −0.0270 BTC |
| Max drawdown | 0.0002 BTC |
| Fill rate | 23.6% |
| Fills (buy / sell) | 65 / 66 |
| Max inventory | 3 ETH |

*Sharpe ratio is not reported — inapplicable to event-driven 
market-making with irregular inter-arrival times.*

The realized/unrealized decomposition confirms the core  
market-making trade-off: spread capture (+0.040 BTC) partially  
offset by inventory risk (−0.027 BTC), yielding net +0.013 BTC.

**Known limitations:**
- Simplified fill model: no adverse selection
- Independent buy/sell Hawkes processes (no cross-excitation)  
- Constant spread assumption in the LOB simulator

---

## Extensions

Directions for future development:

- [ ] Multivariate Hawkes — cross-excitation between buy and sell sides
- [ ] Empirical k calibration from real order book data
- [ ] Adverse selection in the fill model
- [ ] Regime-switching Hawkes for non-stationary markets
- [ ] Spectral estimation (Bacry & Muzy 2014) for stable α/β recovery

---

## References

- Hawkes, A.G. (1971). *Spectra of some self-exciting and mutually exciting point processes*. Biometrika.
- Ogata, Y. (1981). *On Lewis' simulation method for point processes*. IEEE Transactions on Information Theory.
- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance.
- Cont, R., Stoikov, S. & Talreja, R. (2010). *A stochastic model for order book dynamics*. Operations Research.
- Bacry, E., Mastromatteo, I. & Muzy, J.F. (2015). *Hawkes processes in finance*. Market Microstructure and Liquidity.

---

## Author

**Marvin Kameni** — [marvinkmn@gmail.com](mailto:marvinkmn@gmail.com)

*Centrale Lyon — General Engineering, Mathematics & Decision*

*Previously: Algorithmic Trader @ Flowdesk (Singapore)*
