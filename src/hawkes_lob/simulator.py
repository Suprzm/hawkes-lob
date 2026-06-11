import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
from src.hawkes_lob.hawkes import simulate_hawkes
from src.hawkes_lob.config import CALIBRATED_PARAMS


@dataclass
class MarketState:
    """
    Snapshot of the market at a given time.
    Simplified LOB: we track mid-price and spread only.
    """
    time:       float
    mid_price:  float
    spread:     float
    last_side:  str    # 'buy' or 'sell'


@dataclass 
class SimulationResult:
    """Full simulation output."""
    states:      List[MarketState] = field(default_factory=list)
    buy_times:   List[float]       = field(default_factory=list)
    sell_times:  List[float]       = field(default_factory=list)
    T:           float             = 0.0


def simulate_lob(T: float = 3600.0,
                 initial_price: float = 100.0,
                 initial_spread: float = 0.01,
                 volatility: float = 0.001,
                 seed: int = None) :
    """
    Simulates a simplified LOB driven by calibrated Hawkes processes.

    Two independent Hawkes processes drive buy and sell arrivals.
    Price evolves as a random walk driven by order flow imbalance.

    Args:
        T              : simulation horizon in seconds
        initial_price  : starting mid-price
        initial_spread : bid-ask spread (constant for simplicity)
        volatility     : price impact per trade (std of price move)
        seed           : random seed

    Returns:
        SimulationResult with full market state history
        :rtype: [SimulationResult]
    """
    if seed is not None:
        np.random.seed(seed)

    mu    = CALIBRATED_PARAMS['mu']
    alpha = CALIBRATED_PARAMS['alpha']
    beta  = CALIBRATED_PARAMS['beta']

    # Simulate buy and sell arrivals as independent Hawkes processes
    buy_times  = simulate_hawkes(mu=mu, alpha=alpha, beta=beta, T=T, seed=seed)
    sell_times = simulate_hawkes(mu=mu, alpha=alpha, beta=beta, T=T,
                                 seed=seed + 1 if seed else None)

    # Merge and sort all events
    events = (
        [(t, 'buy')  for t in buy_times] +
        [(t, 'sell') for t in sell_times]
    )
    events.sort(key=lambda x: x[0])

    # Simulate price as random walk driven by order flow
    result = SimulationResult(buy_times=buy_times,
                              sell_times=sell_times, T=T)

    price = initial_price

    for t, side in events:
        # Price impact: buys push price up, sells push price down
        direction = 1.0 if side == 'buy' else -1.0
        price_move = direction * abs(np.random.normal(0, volatility * price))
        price = max(price + price_move, 0.01)  # price stays positive

        result.states.append(MarketState(
            time=t,
            mid_price=price,
            spread=initial_spread,
            last_side=side,
        ))

    return result


def plot_simulation(result) :
    """
    Visualises simulated price path and trade arrivals.
    :type result: [SimulationResult]
    :rtype: [None]
    """
    import matplotlib.pyplot as plt

    times  = [s.time      for s in result.states]
    prices = [s.mid_price for s in result.states]
    sides  = [s.last_side for s in result.states]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Trade arrivals
    buy_t  = [t for t, s in zip(times, sides) if s == 'buy']
    sell_t = [t for t, s in zip(times, sides) if s == 'sell']

    ax1.scatter(buy_t,  [1]*len(buy_t),  marker='|', s=100,
                color='steelblue', alpha=0.4, label='buy')
    ax1.scatter(sell_t, [1]*len(sell_t), marker='|', s=100,
                color='crimson', alpha=0.4, label='sell')
    ax1.set_yticks([])
    ax1.legend(loc='upper right')
    ax1.set_title(f'Simulated trade arrivals '
                  f'(n={len(result.states)}, T={result.T:.0f}s)')

    # Price path
    ax2.plot(times, prices, color='gray', linewidth=0.8)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Mid price')
    ax2.set_title('Simulated mid-price path')

    plt.tight_layout()
    plt.show()