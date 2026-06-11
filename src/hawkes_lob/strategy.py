import numpy as np
from typing import List
from dataclasses import dataclass, field
from src.hawkes_lob.simulator import SimulationResult, MarketState


@dataclass
class TradeRecord:
    """Records a single fill (executed trade)."""
    time:      float
    side:      str    # 'buy' or 'sell'
    price:     float
    quantity:  float


@dataclass
class StrategyState:
    """Tracks strategy performance over time."""
    times:          list[float] = field(default_factory=list)
    pnl:            list[float] = field(default_factory=list)
    realized_pnl:   list[float] = field(default_factory=list)
    unrealized_pnl: list[float] = field(default_factory=list)
    inventory:      list[float] = field(default_factory=list)
    bid_prices:     list[float] = field(default_factory=list)
    ask_prices:     list[float] = field(default_factory=list)
    n_fills_buy:    int = 0
    n_fills_sell:   int = 0

# Et dans la boucle après chaque fill

def avellaneda_stoikov(simulation,
                       gamma = 0.1,
                       sigma = 0.001,
                       k = 1.5,
                       T = 3600.0,
                       quantity = 1.0):
    """
    Avellaneda-Stoikov market-making strategy.

    The market maker quotes bid and ask prices around a reservation price
    that accounts for inventory risk. Wider spread when inventory is large.

    Key equations:
        reservation_price = mid - gamma * sigma^2 * (T-t) * inventory
        optimal_spread    = gamma * sigma^2 * (T-t) + (2/gamma)*ln(1 + gamma/k)

    Args:
        simulation : SimulationResult from simulate_lob() / type [SimulationResult]
        gamma      : risk aversion (higher = tighter inventory management) / type [float]
        sigma      : price volatility per unit time / type [float]
        k          : order book liquidity parameter / type [float]
        T          : time horizon (seconds) / type [float]
        quantity   : size of each order / type [float]



    Returns:
        StrategyState with P&L, inventory and quote history
        :rtype: [StrategyState]
    """
    state     = StrategyState()
    cash      = 0.0
    inventory = 0.0

    for market in simulation.states:
        t   = market.time
        mid = market.mid_price
        tau = max(T - t, 1e-6) # time remaining

        # Reservation price — adjusted for inventory risk
        reservation = mid - gamma * sigma**2 * tau * inventory

        # Optimal spread
        spread = (gamma * sigma**2 * tau +
                  (2 / gamma) * np.log(1 + gamma / k))
        spread = max(spread, market.spread) # at least the market spread

        bid = reservation - spread / 2
        ask = reservation + spread / 2

        # Fill logic: we get filled when market crosses our quotes
        # A market sell order fills our bid if execution price <= our bid
        # A market buy order fills our ask if execution price >= our ask
        if market.last_side == 'sell':
            distance  = max(mid - bid, 0)
            fill_prob = np.exp(-k * distance)
            if np.random.random() < fill_prob:
                cash      -= bid * quantity
                inventory += quantity
                state.n_fills_buy += 1

        elif market.last_side == 'buy':
            distance  = max(ask - mid, 0)
            fill_prob = np.exp(-k * distance)
            if np.random.random() < fill_prob:
                cash      += ask * quantity
                inventory -= quantity
                state.n_fills_sell += 1

        # Mark-to-market P&L
        realized_pnl   = cash                 # locked-in P&L from fills
        unrealized_pnl = inventory * mid      # open position at current price
        total_pnl      = realized_pnl + unrealized_pnl

        state.realized_pnl.append(realized_pnl)
        state.unrealized_pnl.append(unrealized_pnl)

        state.times.append(t)
        state.pnl.append(total_pnl)
        state.inventory.append(inventory)
        state.bid_prices.append(bid)
        state.ask_prices.append(ask)

    return state


def plot_strategy(state, simulation):
    """
    Visualises strategy P&L, inventory and quotes.
    :type state: StrategyState
    :type simulation: SimulationResult
    :rtype: None
    """
    import matplotlib.pyplot as plt

    # Build a price lookup by time
    price_by_time = {s.time: s.mid_price for s in simulation.states}
    prices = [price_by_time.get(t, state.bid_prices[i]) 
              for i, t in enumerate(state.times)]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # P&L
    axes[0].plot(state.times, state.pnl, color='green', linewidth=1)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('P&L')
    axes[0].set_title('Strategy P&L (mark-to-market)')
    axes[0].plot(state.times, state.pnl,
             color='green', linewidth=1, label='total (MtM)')
    axes[0].plot(state.times, state.realized_pnl,
             color='darkgreen', linewidth=1,
             linestyle='--', label='realized')
    axes[0].plot(state.times, state.unrealized_pnl,
             color='orange', linewidth=0.8,
             alpha=0.7, label='unrealized')
    axes[0].legend()
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('P&L (BTC)')
    axes[0].set_title('Strategy P&L (mark-to-market)')

    # Inventory
    axes[1].plot(state.times, state.inventory,
                 color='steelblue', linewidth=1)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Inventory')
    axes[1].set_title('Inventory over time')

    # Price + quotes
    axes[2].plot(state.times, prices,
                 color='gray', linewidth=0.8, label='mid price')
    axes[2].plot(state.times, state.bid_prices,
                 color='steelblue', linewidth=0.5,
                 alpha=0.7, label='bid quote')
    axes[2].plot(state.times, state.ask_prices,
                 color='crimson', linewidth=0.5,
                 alpha=0.7, label='ask quote')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Price')
    axes[2].set_title('Quotes around mid-price')
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def strategy_metrics(state):
    """
    Computes key performance metrics.
    :type state: [StrategyState]
    :rtype: [dict]
    """
    pnl          = np.array(state.pnl)
    realized     = np.array(state.realized_pnl)
    unrealized   = np.array(state.unrealized_pnl)
    inventory    = np.array(state.inventory)

    final_pnl    = pnl[-1]
    max_pnl      = pnl.max()
    min_pnl      = pnl.min()
    max_drawdown = (np.maximum.accumulate(pnl) - pnl).max()
    max_inventory = np.abs(inventory).max()
    pnl_std      = pnl.std()
    sharpe       = (final_pnl / pnl_std) if pnl_std > 0 else 0.0
    n_fills     = state.n_fills_buy + state.n_fills_sell
    fill_rate   = n_fills / len(state.pnl) * 100

    return {
        'final_pnl':      final_pnl,
        'realized_pnl':     realized[-1],    
        'unrealized_pnl':   unrealized[-1],
        'max_pnl':        max_pnl,
        'min_pnl':        min_pnl,
        'max_drawdown':     max_drawdown,
        'max_inventory':  max_inventory,
        'sharpe':         sharpe,
        'n_steps':        len(pnl),
        'n_fills':        n_fills,           
        'n_fills_buy':    state.n_fills_buy, 
        'n_fills_sell':   state.n_fills_sell,
        'fill_rate_pct':  fill_rate,
    }