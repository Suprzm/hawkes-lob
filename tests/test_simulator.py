from src.hawkes_lob.simulator import simulate_lob
from src.hawkes_lob.strategy import avellaneda_stoikov, strategy_metrics


def test_simulation_returns_states():
    result = simulate_lob(T=100.0, seed=42)
    assert len(result.states) > 0


def test_simulation_price_positive():
    result = simulate_lob(T=100.0, seed=42)
    assert all(s.mid_price > 0 for s in result.states)


def test_simulation_times_sorted():
    result = simulate_lob(T=100.0, seed=42)
    times = [s.time for s in result.states]
    assert times == sorted(times)


def test_strategy_runs():
    result = simulate_lob(T=100.0, seed=42)
    state  = avellaneda_stoikov(result, T=100.0)
    assert len(state.pnl) > 0


def test_strategy_metrics_keys():
    result  = simulate_lob(T=100.0, seed=42)
    state   = avellaneda_stoikov(result, T=100.0)
    metrics = strategy_metrics(state)
    assert all(k in metrics for k in
               ['final_pnl', 'max_inventory'])