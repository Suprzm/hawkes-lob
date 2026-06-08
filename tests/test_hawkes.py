import numpy as np
from src.hawkes_lob.hawkes import simulate_poisson, simulate_hawkes, hawkes_intensity, fit_hawkes

def test_poisson_returns_list():
    events = simulate_poisson(lambda_=10.0, T=100.0, seed=42)
    assert isinstance(events, list)

def test_poisson_events_in_range():
    T = 100.0
    events = simulate_poisson(lambda_=10.0, T=T, seed=42)
    assert all(0 < t <= T for t in events)

def test_poisson_events_sorted():
    events = simulate_poisson(lambda_=10.0, T=100.0, seed=42)
    assert events == sorted(events)

def test_poisson_mean_close_to_lambda():
    """Over a long time horizon, the number of events should be close to λ*T"""
    lambda_ = 10.0
    T = 10000.0
    events = simulate_poisson(lambda_=lambda_, T=T, seed=42)
    expected = lambda_ * T
    # 5% tolerance
    assert abs(len(events) - expected) / expected < 0.05

def test_hawkes_intensity_base():
    """In the absence of past events, λ(t) = μ"""
    assert hawkes_intensity(t=5.0, events=[], mu=2.0, 
                            alpha=0.5, beta=1.0) == 2.0

def test_hawkes_intensity_increases_after_event():
    """Immediately after an event, λ(t) > μ"""
    intensity = hawkes_intensity(t=1.01, events=[1.0], 
                                 mu=2.0, alpha=0.5, beta=1.0)
    assert intensity > 2.0

def test_hawkes_intensity_decays():
    """The intensity decreases over time after an event"""
    i1 = hawkes_intensity(t=1.1, events=[1.0], mu=2.0, alpha=0.5, beta=1.0)
    i2 = hawkes_intensity(t=1.5, events=[1.0], mu=2.0, alpha=0.5, beta=1.0)
    i3 = hawkes_intensity(t=3.0, events=[1.0], mu=2.0, alpha=0.5, beta=1.0)
    assert i1 > i2 > i3

def test_hawkes_branching_ratio():
    """Should raise an error if α/β ≥ 1"""
    import pytest
    with pytest.raises(AssertionError):
        simulate_hawkes(mu=1.0, alpha=2.0, beta=1.0, T=10.0)

def test_hawkes_events_in_range():
    events = simulate_hawkes(mu=2.0, alpha=0.5, beta=1.0, T=100.0, seed=42)
    assert all(0 < t <= 100.0 for t in events)

def test_hawkes_more_events_than_poisson():
    """Hawkes generates more events than Poisson with the same μ (clustering)"""
    hawkes_events = simulate_hawkes(mu=2.0, alpha=0.5, beta=1.0, 
                                    T=1000.0, seed=42)
    poisson_events = simulate_poisson(lambda_=2.0, T=1000.0, seed=42)
    assert len(hawkes_events) > len(poisson_events)

def test_mle_recovers_parameters():
    """
    Validate MLE on synthetic data.
    
    Tests:
    - mu recovery (< 20% error)
    - branching ratio recovery (< 20% error)  
    - implied mean intensity E[lambda] recovery (< 15% error)
    
    Note: individual alpha/beta are not tested due to known identifiability
    issues on the eta=alpha/beta manifold — documented in README.
    """
    true_mu, true_alpha, true_beta = 2.0, 0.5, 1.0
    true_branching = true_alpha / true_beta  # 0.5

    # Mean intensity in stationary regime: E[lambda] = mu / (1 - eta)
    true_mean_intensity = true_mu / (1 - true_branching)  # = 4.0

    events = simulate_hawkes(mu=true_mu, alpha=true_alpha,
                             beta=true_beta, T=2000.0, seed=42)

    result = fit_hawkes(events, T=2000.0)

    # 1. mu recovers well
    assert abs(result['mu'] - true_mu) / true_mu < 0.20

    # 2. branching ratio well identified
    assert abs(result['branching_ratio'] - true_branching) / true_branching < 0.20

    # 3. implied mean intensity — depends on both mu and eta, not alpha/beta separately
    fitted_mean_intensity = result['mu'] / (1 - result['branching_ratio'])
    assert abs(fitted_mean_intensity - true_mean_intensity) / true_mean_intensity < 0.15

    # 4. sanity check: alpha and beta are at least in the right ballpark
    assert 0.1 < result['alpha'] < 2.0
    assert 0.1 < result['beta'] < 5.0

def test_mle_branching_ratio_valid():
    """Fitted branching ratio must be < 1"""
    events = simulate_hawkes(mu=2.0, alpha=0.5, beta=1.0, T=200.0, seed=42)
    result = fit_hawkes(events, T=200.0)
    assert result['branching_ratio'] < 1.0

def test_mle_returns_expected_keys():
    events = simulate_hawkes(mu=2.0, alpha=0.5, beta=1.0, T=100.0, seed=42)
    result = fit_hawkes(events, T=100.0)
    assert all(k in result for k in 
               ['mu', 'alpha', 'beta', 'branching_ratio', 'neg_log_likelihood'])