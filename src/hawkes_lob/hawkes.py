import numpy as np
from typing import List

### - Poisson distribution - constant rate of events 

def simulate_poisson(lambda_: float, T: float, seed: int = None):
    """
    Simulates a homogeneous Poisson process on [0, T].
    
    Method: simulation using exponential inter-arrival times.
    If X ~ Exp(lambda), then the times between events 
    follow an exponential distribution with parameter lambda.
    
    Args:
        lambda_ : rate (events per unit time)
        T       : simulation horizon
        seed    : random seed for reproducibility
    
    Returns:
        List of event times

    """
    if seed is not None:
        np.random.seed(seed)
    
    events = []
    t = 0.0
    
    while True:
        # Temps jusqu'au prochain événement ~ Exp(lambda)
        inter_arrival = np.random.exponential(1.0 / lambda_)
        t += inter_arrival
        
        if t > T:
            break
            
        events.append(t)
    
    return events


def plot_poisson(events: List[float], T: float, lambda_: float):
    """Visualize the process: events + counting process"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
    # Plot 1 — the events (ticks)
    ax1.scatter(events, np.ones(len(events)), 
                marker='|', s=200, color='steelblue')
    ax1.set_xlim(0, T)
    ax1.set_yticks([])
    ax1.set_title(f'Poisson events (λ={lambda_}, n={len(events)})')
    ax1.set_xlabel('Temps')
    
    # Plot 2 — the counting process N(t)
    times = [0] + events + [T]
    counts = list(range(len(times) - 1)) + [len(events)]
    ax2.step(times, counts, where='post', color='steelblue')
    ax2.plot([0, T], [0, lambda_ * T], 'r--', 
             alpha=0.5, label=f'Theoretical average λt')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('N(t)')
    ax2.set_title('Counting process')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

### Hawkes distribution 

def hawkes_intensity(t: float, events: list[float], mu: float, alpha: float, beta: float):
    """
    Computes the conditional intensity λ(t) of a Hawkes process.
    
    λ(t) = μ + Σ alpha · exp(−β(t - tᵢ))

    Args:
        t      : time at which to calculate the intensity
        events : list of past event times
        mu     : base intensity
        alpha  : amplitude du saut après chaque événement
        beta   : rate of decline
    
    Returns:
        Value of λ(t)
    """
    past_events = [ti for ti in events if ti < t]
    
    if not past_events:
        return mu
    
    excitation = sum(alpha * np.exp(-beta * (t - ti)) for ti in past_events)
    return mu + excitation


def plot_intensity(events: List[float], T: float,
                   mu: float, alpha: float, beta: float):
    """Plot λ(t) over the events"""
    import matplotlib.pyplot as plt
    
    # Detailed time grid for plotting intensity
    times = np.linspace(0, T, 1000)
    intensities = [hawkes_intensity(t, events, mu, alpha, beta) 
                   for t in times]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Events
    ax1.scatter(events, np.ones(len(events)),
                marker='|', s=200, color='steelblue')
    ax1.set_yticks([])
    ax1.set_title(f'Hawkes events (n={len(events)})')
    
    # Intensity
    ax2.plot(times, intensities, color='crimson', linewidth=1.5)
    ax2.axhline(mu, color='gray', linestyle='--', 
                alpha=0.5, label=f'μ = {mu}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('λ(t)')
    ax2.set_title('Conditional intensity λ(t)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def simulate_hawkes(mu: float, alpha: float, beta: float,
                    T: float, seed: int = None) :
    """
    Simulates a one-dimensional Hawkes process using the thinning method (Ogata 1981).
    
    Concept: We simulate a Poisson process with an upper bound on the rate λ_bar,
    then accept or reject each proposed event
    
    Stationarity condition: α/β < 1 (branching ratio)
    
    Args:
        mu    : base rate
        alpha : jump amplitude
        beta  : decay rate
        T     : simulation horizon
        seed  : random seed
    
    Returns:
        List of event times
    """
    assert alpha / beta < 1, \
        f"Branching ratio α/β = {alpha/beta:.2f} ≥ 1 : processus non stationnaire"
    
    if seed is not None:
        np.random.seed(seed)
    
    events = []
    t = 0.0
    lambda_bar = mu  # intensité majorante initiale
    
    while t < T:
        # 1. Proposer un candidat depuis Poisson(lambda_bar)
        t += np.random.exponential(1.0 / lambda_bar)
        
        if t > T:
            break
        
        # 2. Calculer l'intensité réelle à ce temps
        lambda_t = hawkes_intensity(t, events, mu, alpha, beta)
        
        # 3. Accepter avec probabilité λ(t) / λ_bar
        u = np.random.uniform(0, 1)
        if u <= lambda_t / lambda_bar:
            events.append(t)
        
        # 4. Mettre à jour λ_bar = intensité actuelle
        #    (car λ(t) est décroissante entre événements)
        lambda_bar = hawkes_intensity(t, events, mu, alpha, beta)
        lambda_bar = max(lambda_bar, mu)  # jamais en dessous de mu
    
    return events