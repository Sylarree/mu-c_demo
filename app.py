import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import streamlit as st


# =========================================================
# Configuration
# =========================================================

st.set_page_config(
    page_title="μc-rule Scheduling Demo",
    layout="wide",
)

TITLE = "μc-rule Scheduling Demo"
SUBTITLE = "Two queues, one server, same sample path across policies"


# =========================================================
# Data structures
# =========================================================

@dataclass
class Scenario:
    horizon: int
    lambda1: float
    lambda2: float
    mu1: float
    mu2: float
    c1: float
    c2: float
    discount_alpha: float
    seed: int


@dataclass
class SimulationResult:
    times: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    action: np.ndarray         # 1 = serve q1, 2 = serve q2, 0 = idle
    inst_cost: np.ndarray
    cum_disc_cost: np.ndarray


@dataclass
class AggregateResult:
    times: np.ndarray
    mean_cum_disc_cost: np.ndarray
    std_cum_disc_cost: np.ndarray
    mean_final_cost: float


# =========================================================
# Fixed scenario
# =========================================================

def get_scenario() -> Scenario:
    # Designed so μc-rule and highest-cost-only differ:
    # c2 > c1, but μ1*c1 > μ2*c2
    # Also made easier to visualize: service not too weak, arrivals calmer
    return Scenario(
        horizon=80,
        lambda1=0.45,
        lambda2=0.55,
        mu1=0.95,
        mu2=0.45,
        c1=3.0,
        c2=4.0,
        discount_alpha=0.985,
        seed=7,
    )


# =========================================================
# Random sample path generation
# =========================================================

def generate_common_sample_path(s: Scenario) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(s.seed)

    arrivals1 = rng.poisson(s.lambda1, size=s.horizon)
    arrivals2 = rng.poisson(s.lambda2, size=s.horizon)

    # If queue i is served during step t, service completes with prob mu_i
    service_success_q1 = rng.binomial(1, s.mu1, size=s.horizon)
    service_success_q2 = rng.binomial(1, s.mu2, size=s.horizon)

    return {
        "arrivals1": arrivals1,
        "arrivals2": arrivals2,
        "service_success_q1": service_success_q1,
        "service_success_q2": service_success_q2,
    }


# =========================================================
# Policies
# =========================================================

def policy_mu_c(q1: int, q2: int, s: Scenario, t: int) -> int:
    if q1 == 0 and q2 == 0:
        return 0
    if q1 == 0:
        return 2
    if q2 == 0:
        return 1
    return 1 if s.mu1 * s.c1 >= s.mu2 * s.c2 else 2


def policy_longest_queue(q1: int, q2: int, s: Scenario, t: int) -> int:
    if q1 == 0 and q2 == 0:
        return 0
    if q1 == 0:
        return 2
    if q2 == 0:
        return 1
    return 1 if q1 >= q2 else 2


def policy_highest_cost_only(q1: int, q2: int, s: Scenario, t: int) -> int:
    if q1 == 0 and q2 == 0:
        return 0
    if q1 == 0:
        return 2
    if q2 == 0:
        return 1
    return 1 if s.c1 >= s.c2 else 2


def policy_random_from_rng(
    q1: int, q2: int, s: Scenario, t: int, rng: np.random.Generator
) -> int:
    if q1 == 0 and q2 == 0:
        return 0
    if q1 == 0:
        return 2
    if q2 == 0:
        return 1
    return int(rng.choice([1, 2]))


POLICIES: Dict[str, Callable[[int, int, Scenario, int], int]] = {
    "μc-rule": policy_mu_c,
    "Longest queue first": policy_longest_queue,
    "Highest cost only": policy_highest_cost_only,
    "Random": lambda q1, q2, s, t: 0,   # handled separately
}


# =========================================================
# Simulation
# =========================================================

def simulate_policy(
    policy_name: str,
    policy_fn,
    s: Scenario,
    common_path: Dict[str, np.ndarray],
    random_policy_rng: np.random.Generator | None = None,
) -> SimulationResult:
    q1 = np.zeros(s.horizon + 1, dtype=int)
    q2 = np.zeros(s.horizon + 1, dtype=int)
    action = np.zeros(s.horizon, dtype=int)
    inst_cost = np.zeros(s.horizon, dtype=float)
    cum_disc_cost = np.zeros(s.horizon, dtype=float)

    arrivals1 = common_path["arrivals1"]
    arrivals2 = common_path["arrivals2"]
    service_success_q1 = common_path["service_success_q1"]
    service_success_q2 = common_path["service_success_q2"]

    running = 0.0

    for t in range(s.horizon):
        if policy_name == "Random":
            a = policy_random_from_rng(q1[t], q2[t], s, t, random_policy_rng)
        else:
            a = policy_fn(q1[t], q2[t], s, t)

        action[t] = a

        # Cost at step t uses current queue lengths
        inst_cost[t] = s.c1 * q1[t] + s.c2 * q2[t]
        running += (s.discount_alpha ** t) * inst_cost[t]
        cum_disc_cost[t] = running

        next_q1 = q1[t]
        next_q2 = q2[t]

        # Service first, then arrivals
        if a == 1 and q1[t] > 0:
            if service_success_q1[t] == 1:
                next_q1 -= 1
        elif a == 2 and q2[t] > 0:
            if service_success_q2[t] == 1:
                next_q2 -= 1

        next_q1 += arrivals1[t]
        next_q2 += arrivals2[t]

        q1[t + 1] = next_q1
        q2[t + 1] = next_q2

    return SimulationResult(
        times=np.arange(s.horizon),
        q1=q1,
        q2=q2,
        action=action,
        inst_cost=inst_cost,
        cum_disc_cost=cum_disc_cost,
    )


@st.cache_data
def compute_all_results(n_replications: int = 200) -> Tuple[
    Scenario, Dict[str, SimulationResult], Dict[str, AggregateResult]
]:
    s = get_scenario()

    # Representative run for animations
    common_path_demo = generate_common_sample_path(s)
    demo_results: Dict[str, SimulationResult] = {}
    for name, fn in POLICIES.items():
        rng = np.random.default_rng(123) if name == "Random" else None
        demo_results[name] = simulate_policy(name, fn, s, common_path_demo, rng)

    # Averaged cost curves for comparison
    aggregate_results: Dict[str, AggregateResult] = {}
    for name, fn in POLICIES.items():
        all_cum_costs = []

        for rep in range(n_replications):
            rep_s = Scenario(**{**s.__dict__, "seed": s.seed + rep})
            common_path = generate_common_sample_path(rep_s)
            rng = np.random.default_rng(10000 + rep) if name == "Random" else None
            res = simulate_policy(name, fn, rep_s, common_path, rng)
            all_cum_costs.append(res.cum_disc_cost)

        all_cum_costs = np.array(all_cum_costs)
        aggregate_results[name] = AggregateResult(
            times=np.arange(s.horizon),
            mean_cum_disc_cost=np.mean(all_cum_costs, axis=0),
            std_cum_disc_cost=np.std(all_cum_costs, axis=0),
            mean_final_cost=float(np.mean(all_cum_costs[:, -1])),
        )

    return s, demo_results, aggregate_results


# =========================================================
# Plotting helpers
# =========================================================

def policy_color(action: int) -> str:
    if action == 1:
        return "#ff9999"   # red-ish
    if action == 2:
        return "#99ccff"   # blue-ish
    return "#dddddd"


def action_text(action: int) -> str:
    if action == 1:
        return "Serve Q1"
    if action == 2:
        return "Serve Q2"
    return "Idle"


def action_color(action: int) -> str:
    if action == 1:
        return "red"
    if action == 2:
        return "blue"
    return "black"


def draw_mini_system_panel(
    ax,
    result: SimulationResult,
    t: int,
    scenario: Scenario,
    policy_name: str,
):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    q1_now = int(result.q1[t])
    q2_now = int(result.q2[t])
    a = int(result.action[t]) if t < len(result.action) else 0

    max_vis = max(
        1,
        np.max(result.q1[: len(result.action) + 1]),
        np.max(result.q2[: len(result.action) + 1]),
    )

    # Layout: two horizontal queues, stacked vertically, server on the right
    q_x = 1.0
    q_w = 4.5
    q_h = 0.7
    q1_y = 3.7
    q2_y = 2.2

    # Outline bars
    ax.add_patch(Rectangle((q_x, q1_y), q_w, q_h, fill=False, linewidth=1.8))
    ax.add_patch(Rectangle((q_x, q2_y), q_w, q_h, fill=False, linewidth=1.8))

    # Fills
    fill1 = q_w * min(q1_now / max_vis, 1.0)
    fill2 = q_w * min(q2_now / max_vis, 1.0)

    ax.add_patch(Rectangle((q_x, q1_y), fill1, q_h, alpha=0.55))
    ax.add_patch(Rectangle((q_x, q2_y), fill2, q_h, alpha=0.55))

    ax.text(q_x - 0.1, q1_y + q_h / 2, "Q1", ha="right", va="center", fontsize=10)
    ax.text(q_x - 0.1, q2_y + q_h / 2, "Q2", ha="right", va="center", fontsize=10)

    ax.text(q_x + q_w + 0.15, q1_y + q_h / 2, f"{q1_now}", ha="left", va="center", fontsize=10)
    ax.text(q_x + q_w + 0.15, q2_y + q_h / 2, f"{q2_now}", ha="left", va="center", fontsize=10)

    # Server
    server_x = 7.1
    server_y = 2.8
    server_w = 1.7
    server_h = 1.2

    ax.add_patch(
        Rectangle(
            (server_x, server_y),
            server_w,
            server_h,
            facecolor=policy_color(a),
            edgecolor="black",
            linewidth=1.8,
        )
    )
    ax.text(server_x + server_w / 2, server_y + server_h / 2, "Server",
            ha="center", va="center", fontsize=10)

    # Arrows
    ax.add_patch(
        FancyArrowPatch(
            (q_x + q_w, q1_y + q_h / 2),
            (server_x, server_y + 0.9),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=2.0,
            color="red" if a == 1 else "black",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (q_x + q_w, q2_y + q_h / 2),
            (server_x, server_y + 0.3),
            arrowstyle="->",
            mutation_scale=14,
            linewidth=2.0,
            color="blue" if a == 2 else "black",
        )
    )

    # Title and small status
    ax.text(0.1, 5.55, policy_name, fontsize=11, weight="bold")
    ax.text(0.1, 5.05, action_text(a), fontsize=10, color=action_color(a), weight="bold")

    if policy_name == "μc-rule":
        ax.text(
            0.1, 4.55,
            rf"$\mu_1c_1={scenario.mu1*scenario.c1:.2f},\ \mu_2c_2={scenario.mu2*scenario.c2:.2f}$",
            fontsize=9
        )


def draw_system_grid(
    demo_results: Dict[str, SimulationResult],
    t: int,
    scenario: Scenario,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for ax, name in zip(axes, demo_results.keys()):
        draw_mini_system_panel(ax, demo_results[name], t, scenario, name)

    fig.tight_layout()
    return fig


def draw_cost_grid(
    aggregate_results: Dict[str, AggregateResult],
    t: int,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()

    for ax, name in zip(axes, aggregate_results.keys()):
        res = aggregate_results[name]
        x = res.times[: t + 1]
        y = res.mean_cum_disc_cost[: t + 1]

        ax.plot(x, y, linewidth=2)
        ax.set_title(name, fontsize=10, weight="bold")
        ax.set_xlim(0, len(res.times) - 1)
        ax.set_xlabel("time", fontsize=9)
        ax.set_ylabel("mean cum. cost", fontsize=9)
        ax.grid(alpha=0.25)

        final_so_far = y[-1] if len(y) > 0 else 0.0
        ax.text(
            0.03,
            0.92,
            f"{final_so_far:.1f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", alpha=0.15),
        )

    fig.tight_layout()
    return fig


# =========================================================
# UI helpers
# =========================================================

def scenario_summary(s: Scenario) -> str:
    return (
        f"Fixed scenario: "
        f"λ₁={s.lambda1}, λ₂={s.lambda2}, "
        f"μ₁={s.mu1}, μ₂={s.mu2}, "
        f"c₁={s.c1}, c₂={s.c2}, "
        f"α={s.discount_alpha}, horizon={s.horizon}"
    )


# =========================================================
# Main app
# =========================================================

def main() -> None:
    st.title(TITLE)
    st.caption(SUBTITLE)

    scenario, demo_results, aggregate_results = compute_all_results()

    with st.expander("Scenario", expanded=False):
        st.write(scenario_summary(scenario))
        st.write(
            "Top row: one representative sample path for each policy. "
            "Bottom row: mean cumulative discounted cost over many replications."
        )
        st.write(
            "This scenario is chosen so that highest-cost-only and μc-rule are different policies."
        )

    controls = st.columns([1, 1])
    with controls[0]:
        autoplay = st.checkbox("Autoplay", value=False)
    with controls[1]:
        speed = st.selectbox("Speed", ["Slow", "Medium", "Fast"], index=2)

    max_t = scenario.horizon - 1

    if "timestep" not in st.session_state:
        st.session_state.timestep = 0

    t = st.slider("Time", 0, max_t, st.session_state.timestep, 1)
    st.session_state.timestep = t

    st.subheader("System behavior")
    fig_systems = draw_system_grid(demo_results, st.session_state.timestep, scenario)
    st.pyplot(fig_systems, clear_figure=True)

    st.subheader("Policy comparison")
    fig_costs = draw_cost_grid(aggregate_results, st.session_state.timestep)
    st.pyplot(fig_costs, clear_figure=True)

    ranking = sorted(
        [(name, res.mean_final_cost) for name, res in aggregate_results.items()],
        key=lambda kv: kv[1],
    )
    st.markdown("**Mean final discounted cost ranking**")
    for i, (name, value) in enumerate(ranking, start=1):
        st.write(f"{i}. {name}: {value:.2f}")

    if autoplay and st.session_state.timestep < max_t:
        step_jump = {"Slow": 1, "Medium": 3, "Fast": 6}[speed]
        delay = {"Slow": 0.40, "Medium": 0.10, "Fast": 0.02}[speed]
        time.sleep(delay)
        st.session_state.timestep = min(max_t, st.session_state.timestep + step_jump)
        st.rerun()


if __name__ == "__main__":
    main()