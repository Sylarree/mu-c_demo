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
    # Sharper tradeoff so policy differences are easier to see
    return Scenario(
        horizon=100,
        lambda1=0.65,
        lambda2=0.85,
        mu1=0.95,
        mu2=0.35,
        c1=1.2,
        c2=4.8,
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
    # Random is handled separately because it needs an RNG
    "Random": lambda q1, q2, s, t: 0,
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

    # One representative run for left-side animation
    common_path_demo = generate_common_sample_path(s)
    demo_results: Dict[str, SimulationResult] = {}
    for name, fn in POLICIES.items():
        rng = np.random.default_rng(123) if name == "Random" else None
        demo_results[name] = simulate_policy(name, fn, s, common_path_demo, rng)

    # Averaged results for right-side comparison
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
# Plotting: left big queue/server figure
# =========================================================

def draw_system_figure(
    result: SimulationResult,
    t: int,
    scenario: Scenario,
    policy_name: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    q1_now = int(result.q1[t])
    q2_now = int(result.q2[t])
    a = int(result.action[t]) if t < len(result.action) else 0
    inst = float(result.inst_cost[t]) if t < len(result.inst_cost) else 0.0
    cum = float(result.cum_disc_cost[t]) if t < len(result.cum_disc_cost) else 0.0

    mu_c1 = scenario.mu1 * scenario.c1
    mu_c2 = scenario.mu2 * scenario.c2

    # Clean text area
    ax.text(0.3, 9.6, f"Policy: {policy_name}", fontsize=13, weight="bold")
    ax.text(0.3, 9.1, f"Time step: {t}", fontsize=12)

    if a == 1:
        serve_txt = "Serving Queue 1"
        serve_color = "red"
    elif a == 2:
        serve_txt = "Serving Queue 2"
        serve_color = "blue"
    else:
        serve_txt = "Idle"
        serve_color = "black"

    ax.text(0.3, 8.6, serve_txt, fontsize=12, color=serve_color, weight="bold")
    ax.text(0.3, 8.0, rf"$\mu_1c_1={mu_c1:.2f}$,   $\mu_2c_2={mu_c2:.2f}$", fontsize=12)
    ax.text(0.3, 7.4, f"Instantaneous cost: {inst:.2f}", fontsize=12)
    ax.text(0.3, 6.8, f"Cumulative discounted cost: {cum:.2f}", fontsize=12)

    # Queue bars side by side
    max_vis = max(
        1,
        np.max(result.q1[: len(result.action) + 1]),
        np.max(result.q2[: len(result.action) + 1]),
    )

    bar_bottom = 1.4
    bar_height = 4.6
    bar_width = 1.6
    q1_x = 1.5
    q2_x = 4.0

    h1 = bar_height * min(q1_now / max_vis, 1.0)
    h2 = bar_height * min(q2_now / max_vis, 1.0)

    for x, label, h, q_now in [
        (q1_x, "Queue 1", h1, q1_now),
        (q2_x, "Queue 2", h2, q2_now),
    ]:
        ax.add_patch(Rectangle((x, bar_bottom), bar_width, bar_height, fill=False, linewidth=2))
        ax.add_patch(Rectangle((x, bar_bottom), bar_width, h, alpha=0.55))
        ax.text(x + bar_width / 2, bar_bottom + bar_height + 0.35, label, ha="center", fontsize=12)
        ax.text(x + bar_width / 2, bar_bottom + bar_height / 2, f"{q_now}", ha="center", va="center", fontsize=14)

    # Server
    server_x = 8.0
    server_y = 3.0
    server_w = 2.2
    server_h = 1.8

    server_color = "#DDDDDD"
    if a == 1:
        server_color = "#ffb3b3"
    elif a == 2:
        server_color = "#b3d9ff"

    ax.add_patch(Rectangle((server_x, server_y), server_w, server_h,
                           facecolor=server_color, edgecolor="black", linewidth=2))
    ax.text(server_x + server_w / 2, server_y + server_h / 2,
            "Server", ha="center", va="center", fontsize=13)

    # Arrows
    q1_arrow_color = "red" if a == 1 else "black"
    q2_arrow_color = "blue" if a == 2 else "black"

    ax.add_patch(
        FancyArrowPatch(
            (q1_x + bar_width, bar_bottom + bar_height / 2),
            (server_x, server_y + 1.2),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.5,
            color=q1_arrow_color,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (q2_x + bar_width, bar_bottom + bar_height / 2),
            (server_x, server_y + 0.6),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.5,
            color=q2_arrow_color,
        )
    )

    fig.tight_layout()
    return fig


# =========================================================
# Plotting: right-side cost panels
# =========================================================

def draw_cost_grid(
    aggregate_results: Dict[str, AggregateResult],
    t: int,
    selected_policy: str,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flatten()

    names = list(aggregate_results.keys())

    for ax, name in zip(axes, names):
        res = aggregate_results[name]
        x = res.times[: t + 1]
        y = res.mean_cum_disc_cost[: t + 1]

        ax.plot(x, y, linewidth=2)
        ax.set_title(name, fontsize=10, weight="bold" if name == selected_policy else None)

        if name == selected_policy:
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)

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
        st.write("Left panel: one representative sample path.")
        st.write("Right panel: mean cumulative discounted cost over many replications.")

    top_controls = st.columns([1.2, 1, 1])
    with top_controls[0]:
        selected_policy = st.selectbox(
            "Policy shown on the left",
            list(demo_results.keys()),
            index=0,
        )
    with top_controls[1]:
        autoplay = st.checkbox("Autoplay", value=False)
    with top_controls[2]:
        speed = st.selectbox("Speed", ["Slow", "Medium", "Fast"], index=1)

    max_t = scenario.horizon - 1

    if "timestep" not in st.session_state:
        st.session_state.timestep = 0

    t = st.slider("Time", 0, max_t, st.session_state.timestep, 1)
    st.session_state.timestep = t

    left, right = st.columns([1.05, 1.15])

    with left:
        st.subheader("System behavior")
        fig_left = draw_system_figure(
            demo_results[selected_policy],
            st.session_state.timestep,
            scenario,
            selected_policy,
        )
        st.pyplot(fig_left, clear_figure=True)

    with right:
        st.subheader("Policy comparison")
        fig_right = draw_cost_grid(
            aggregate_results,
            st.session_state.timestep,
            selected_policy,
        )
        st.pyplot(fig_right, clear_figure=True)

        ranking = sorted(
            [(name, res.mean_final_cost) for name, res in aggregate_results.items()],
            key=lambda kv: kv[1],
        )
        st.markdown("**Mean final discounted cost ranking**")
        for i, (name, value) in enumerate(ranking, start=1):
            st.write(f"{i}. {name}: {value:.2f}")

    if autoplay and st.session_state.timestep < max_t:
        step_jump = {"Slow": 1, "Medium": 2, "Fast": 5}[speed]
        delay = {"Slow": 0.50, "Medium": 0.20, "Fast": 0.03}[speed]
        time.sleep(delay)
        st.session_state.timestep = min(max_t, st.session_state.timestep + step_jump)
        st.rerun()


if __name__ == "__main__":
    main()