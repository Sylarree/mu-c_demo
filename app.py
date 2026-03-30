import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

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


# =========================================================
# Fixed scenario
# =========================================================

def get_scenario() -> Scenario:
    # Chosen so that "serve more expensive queue" and "serve longer queue"
    # can differ from μc-rule in interesting ways.
    return Scenario(
        horizon=80,
        lambda1=0.75,
        lambda2=0.95,
        mu1=0.90,
        mu2=0.45,
        c1=2.0,
        c2=3.5,
        discount_alpha=0.98,
        seed=7,
    )


# =========================================================
# Random sample path generation
# =========================================================

def generate_common_sample_path(s: Scenario) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(s.seed)

    arrivals1 = rng.poisson(s.lambda1, size=s.horizon)
    arrivals2 = rng.poisson(s.lambda2, size=s.horizon)

    # Pre-generate geometric-style service completion Bernoulli trials:
    # if queue i is served during time t, service completes with prob mu_i
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


def policy_random_fixed_seed_factory(seed: int) -> Callable[[int, int, Scenario, int], int]:
    rng = np.random.default_rng(seed)

    def _policy(q1: int, q2: int, s: Scenario, t: int) -> int:
        if q1 == 0 and q2 == 0:
            return 0
        if q1 == 0:
            return 2
        if q2 == 0:
            return 1
        return int(rng.choice([1, 2]))
    return _policy


POLICIES: Dict[str, Callable[[int, int, Scenario, int], int]] = {
    "μc-rule": policy_mu_c,
    "Longest queue first": policy_longest_queue,
    "Highest cost only": policy_highest_cost_only,
    "Random": policy_random_fixed_seed_factory(12345),
}


# =========================================================
# Simulation
# =========================================================

def simulate_policy(
    policy_name: str,
    policy_fn: Callable[[int, int, Scenario, int], int],
    s: Scenario,
    common_path: Dict[str, np.ndarray],
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
        a = policy_fn(q1[t], q2[t], s, t)
        action[t] = a

        # Cost during step t based on current queue lengths
        inst_cost[t] = s.c1 * q1[t] + s.c2 * q2[t]
        running += (s.discount_alpha ** t) * inst_cost[t]
        cum_disc_cost[t] = running

        # One-step update: service first, then arrivals
        next_q1 = q1[t]
        next_q2 = q2[t]

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
def compute_all_results() -> Tuple[Scenario, Dict[str, SimulationResult]]:
    s = get_scenario()
    common_path = generate_common_sample_path(s)
    results = {
        name: simulate_policy(name, fn, s, common_path)
        for name, fn in POLICIES.items()
    }
    return s, results


# =========================================================
# Plotting: left big queue/server figure
# =========================================================

def draw_system_figure(
    result: SimulationResult,
    t: int,
    scenario: Scenario,
    policy_name: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    q1_now = int(result.q1[t])
    q2_now = int(result.q2[t])
    a = int(result.action[t]) if t < len(result.action) else 0
    inst = float(result.inst_cost[t]) if t < len(result.inst_cost) else 0.0
    cum = float(result.cum_disc_cost[t]) if t < len(result.cum_disc_cost) else 0.0

    # Queue boxes
    q_width = 2.2
    q_max_height = 5.5
    box_x = 1.0
    q1_y = 4.0
    q2_y = 0.8

    max_vis = max(
        1,
        np.max(result.q1[: len(result.action) + 1]),
        np.max(result.q2[: len(result.action) + 1]),
    )

    h1 = q_max_height * min(q1_now / max_vis, 1.0)
    h2 = q_max_height * min(q2_now / max_vis, 1.0)

    ax.add_patch(Rectangle((box_x, q1_y), q_width, q_max_height, fill=False, linewidth=2))
    ax.add_patch(Rectangle((box_x, q2_y), q_width, q_max_height, fill=False, linewidth=2))

    ax.add_patch(Rectangle((box_x, q1_y), q_width, h1, alpha=0.55))
    ax.add_patch(Rectangle((box_x, q2_y), q_width, h2, alpha=0.55))

    ax.text(box_x + q_width / 2, q1_y + q_max_height + 0.3, "Queue 1", ha="center", fontsize=12)
    ax.text(box_x + q_width / 2, q2_y + q_max_height + 0.3, "Queue 2", ha="center", fontsize=12)

    ax.text(box_x + q_width / 2, q1_y + q_max_height / 2, f"$X_1={q1_now}$", ha="center", va="center", fontsize=13)
    ax.text(box_x + q_width / 2, q2_y + q_max_height / 2, f"$X_2={q2_now}$", ha="center", va="center", fontsize=13)

    # Server
    server_x = 6.2
    server_y = 3.6
    server_w = 2.0
    server_h = 2.0

    server_color = "#DDDDDD"
    if a == 1:
        server_color = "#ffb3b3"
    elif a == 2:
        server_color = "#b3d9ff"

    ax.add_patch(Rectangle((server_x, server_y), server_w, server_h, facecolor=server_color, edgecolor="black", linewidth=2))
    ax.text(server_x + server_w / 2, server_y + server_h / 2, "Server", ha="center", va="center", fontsize=13)

    # Arrows
    q1_arrow_color = "black"
    q2_arrow_color = "black"
    if a == 1:
        q1_arrow_color = "red"
    elif a == 2:
        q2_arrow_color = "blue"

    ax.add_patch(
        FancyArrowPatch(
            (box_x + q_width, q1_y + q_max_height / 2),
            (server_x, server_y + 1.4),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.5,
            color=q1_arrow_color,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (box_x + q_width, q2_y + q_max_height / 2),
            (server_x, server_y + 0.6),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.5,
            color=q2_arrow_color,
        )
    )

    # Policy stats
    mu_c1 = scenario.mu1 * scenario.c1
    mu_c2 = scenario.mu2 * scenario.c2

    if a == 1:
        serve_txt = "Serving Queue 1"
        serve_color = "red"
    elif a == 2:
        serve_txt = "Serving Queue 2"
        serve_color = "blue"
    else:
        serve_txt = "Idle"
        serve_color = "black"

    ax.text(0.2, 9.6, f"Policy: {policy_name}", fontsize=13, weight="bold")
    ax.text(0.2, 9.1, f"Time step: {t}", fontsize=12)
    ax.text(0.2, 8.6, serve_txt, fontsize=12, color=serve_color, weight="bold")
    ax.text(0.2, 8.1, rf"$\mu_1c_1={mu_c1:.2f}$,  $\mu_2c_2={mu_c2:.2f}$", fontsize=12)
    ax.text(0.2, 7.6, f"Instantaneous cost: {inst:.2f}", fontsize=12)
    ax.text(0.2, 7.1, f"Cumulative discounted cost: {cum:.2f}", fontsize=12)

    fig.tight_layout()
    return fig


# =========================================================
# Plotting: right-side cost panels
# =========================================================

def draw_cost_grid(results: Dict[str, SimulationResult], t: int, selected_policy: str) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flatten()

    names = list(results.keys())

    for ax, name in zip(axes, names):
        res = results[name]
        x = res.times[: t + 1]
        y = res.cum_disc_cost[: t + 1]

        ax.plot(x, y, linewidth=2)
        ax.set_title(name, fontsize=10, weight="bold" if name == selected_policy else None)

        if name == selected_policy:
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)

        ax.set_xlim(0, len(res.times) - 1)
        ax.set_xlabel("time", fontsize=9)
        ax.set_ylabel("cum. cost", fontsize=9)
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
# UI
# =========================================================

def scenario_summary(s: Scenario) -> str:
    return (
        f"Fixed scenario: "
        f"λ₁={s.lambda1}, λ₂={s.lambda2}, "
        f"μ₁={s.mu1}, μ₂={s.mu2}, "
        f"c₁={s.c1}, c₂={s.c2}, "
        f"α={s.discount_alpha}, horizon={s.horizon}"
    )


def main() -> None:
    st.title(TITLE)
    st.caption(SUBTITLE)

    scenario, results = compute_all_results()

    with st.expander("Scenario", expanded=False):
        st.write(scenario_summary(scenario))
        st.write(
            "All policies are evaluated on the same arrivals and the same service-completion sample path."
        )

    top_controls = st.columns([1.2, 1, 1])
    with top_controls[0]:
        selected_policy = st.selectbox(
            "Policy shown on the left",
            list(results.keys()),
            index=0,
        )
    with top_controls[1]:
        autoplay = st.checkbox("Autoplay", value=False)
    with top_controls[2]:
        speed = st.selectbox("Speed", ["Slow", "Medium", "Fast"], index=1)

    max_t = scenario.horizon - 1

    # persistent time
    if "timestep" not in st.session_state:
        st.session_state.timestep = 0

    # manual slider
    t = st.slider("Time", 0, max_t, st.session_state.timestep, 1)
    st.session_state.timestep = t

    left, right = st.columns([1.05, 1.15])

    with left:
        st.subheader("System behavior")
        fig_left = draw_system_figure(
            results[selected_policy],
            st.session_state.timestep,
            scenario,
            selected_policy,
        )
        st.pyplot(fig_left, clear_figure=True)

    with right:
        st.subheader("Policy comparison")
        fig_right = draw_cost_grid(results, st.session_state.timestep, selected_policy)
        st.pyplot(fig_right, clear_figure=True)

        final_costs = {
            name: res.cum_disc_cost[-1]
            for name, res in results.items()
        }
        ranking = sorted(final_costs.items(), key=lambda kv: kv[1])
        st.markdown("**Final discounted cost ranking**")
        for i, (name, value) in enumerate(ranking, start=1):
            st.write(f"{i}. {name}: {value:.2f}")

    if autoplay and st.session_state.timestep < max_t:
        delay = {"Slow": 0.55, "Medium": 0.25, "Fast": 0.08}[speed]
        time.sleep(delay)
        st.session_state.timestep += 1
        st.rerun()


if __name__ == "__main__":
    main()