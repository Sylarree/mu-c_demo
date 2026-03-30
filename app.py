import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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
    mean_final_cost: float
    std_final_cost: float


# =========================================================
# Default scenario
# =========================================================

def get_default_scenario() -> Scenario:
    return Scenario(
        horizon=200,
        lambda1=0.48,
        lambda2=0.58,
        mu1=0.95,
        mu2=0.45,
        c1=3.0,
        c2=4.0,
        discount_alpha=0.992,
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
    random_policy_rng: Optional[np.random.Generator] = None,
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
def compute_all_results(
    scenario: Scenario,
    n_replications: int = 200
) -> Tuple[Scenario, Dict[str, SimulationResult], Dict[str, AggregateResult]]:
    s = scenario

    # Representative run for animations
    common_path_demo = generate_common_sample_path(s)
    demo_results: Dict[str, SimulationResult] = {}
    for name, fn in POLICIES.items():
        rng = np.random.default_rng(123) if name == "Random" else None
        demo_results[name] = simulate_policy(name, fn, s, common_path_demo, rng)

    # Mean final discounted cost over many replications
    aggregate_results: Dict[str, AggregateResult] = {}
    for name, fn in POLICIES.items():
        final_costs = []

        for rep in range(n_replications):
            rep_s = Scenario(**{**s.__dict__, "seed": s.seed + rep})
            common_path = generate_common_sample_path(rep_s)
            rng = np.random.default_rng(10000 + rep) if name == "Random" else None
            res = simulate_policy(name, fn, rep_s, common_path, rng)
            final_costs.append(res.cum_disc_cost[-1])

        final_costs = np.array(final_costs)
        aggregate_results[name] = AggregateResult(
            mean_final_cost=float(np.mean(final_costs)),
            std_final_cost=float(np.std(final_costs)),
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


def detect_preemption(result: SimulationResult, t: int) -> Optional[str]:
    """
    Simple visual preemption detector:
    if the served queue changes from t-1 to t while the previously served queue
    still has pending jobs at time t, label that as a preemption.
    """
    if t <= 0:
        return None

    prev_a = int(result.action[t - 1])
    curr_a = int(result.action[t])

    if prev_a == 0 or curr_a == 0 or prev_a == curr_a:
        return None

    prev_queue_remaining = result.q1[t] if prev_a == 1 else result.q2[t]

    if prev_queue_remaining > 0:
        return f"Preemptive switch: Q{prev_a} → Q{curr_a}"

    return None


def draw_mini_system_panel(
    ax,
    result: SimulationResult,
    t: int,
    scenario: Scenario,
    policy_name: str,
):
    t = min(t, len(result.action) - 1)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
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
    q1_y = 4.2
    q2_y = 2.6

    # Outline bars
    ax.add_patch(Rectangle((q_x, q1_y), q_w, q_h, fill=False, linewidth=1.8))
    ax.add_patch(Rectangle((q_x, q2_y), q_w, q_h, fill=False, linewidth=1.8))

    # Fills
    fill1 = q_w * min(q1_now / max_vis, 1.0)
    fill2 = q_w * min(q2_now / max_vis, 1.0)

    ax.add_patch(Rectangle((q_x, q1_y), fill1, q_h, alpha=0.55))
    ax.add_patch(Rectangle((q_x, q2_y), fill2, q_h, alpha=0.55))

    # Labels
    ax.text(q_x - 0.15, q1_y + q_h / 2, "Q1", ha="right", va="center", fontsize=10)
    ax.text(q_x - 0.15, q2_y + q_h / 2, "Q2", ha="right", va="center", fontsize=10)

    # Queue counts moved away from arrows
    ax.text(q_x + q_w + 0.10, q1_y + q_h + 0.28, f"{q1_now}", ha="left", va="center", fontsize=10)
    ax.text(q_x + q_w + 0.10, q2_y - 0.22, f"{q2_now}", ha="left", va="center", fontsize=10)

    # Server
    server_x = 7.1
    server_y = 3.1
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
    ax.text(0.1, 6.45, policy_name, fontsize=11, weight="bold")
    ax.text(0.1, 5.95, action_text(a), fontsize=10, color=action_color(a), weight="bold")

    if policy_name == "μc-rule":
        ax.text(
            0.1, 5.45,
            rf"$\mu_1c_1={scenario.mu1*scenario.c1:.2f},\ \mu_2c_2={scenario.mu2*scenario.c2:.2f}$",
            fontsize=9
        )

    # Preemption banner
    preempt_msg = detect_preemption(result, t)
    if preempt_msg is not None:
        banner_x = 0.35
        banner_y = 1.05
        banner_w = 8.9
        banner_h = 0.55
        ax.add_patch(
            Rectangle(
                (banner_x, banner_y),
                banner_w,
                banner_h,
                facecolor="#fff3a6",
                edgecolor="#c9b400",
                linewidth=1.2,
            )
        )
        ax.text(
            banner_x + banner_w / 2,
            banner_y + banner_h / 2,
            preempt_msg,
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            weight="bold",
        )


def draw_system_grid(
    demo_results: Dict[str, SimulationResult],
    t: int,
    scenario: Scenario,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.3))
    axes = axes.flatten()

    for ax, name in zip(axes, demo_results.keys()):
        draw_mini_system_panel(ax, demo_results[name], t, scenario, name)

    fig.tight_layout()
    return fig


def draw_cost_bar_chart(
    aggregate_results: Dict[str, AggregateResult],
) -> plt.Figure:
    names = list(aggregate_results.keys())
    means = [aggregate_results[name].mean_final_cost for name in names]
    stds = [aggregate_results[name].std_final_cost for name in names]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.bar(names, means, yerr=stds, capsize=6, alpha=0.8)

    ax.set_ylabel("mean final discounted cost")
    ax.set_title("Mean final discounted cost by policy")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    return fig


# =========================================================
# UI helpers
# =========================================================

def scenario_summary(s: Scenario) -> str:
    return (
        f"Current scenario: "
        f"λ₁={s.lambda1}, λ₂={s.lambda2}, "
        f"μ₁={s.mu1}, μ₂={s.mu2}, "
        f"c₁={s.c1}, c₂={s.c2}, "
        f"α={s.discount_alpha}, horizon={s.horizon}, seed={s.seed}"
    )


# =========================================================
# Main app
# =========================================================

def main() -> None:
    st.title(TITLE)
    st.caption(SUBTITLE)

    default_scenario = get_default_scenario()

    # -----------------------------------------------------
    # Editable parameters row
    # -----------------------------------------------------
    param_cols = st.columns(9)
    with param_cols[0]:
        lambda1 = st.number_input("λ1", value=float(default_scenario.lambda1), step=0.01, format="%.3f")
    with param_cols[1]:
        lambda2 = st.number_input("λ2", value=float(default_scenario.lambda2), step=0.01, format="%.3f")
    with param_cols[2]:
        mu1 = st.number_input("μ1", value=float(default_scenario.mu1), step=0.01, format="%.3f")
    with param_cols[3]:
        mu2 = st.number_input("μ2", value=float(default_scenario.mu2), step=0.01, format="%.3f")
    with param_cols[4]:
        c1 = st.number_input("c1", value=float(default_scenario.c1), step=0.1, format="%.3f")
    with param_cols[5]:
        c2 = st.number_input("c2", value=float(default_scenario.c2), step=0.1, format="%.3f")
    with param_cols[6]:
        alpha = st.number_input("α", value=float(default_scenario.discount_alpha), step=0.001, format="%.3f")
    with param_cols[7]:
        horizon = st.number_input("Horizon", value=int(default_scenario.horizon), step=10, min_value=1)
    with param_cols[8]:
        seed = st.number_input("Seed", value=int(default_scenario.seed), step=1, min_value=0)

    scenario = Scenario(
        horizon=int(horizon),
        lambda1=float(lambda1),
        lambda2=float(lambda2),
        mu1=float(mu1),
        mu2=float(mu2),
        c1=float(c1),
        c2=float(c2),
        discount_alpha=float(alpha),
        seed=int(seed),
    )

    scenario, demo_results, aggregate_results = compute_all_results(scenario, n_replications=200)

    with st.expander("Scenario", expanded=False):
        st.write(scenario_summary(scenario))
        st.write(
            "Top: one representative sample path for each policy. "
            "Bottom: mean final discounted cost over 200 replications."
        )
        st.write(
            "Yellow banner = a preemption-style switch: the served queue changed while the previously served queue still had pending work."
        )

    controls = st.columns([1, 1])
    with controls[0]:
        autoplay = st.checkbox("Autoplay", value=False)
    with controls[1]:
        speed = st.selectbox("Speed", ["Slow", "Medium", "Fast"], index=2)

    max_t = scenario.horizon - 1

    if "timestep" not in st.session_state:
        st.session_state.timestep = 0

    st.session_state.timestep = min(st.session_state.timestep, max_t)

    t = st.slider("Time", 0, max_t, st.session_state.timestep, 1)
    st.session_state.timestep = t

    st.subheader("System behavior")
    fig_systems = draw_system_grid(demo_results, st.session_state.timestep, scenario)
    st.pyplot(fig_systems, clear_figure=True)

    title_col, note_col = st.columns([1.2, 2.3])
    with title_col:
        st.subheader("Policy comparison")
    with note_col:
        st.markdown(
            "<div style='padding-top: 12px; color: gray;'>"
            "(mean final discounted cost over 200 replications)"
            "</div>",
            unsafe_allow_html=True,
        )

    fig_costs = draw_cost_bar_chart(aggregate_results)
    st.pyplot(fig_costs, clear_figure=True)

    if autoplay and st.session_state.timestep < max_t:
        step_jump = {"Slow": 1, "Medium": 3, "Fast": 6}[speed]
        delay = {"Slow": 0.40, "Medium": 0.10, "Fast": 0.02}[speed]
        time.sleep(delay)
        st.session_state.timestep = min(max_t, st.session_state.timestep + step_jump)
        st.rerun()


if __name__ == "__main__":
    main()