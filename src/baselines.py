from __future__ import annotations

from dataclasses import dataclass


MVP_POLICIES = ["random", "myopic", "entropy", "du_net"]

BANDIT_POLICIES = [
    "linear_ucb",
    "oful",
    "low_rank_bandit",
    "thompson_sampling",
    "logistic_ucb",
    "logistic_thompson_sampling",
    "finite_ucb",
    "epsilon_greedy",
    "adaptive_greedy",
    "explore_then_exploit",
]

NETWORK_BASELINE_POLICIES = [
    "graph_social",
    "social_neighbor_item",
    "social_mf_hybrid",
]

ABLATION_POLICIES = [
    "du_net",
    "du_net_no_dls",
    "du_net_no_entropy",
    "du_net_no_reward",
    "du_net_no_network",
    "du_net_static_sigma",
    "du_net_dynamic_sigma",
]

PAPER_BENCHMARK_TARGET_COUNT = 16


@dataclass(frozen=True)
class PaperBenchmarkSpec:
    family: str
    paper_name: str
    current_policy: str | None
    implementation_status: str
    note: str


PAPER_BENCHMARKS_EXPLICIT = [
    PaperBenchmarkSpec(
        family="sampling",
        paper_name="Uniform random sampling (URS)",
        current_policy="random",
        implementation_status="implemented",
        note="Implemented as uniform random sampling from the candidate acquisition set.",
    ),
    PaperBenchmarkSpec(
        family="sampling",
        paper_name="Myopic policy",
        current_policy="myopic",
        implementation_status="implemented",
        note="Implemented as reward-only acquisition using predicted ratings.",
    ),
    PaperBenchmarkSpec(
        family="sampling",
        paper_name="Information-theoretic / maximum entropy policy",
        current_policy="entropy",
        implementation_status="proxy",
        note="Implemented with coverage/subspace/log-det/network entropy variants; exact paper entropy remains partial.",
    ),
    PaperBenchmarkSpec(
        family="low-rank bandit",
        paper_name="Explore Subspace Then Refine (ESTR)",
        current_policy="low_rank_bandit",
        implementation_status="proxy",
        note="Current low-rank bandit is a scalable count/MF proxy, not the full ESTR adaptation.",
    ),
    PaperBenchmarkSpec(
        family="online linear bandit",
        paper_name="LinUCB",
        current_policy="linear_ucb",
        implementation_status="proxy",
        note="Implemented with interaction-feature ridge UCB; MF-centered pilots are stronger but remain proxy rows.",
    ),
    PaperBenchmarkSpec(
        family="online linear bandit",
        paper_name="OFUL",
        current_policy="oful",
        implementation_status="implemented",
        note="Implemented as a separate ridge-linear OFUL score with a configurable confidence radius.",
    ),
    PaperBenchmarkSpec(
        family="online linear bandit",
        paper_name="Linear Thompson Sampling",
        current_policy="thompson_sampling",
        implementation_status="proxy",
        note="Current Thompson row is a simplified noisy-count/MF proxy, not a linear posterior sampler.",
    ),
    PaperBenchmarkSpec(
        family="online linear bandit",
        paper_name="Logistic UCB",
        current_policy="logistic_ucb",
        implementation_status="proxy",
        note="Implemented as a logistic transform of MF reward with linear-feature UCB uncertainty; not the exact paper estimator.",
    ),
    PaperBenchmarkSpec(
        family="online linear bandit",
        paper_name="Logistic Thompson Sampling",
        current_policy="logistic_thompson_sampling",
        implementation_status="proxy",
        note="Implemented as logistic MF reward with uncertainty-scaled Gaussian sampling; not a full logistic posterior sampler.",
    ),
    PaperBenchmarkSpec(
        family="stochastic finite-arm bandit",
        paper_name="Upper Confidence Bound",
        current_policy="finite_ucb",
        implementation_status="proxy",
        note="Implemented with empirical user/item finite-arm means and count uncertainty; exact repeated-arm UCB is not directly available in this acquisition pool.",
    ),
    PaperBenchmarkSpec(
        family="stochastic finite-arm bandit",
        paper_name="Thompson Sampling",
        current_policy="thompson_sampling",
        implementation_status="proxy",
        note="Shares the current simplified Thompson proxy; finite-arm posterior behavior is not reproduced.",
    ),
    PaperBenchmarkSpec(
        family="stochastic finite-arm bandit",
        paper_name="Epsilon Greedy",
        current_policy="epsilon_greedy",
        implementation_status="proxy",
        note="Implemented as empirical finite-arm greedy selection with an epsilon random-exploration share.",
    ),
    PaperBenchmarkSpec(
        family="stochastic finite-arm bandit",
        paper_name="Adaptive Greedy",
        current_policy="adaptive_greedy",
        implementation_status="proxy",
        note="Implemented as empirical finite-arm greedy selection with a decaying epsilon floor.",
    ),
    PaperBenchmarkSpec(
        family="stochastic finite-arm bandit",
        paper_name="Explore-Then-Exploit",
        current_policy="explore_then_exploit",
        implementation_status="proxy",
        note="Implemented as random finite-arm exploration for an initial cycle fraction followed by empirical greedy exploitation.",
    ),
]

IMPLEMENTED_EXTRA_BASELINE_POLICIES = [
    "graph_social",
    "mf_linear_ucb",
    "mf_oful",
    "mf_low_rank_ucb",
    "mf_thompson_sampling",
    "mf_residual_ucb",
    "mf_residual_thompson_sampling",
    "prediction_uncertainty",
    "leverage_uncertainty",
    "active_learning_hybrid",
    "social_neighbor_item",
    "social_mf_hybrid",
]


def unresolved_paper_benchmark_slots() -> int:
    return max(PAPER_BENCHMARK_TARGET_COUNT - len(PAPER_BENCHMARKS_EXPLICIT), 0)


def missing_explicit_paper_benchmarks() -> list[PaperBenchmarkSpec]:
    return [spec for spec in PAPER_BENCHMARKS_EXPLICIT if spec.implementation_status == "missing"]
