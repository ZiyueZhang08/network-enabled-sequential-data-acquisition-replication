from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.acquisition import (
    AcquisitionConfig,
    _adaptive_grids_for_cycle,
    _du_net_combined_score,
    _select_batch_composition,
    _select_network_batch_composition,
    _select_rows,
    _select_rows_with_diversity_cap,
    _social_neighbor_item_scores,
    _transform_entropy_score,
    run_policy,
)
from src.data_loader import (
    generate_synthetic_dataset,
    load_gcmc_mat,
    load_movielens_100k,
    split_dataset,
)
from src.baselines import (
    PAPER_BENCHMARKS_EXPLICIT,
    PAPER_BENCHMARK_TARGET_COUNT,
    missing_explicit_paper_benchmarks,
    unresolved_paper_benchmark_slots,
)
from src.dls import shrink_dls_user_factors
from src.dls_paper import full_gaussian_dls_user_factors, paper_gaussian_dls_user_factors, scoped_gaussian_dls_user_factors
from src.entropy import (
    coherence_lite_entropy_scores,
    network_observational_entropy_scores,
    network_observational_expected_entropy_scores,
    network_observational_reliability_stochastic_entropy_scores,
    network_observational_stochastic_entropy_scores,
    network_observational_weighted_entropy_scores,
    network_neighbor_prediction_gain_entropy_scores,
    observational_logdet_entropy_scores,
)
from src.entropy_paper import entropy_coherence_base_scores, entropy_observational_lite_scores
from src.entropy_schur import (
    explicit_kernel_schur_conditional_variance_entropy_scores,
    schur_conditional_variance_entropy_scores,
)
from src.graph_embedding import user_embedding
from src.graph_transforms import transform_user_graph
from src.matrix_factorization import MFConfig, MatrixFactorization
from src.metrics import (
    cumulative_reward,
    linear_reward_norm_weighted_rmse,
    pdf_literal_norm_rmse,
    reward_tail_proxy_rmse,
    reward_weighted_rmse,
    rmse,
    sigmoid_reward_weighted_rmse,
    standard_mask_rmse,
)
from src.reward import network_reward_scores
from src.tuning import _select_inner_batch


ROOT = Path(__file__).resolve().parents[1]


class ConstantModel:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def predict_pairs(self, users: np.ndarray, items: np.ndarray, user_factors=None) -> np.ndarray:
        return self.values[users].astype(float)


class LoaderTests(unittest.TestCase):
    def test_movielens_loader_shape(self) -> None:
        dataset = load_movielens_100k(ROOT / "data" / "raw" / "ml-100k")
        self.assertEqual(dataset.num_users, 943)
        self.assertEqual(dataset.num_items, 1682)
        self.assertEqual(len(dataset.ratings), 100000)
        self.assertEqual(dataset.adjacency.shape, (943, 943))
        self.assertTrue(dataset.adjacency.any())
        self.assertIsNotNone(dataset.user_features)
        self.assertEqual(dataset.user_features.shape[0], dataset.num_users)

    def test_movielens_gcmc_feature_network_mode(self) -> None:
        current = load_movielens_100k(ROOT / "data" / "raw" / "ml-100k")
        gcmc = load_movielens_100k(ROOT / "data" / "raw" / "ml-100k", network_mode="gcmc_features")
        self.assertEqual(gcmc.num_users, current.num_users)
        self.assertEqual(gcmc.num_items, current.num_items)
        self.assertEqual(gcmc.adjacency.shape, current.adjacency.shape)
        self.assertIsNotNone(gcmc.user_features)
        self.assertEqual(gcmc.user_features.shape[0], gcmc.num_users)
        self.assertTrue(gcmc.adjacency.any())
        self.assertFalse(np.array_equal(gcmc.adjacency, current.adjacency))

    def test_gcmc_loader_shapes_and_predefined_split(self) -> None:
        flixster = load_gcmc_mat(ROOT / "data" / "raw" / "gcmc" / "flixster", "flixster")
        self.assertEqual(flixster.num_users, 2341)
        self.assertEqual(flixster.num_items, 2956)
        self.assertEqual(len(flixster.train_ratings), 23556)
        self.assertEqual(len(flixster.test_ratings), 2617)

        split = split_dataset(
            flixster,
            seed=0,
            test_ratio=0.0,
            initial_fraction_of_train=0.01,
            use_predefined_split=True,
        )
        self.assertEqual(len(split.initial) + len(split.candidate), len(flixster.train_ratings))
        self.assertEqual(len(split.test), len(flixster.test_ratings))

        douban = load_gcmc_mat(ROOT / "data" / "raw" / "gcmc" / "douban", "douban")
        self.assertEqual(douban.num_users, 2999)
        self.assertEqual(douban.num_items, 3000)
        self.assertEqual(len(douban.train_ratings), 123202)
        self.assertEqual(len(douban.test_ratings), 13689)


class BaselineCatalogTests(unittest.TestCase):
    def test_paper_benchmark_catalog_keeps_unresolved_slots_visible(self) -> None:
        self.assertEqual(PAPER_BENCHMARK_TARGET_COUNT, 16)
        self.assertEqual(len(PAPER_BENCHMARKS_EXPLICIT), 14)
        self.assertEqual(unresolved_paper_benchmark_slots(), 2)

        missing_names = {spec.paper_name for spec in missing_explicit_paper_benchmarks()}
        self.assertNotIn("OFUL", missing_names)
        self.assertNotIn("Logistic UCB", missing_names)
        self.assertNotIn("Explore-Then-Exploit", missing_names)
        self.assertEqual(len(missing_names), 0)


class AcquisitionTests(unittest.TestCase):
    def test_sigmoid_reward_weighted_rmse_emphasizes_high_rewards(self) -> None:
        class ToyMetricModel:
            def predict_pairs(self, users: np.ndarray, items: np.ndarray, user_factors=None) -> np.ndarray:
                return np.array([1.0, 4.0])

        ratings = pd.DataFrame(
            {
                "user_id": [0, 1],
                "item_id": [0, 0],
                "rating": [1.0, 5.0],
            }
        )
        model = ToyMetricModel()
        self.assertGreater(
            sigmoid_reward_weighted_rmse(model, ratings, midpoint=3.0),
            rmse(model, ratings),
        )

    def test_pdf_literal_standard_and_tail_proxy_metrics_are_separate(self) -> None:
        class ToyMetricModel:
            def predict_pairs(self, users: np.ndarray, items: np.ndarray, user_factors=None) -> np.ndarray:
                return np.array([2.0, 3.0, 4.0])

        ratings = pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "item_id": [0, 0, 0],
                "rating": [1.0, 5.0, 4.0],
            }
        )
        model = ToyMetricModel()
        errors = np.array([-1.0, 2.0, 0.0])
        self.assertAlmostEqual(pdf_literal_norm_rmse(model, ratings), float(np.sqrt(np.sum(errors * errors)) / 3.0))
        self.assertAlmostEqual(standard_mask_rmse(model, ratings), float(np.sqrt(np.mean(errors * errors))))
        self.assertAlmostEqual(reward_tail_proxy_rmse(model, ratings), linear_reward_norm_weighted_rmse(model, ratings))
        expected_reward = np.sum(1.0 / (1.0 + np.exp(-(ratings["rating"].to_numpy(dtype=float) - 3.0))))
        self.assertAlmostEqual(cumulative_reward(ratings), float(expected_reward))

    def test_linear_reward_norm_weighted_rmse_uses_weights_inside_norm(self) -> None:
        class ToyMetricModel:
            def predict_pairs(self, users: np.ndarray, items: np.ndarray, user_factors=None) -> np.ndarray:
                return np.array([1.0, 4.0])

        ratings = pd.DataFrame(
            {
                "user_id": [0, 1],
                "item_id": [0, 0],
                "rating": [1.0, 5.0],
            }
        )
        model = ToyMetricModel()
        expected = np.sqrt(((0.1 * 0.0) ** 2 + (2.1 * 1.0) ** 2) / 2.0)
        self.assertAlmostEqual(
            linear_reward_norm_weighted_rmse(model, ratings, midpoint=3.0),
            expected,
        )

    def test_reward_weighted_rmse_beta_uses_standard_weighted_mean(self) -> None:
        class ToyMetricModel:
            def predict_pairs(self, users: np.ndarray, items: np.ndarray, user_factors=None) -> np.ndarray:
                return np.array([2.0, 4.0])

        ratings = pd.DataFrame(
            {
                "user_id": [0, 1],
                "item_id": [0, 0],
                "rating": [1.0, 5.0],
            }
        )
        model = ToyMetricModel()
        weights = np.array([1.0, 2.0])
        expected = np.sqrt(np.sum(weights * np.array([1.0, 1.0]) ** 2) / np.sum(weights))
        self.assertAlmostEqual(
            reward_weighted_rmse(model, ratings, midpoint=3.0, beta=0.5),
            expected,
        )

    def test_acquisition_loop_respects_batch_size(self) -> None:
        dataset = generate_synthetic_dataset(seed=1, num_users=30, num_items=24, observed_fraction=0.35)
        split = split_dataset(dataset, seed=2, test_ratio=0.2, initial_fraction_of_train=0.1)
        config = AcquisitionConfig(cycles=3, batch_size=5, selection="random")
        metrics = run_policy(
            policy="random",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=3, epochs=1, algorithm="mf"),
            acquisition_config=config,
            seed=3,
        )
        observed_sizes = metrics["observed_size"].to_numpy(dtype=int)
        self.assertEqual(list(np.diff(observed_sizes)), [5, 5, 5])

    def test_raw_prediction_and_row_level_sink_are_available(self) -> None:
        model = MatrixFactorization(1, 1, MFConfig(rank=1, epochs=0, algorithm="mf"), seed=0)
        model.global_mean = 10.0
        model.user_factors[:] = 0.0
        model.item_factors[:] = 0.0
        users = np.array([0])
        items = np.array([0])
        self.assertEqual(float(model.predict_pairs_raw(users, items)[0]), 10.0)
        self.assertEqual(float(model.predict_pairs(users, items)[0]), 5.0)

        dataset = generate_synthetic_dataset(seed=4, num_users=18, num_items=14, observed_fraction=0.45)
        split = split_dataset(dataset, seed=5, test_ratio=0.2, initial_fraction_of_train=0.2)
        sink: list[pd.DataFrame] = []
        run_policy(
            policy="random",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=2, epochs=1, algorithm="mf"),
            acquisition_config=AcquisitionConfig(cycles=1, batch_size=3, selection="random"),
            seed=6,
            prediction_sink=sink,
            prediction_cycles=None,
        )
        predictions = pd.concat(sink, ignore_index=True)
        for column in [
            "method",
            "cycle",
            "user_id",
            "item_id",
            "y_true",
            "y_pred_clipped",
            "y_pred_raw",
            "err_raw",
            "err_clipped",
            "selected_or_test",
            "reward",
            "item_popularity",
            "user_degree",
            "test_mask_id",
            "test_weight_candidates",
        ]:
            self.assertIn(column, predictions.columns)
        self.assertTrue({"test", "selected"}.issubset(set(predictions["selected_or_test"])))
        self.assertTrue(((predictions["y_pred_clipped"] >= 1.0) & (predictions["y_pred_clipped"] <= 5.0)).all())
        np.testing.assert_allclose(predictions["err_raw"], predictions["y_true"] - predictions["y_pred_raw"])
        np.testing.assert_allclose(predictions["err_clipped"], predictions["y_true"] - predictions["y_pred_clipped"])

    def test_topk_and_temperature_exp_selection(self) -> None:
        scores = np.array([1.0, 4.0, 2.0, 5.0, 3.0])
        rng = np.random.default_rng(0)
        topk = _select_rows(scores, batch_size=2, rng=rng, mode="topk")
        self.assertEqual(set(topk.tolist()), {1, 3})

        exp = _select_rows(scores, batch_size=3, rng=np.random.default_rng(1), mode="exp", temperature=0.25)
        self.assertEqual(len(exp), 3)
        self.assertEqual(len(set(exp.tolist())), 3)
        self.assertTrue(np.all((0 <= exp) & (exp < len(scores))))

        topm_exp = _select_rows(
            np.arange(10, dtype=float),
            batch_size=3,
            rng=np.random.default_rng(2),
            mode="top_m_exp",
            temperature=0.1,
            top_m=4,
        )
        self.assertEqual(len(topm_exp), 3)
        self.assertTrue(set(topm_exp.tolist()).issubset({6, 7, 8, 9}))

    def test_paper_tuning_inner_selection_can_use_deterministic_topk(self) -> None:
        scores = np.array([0.0, 2.0, 1.0, 4.0])
        selected = _select_inner_batch(scores, 2, np.random.default_rng(1), temperature=1.0, mode="topk")
        self.assertEqual(set(selected.tolist()), {1, 3})

    def test_adaptive_stage_grid_selects_cycle_specific_candidates(self) -> None:
        config = AcquisitionConfig(
            adaptive_lambda_grid=(0.1,),
            sigma_grid=(0.4,),
            adaptive_diffusion_prob_grid=(0.005,),
            adaptive_dls_shrinkage_grid=(0.25,),
            adaptive_stage_grids=(
                {
                    "start_cycle": 1,
                    "end_cycle": 5,
                    "lambda_grid": [0.25, 0.5, 0.75],
                    "sigma_grid": [0.2, 0.4, 0.8],
                    "alpha_grid": [0.75],
                },
                {"start_cycle": 6, "end_cycle": 15, "lambda_grid": [0.125, 0.25, 0.5]},
                {"start_cycle": 16, "end_cycle": 20, "lambda_grid": [0.05, 0.125, 0.25], "sigma_grid": [0.1, 0.2, 0.4]},
            ),
        )
        early_lambda, early_sigma, _, early_dls = _adaptive_grids_for_cycle(config, cycle=0)
        mid_lambda, mid_sigma, _, mid_dls = _adaptive_grids_for_cycle(config, cycle=5)
        late_lambda, late_sigma, _, late_dls = _adaptive_grids_for_cycle(config, cycle=15)
        self.assertEqual(early_lambda, (0.25, 0.5, 0.75))
        self.assertEqual(early_sigma, (0.2, 0.4, 0.8))
        self.assertEqual(early_dls, (0.75,))
        self.assertEqual(mid_lambda, (0.125, 0.25, 0.5))
        self.assertEqual(mid_sigma, (0.4,))
        self.assertEqual(mid_dls, (0.25,))
        self.assertEqual(late_lambda, (0.05, 0.125, 0.25))
        self.assertEqual(late_sigma, (0.1, 0.2, 0.4))
        self.assertEqual(late_dls, (0.25,))

    def test_diversity_cap_selection_limits_then_fills(self) -> None:
        candidates = pd.DataFrame(
            {
                "user_id": [0, 0, 0, 1, 1, 2],
                "item_id": [0, 0, 1, 1, 2, 2],
                "rating": [5, 5, 5, 5, 5, 5],
            }
        )
        scores = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
        selected = _select_rows_with_diversity_cap(scores, candidates, batch_size=4, max_per_user=1, max_per_item=1)
        self.assertEqual(len(selected), 4)
        first_pass = selected[:3]
        self.assertEqual(len(set(candidates.iloc[first_pass]["user_id"].tolist())), 3)
        self.assertEqual(len(set(candidates.iloc[first_pass]["item_id"].tolist())), 3)

    def test_fixed_dls_shrinkage_is_residual_blend(self) -> None:
        raw = np.array([[1.0, 0.0], [0.0, 1.0]])
        smoothed = np.array([[0.0, 1.0], [1.0, 0.0]])
        effective = np.ones(2)
        degree = np.ones(2)
        mixed, rho = shrink_dls_user_factors(
            raw,
            smoothed,
            effective,
            degree,
            top_k=2,
            strength=0.25,
            mode="fixed",
        )
        np.testing.assert_allclose(rho, np.array([0.25, 0.25]))
        np.testing.assert_allclose(mixed, 0.75 * raw + 0.25 * smoothed)

    def test_paper_gaussian_dls_has_no_residual_blend(self) -> None:
        raw = np.array([[1.0, 0.0], [0.0, 1.0]])
        neighbor_idx = np.array([[1], [0]])
        neighbor_dist_sq = np.zeros((2, 1))
        smoothed, effective = paper_gaussian_dls_user_factors(raw, neighbor_idx, neighbor_dist_sq, sigma=0.4)
        np.testing.assert_allclose(smoothed, np.array([[0.0, 1.0], [1.0, 0.0]]))
        np.testing.assert_allclose(effective, np.ones(2))

    def test_scoped_paper_dls_full_uses_all_users(self) -> None:
        raw = np.eye(3)
        network_embedding = np.array([[0.0], [1.0], [2.0]])
        neighbor_idx = np.array([[1], [0], [1]])
        neighbor_dist_sq = np.ones((3, 1))
        scoped, scoped_effective = scoped_gaussian_dls_user_factors(
            raw,
            network_embedding,
            neighbor_idx,
            neighbor_dist_sq,
            sigma=0.8,
            scope="full",
        )
        full, full_effective = full_gaussian_dls_user_factors(raw, network_embedding, sigma=0.8)
        np.testing.assert_allclose(scoped, full)
        np.testing.assert_allclose(scoped_effective, full_effective)
        self.assertGreater(float(scoped[1, 1]), float(scoped[1, 0]))

    def test_entropy_score_transform_defaults_to_identity(self) -> None:
        values = np.array([-2.0, 0.0, 3.0])
        np.testing.assert_allclose(_transform_entropy_score(values, "none"), values)
        paper_log = _transform_entropy_score(np.array([0.0, 0.5, 1.0]), "paper_log")
        self.assertTrue(np.isfinite(paper_log).all())
        self.assertLess(float(paper_log[0]), float(paper_log[1]))
        self.assertLess(float(paper_log[1]), float(paper_log[2]))
        shifted = _transform_entropy_score(values, "shift_log1p")
        self.assertFalse(np.isnan(shifted).any())
        self.assertAlmostEqual(float(shifted[0]), 0.0)
        self.assertGreater(float(shifted[1] - shifted[0]), float(shifted[2] - shifted[1]))

    def test_coherence_lite_entropy_matches_product_and_penalizes_observed_support(self) -> None:
        candidates = pd.DataFrame(
            {
                "user_id": [0, 1],
                "item_id": [0, 1],
                "rating": [5.0, 4.0],
            }
        )
        observed = pd.DataFrame(
            {
                "user_id": [0, 0, 1],
                "item_id": [0, 1, 0],
                "rating": [5.0, 4.0, 3.0],
            }
        )
        user_factors = np.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]])
        item_factors = np.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.2]])
        raw = coherence_lite_entropy_scores(
            candidates,
            observed,
            3,
            3,
            user_factors,
            item_factors,
            observed_penalty_weight=0.0,
        )
        q_user, _ = np.linalg.qr(user_factors)
        q_item, _ = np.linalg.qr(item_factors)
        user_mu = np.sum(q_user * q_user, axis=1)
        item_mu = np.sum(q_item * q_item, axis=1)
        np.testing.assert_allclose(raw, np.array([user_mu[0] * item_mu[0], user_mu[1] * item_mu[1]]))

        penalized = coherence_lite_entropy_scores(
            candidates,
            observed,
            3,
            3,
            user_factors,
            item_factors,
            observed_penalty_weight=0.5,
        )
        self.assertLess(float(penalized[0]), float(raw[0]))
        self.assertLess(float(penalized[1]), float(raw[1]))

    def test_paper_entropy_lite_uses_coherence_and_observed_correction(self) -> None:
        candidates = pd.DataFrame({"user_id": [0, 1], "item_id": [0, 1], "rating": [5.0, 4.0]})
        observed = pd.DataFrame({"user_id": [0], "item_id": [0], "rating": [5.0]})
        user_factors = np.array([[1.0, 0.0], [0.0, 1.0]])
        item_factors = np.array([[1.0, 0.0], [0.0, 1.0]])
        base = entropy_coherence_base_scores(candidates, user_factors, item_factors)
        lite = entropy_observational_lite_scores(candidates, observed, user_factors, item_factors)
        no_correction = entropy_observational_lite_scores(
            candidates,
            observed,
            user_factors,
            item_factors,
            correction_weight=0.0,
        )
        np.testing.assert_allclose(base, np.ones(2))
        self.assertEqual(lite.shape, (2,))
        self.assertLess(float(lite[0]), float(base[0]))
        self.assertAlmostEqual(float(lite[1]), float(base[1]))
        np.testing.assert_allclose(no_correction, base)

    def test_schur_entropy_returns_full_conditional_variance(self) -> None:
        candidates = pd.DataFrame({"user_id": [0, 1], "item_id": [0, 1], "rating": [5.0, 4.0]})
        observed = pd.DataFrame({"user_id": [0], "item_id": [0], "rating": [5.0]})
        empty_observed = observed.iloc[[]].copy()
        user_factors = np.array([[1.0, 0.0], [0.0, 1.0]])
        item_factors = np.array([[1.0, 0.0], [0.0, 1.0]])
        base = entropy_coherence_base_scores(candidates, user_factors, item_factors)
        no_observed = schur_conditional_variance_entropy_scores(
            candidates,
            empty_observed,
            user_factors,
            item_factors,
            lambda_noise=1.0,
            shortlist_size=0,
        )
        conditioned = schur_conditional_variance_entropy_scores(
            candidates,
            observed,
            user_factors,
            item_factors,
            lambda_noise=1.0,
            shortlist_size=0,
        )
        self.assertEqual(conditioned.shape, (2,))
        self.assertTrue(np.all(np.isfinite(conditioned)))
        self.assertTrue(np.all(conditioned >= 0.0))
        np.testing.assert_allclose(no_observed, base)
        self.assertLess(float(conditioned[0]), float(no_observed[0]))
        self.assertAlmostEqual(float(conditioned[1]), float(no_observed[1]))

    def test_schur_entropy_shortlist_does_not_keep_uncorrected_base(self) -> None:
        candidates = pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "item_id": [0, 1, 2],
                "rating": [5.0, 4.0, 3.0],
            }
        )
        observed = pd.DataFrame({"user_id": [0], "item_id": [0], "rating": [5.0]})
        user_factors = np.eye(3)
        item_factors = np.eye(3)
        scores = schur_conditional_variance_entropy_scores(
            candidates,
            observed,
            user_factors,
            item_factors,
            lambda_noise=1.0,
            shortlist_size=1,
        )
        self.assertEqual(int(np.count_nonzero(scores)), 1)

    def test_schur_entropy_matches_explicit_kernel_schur_complement(self) -> None:
        candidates = pd.DataFrame(
            {
                "user_id": [0, 0, 1, 2],
                "item_id": [0, 2, 1, 2],
                "rating": [5.0, 4.0, 3.0, 2.0],
            }
        )
        observed = pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "item_id": [1, 0, 2],
                "rating": [4.0, 5.0, 2.0],
            }
        )
        user_factors = np.array(
            [
                [1.0, 0.2],
                [0.1, 1.0],
                [0.7, 0.4],
            ]
        )
        item_factors = np.array(
            [
                [0.9, 0.3],
                [0.2, 1.1],
                [0.6, 0.5],
            ]
        )
        feature_space = schur_conditional_variance_entropy_scores(
            candidates,
            observed,
            user_factors,
            item_factors,
            lambda_noise=0.75,
            shortlist_size=0,
            batch_size=2,
        )
        explicit_kernel = explicit_kernel_schur_conditional_variance_entropy_scores(
            candidates,
            observed,
            user_factors,
            item_factors,
            lambda_noise=0.75,
            batch_size=2,
        )
        np.testing.assert_allclose(feature_space, explicit_kernel, rtol=1e-10, atol=1e-10)

    def test_batch_composition_selection_takes_reward_and_entropy_blocks(self) -> None:
        candidates = pd.DataFrame(
            {
                "user_id": [0, 1, 2, 3, 4],
                "item_id": [0, 0, 0, 0, 0],
                "rating": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        observed = pd.DataFrame({"user_id": [0], "item_id": [0], "rating": [4.0]})
        scores = {
            "score": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            "reward_score": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            "entropy_score": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }
        config = AcquisitionConfig(
            batch_size=4,
            selection="batch_composition",
            batch_composition_reward_fraction=0.5,
            batch_composition_entropy_fraction=0.5,
        )
        selected = _select_batch_composition(scores, candidates, observed, 5, 1, 4, config)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(set(selected.tolist())), 4)
        self.assertTrue({0, 1}.issubset(set(selected.tolist())))
        self.assertTrue({3, 4}.issubset(set(selected.tolist())))

    def test_network_batch_composition_penalizes_redundant_social_neighbors(self) -> None:
        candidates = pd.DataFrame(
            {
                "user_id": [0, 1, 2, 3],
                "item_id": [0, 0, 1, 2],
                "rating": [5.0, 5.0, 4.0, 3.0],
            }
        )
        observed = pd.DataFrame({"user_id": [0], "item_id": [1], "rating": [4.0]})
        adjacency = np.array(
            [
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ]
        )
        scores = {
            "score": np.array([10.0, 1.0, 1.0, 1.0]),
            "reward_score": np.array([10.0, 1.0, 1.0, 1.0]),
            "entropy_score": np.array([1.0, 5.0, 4.9, 4.8]),
        }
        config = AcquisitionConfig(
            batch_size=3,
            selection="network_batch_composition",
            batch_composition_reward_fraction=1 / 3,
            batch_composition_entropy_fraction=2 / 3,
            batch_composition_network_penalty=1.0,
        )
        selected = _select_network_batch_composition(scores, candidates, observed, adjacency, 4, 3, 3, config)
        self.assertIn(0, selected.tolist())
        self.assertIn(2, selected.tolist())
        self.assertNotEqual(selected.tolist()[1], 1)

    def test_du_net_interaction_score_is_default_off_and_joint_high_only(self) -> None:
        reward = np.array([3.0, 2.0, 1.0, 0.0])
        entropy = np.array([3.0, 0.0, 2.0, 1.0])
        linear = _du_net_combined_score(reward, entropy, AcquisitionConfig(lambda_entropy=0.5))
        explicit_off = _du_net_combined_score(
            reward,
            entropy,
            AcquisitionConfig(lambda_entropy=0.5, score_interaction_weight=0.0),
        )
        gated = _du_net_combined_score(
            reward,
            entropy,
            AcquisitionConfig(lambda_entropy=0.5, score_interaction_weight=0.5),
        )
        self.assertTrue(np.allclose(linear, explicit_off))
        self.assertGreater(float(gated[0] - linear[0]), 0.0)
        self.assertAlmostEqual(float(gated[1] - linear[1]), 0.0)
        self.assertAlmostEqual(float(gated[2] - linear[2]), 0.0)

    def test_reward_frontier_guard_limits_low_reward_entropy_bonus(self) -> None:
        reward = np.array([10.0, 9.0, 0.0, 0.0])
        entropy = np.array([0.0, 10.0, 10.0, 10.0])
        linear = _du_net_combined_score(reward, entropy, AcquisitionConfig(lambda_entropy=1.0))
        explicit_off = _du_net_combined_score(
            reward,
            entropy,
            AcquisitionConfig(lambda_entropy=1.0, entropy_reward_frontier_quantile=0.0),
        )
        guarded = _du_net_combined_score(
            reward,
            entropy,
            AcquisitionConfig(lambda_entropy=1.0, entropy_reward_frontier_quantile=0.5),
        )
        self.assertTrue(np.allclose(linear, explicit_off))
        self.assertAlmostEqual(float(guarded[1]), float(linear[1]))
        self.assertLess(float(guarded[2]), float(linear[2]))
        self.assertLess(float(guarded[3]), float(linear[3]))

    def test_diffusion_normalization_changes_scores(self) -> None:
        candidates = pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "item_id": [0, 0, 0],
                "rating": [5.0, 4.0, 3.0],
            }
        )
        adjacency = np.array(
            [
                [False, True, True],
                [True, False, False],
                [True, False, False],
            ]
        )
        model = ConstantModel(np.array([1.0, 3.0, 5.0]))
        raw = network_reward_scores(model, candidates, adjacency, diffusion_prob=1.0, normalization="none")
        degree = network_reward_scores(model, candidates, adjacency, diffusion_prob=1.0, normalization="degree")
        self.assertNotAlmostEqual(float(raw[0]), float(degree[0]))
        self.assertAlmostEqual(float(raw[0]), 9.0)
        self.assertAlmostEqual(float(degree[0]), 5.0)

    def test_social_neighbor_item_scores_use_observed_neighbors_only(self) -> None:
        candidates = pd.DataFrame(
            {
                "user_id": [0, 2],
                "item_id": [0, 0],
                "rating": [1.0, 5.0],
            }
        )
        observed = pd.DataFrame(
            {
                "user_id": [1, 2],
                "item_id": [0, 1],
                "rating": [5.0, 2.0],
            }
        )
        adjacency = np.array(
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ]
        )
        model = ConstantModel(np.array([1.0, 2.0, 3.0]))
        social, evidence, direct = _social_neighbor_item_scores(
            model,
            candidates,
            observed,
            adjacency,
            model.values.reshape(-1, 1),
        )
        self.assertAlmostEqual(float(social[0]), 5.0)
        self.assertAlmostEqual(float(evidence[0]), 1.0)
        self.assertAlmostEqual(float(social[1]), float(direct[1]))
        self.assertAlmostEqual(float(evidence[1]), 0.0)

    def test_graph_density_transforms_are_undirected_without_self_loops(self) -> None:
        adjacency = np.array(
            [
                [False, True, True, False],
                [True, False, True, False],
                [True, True, False, True],
                [False, False, True, False],
            ]
        )
        for mode in ["edge_dropout", "rewiring", "densification"]:
            transformed = transform_user_graph(adjacency, mode=mode, rate=0.5, seed=10)
            self.assertEqual(transformed.shape, adjacency.shape)
            self.assertTrue(np.array_equal(transformed, transformed.T))
            self.assertFalse(np.diag(transformed).any())
        dropped = transform_user_graph(adjacency, mode="edge_dropout", rate=1.0, seed=10)
        self.assertEqual(int(np.triu(dropped, k=1).sum()), 0)

    def test_logdet_entropy_and_shrinkage_dls(self) -> None:
        observed = pd.DataFrame({"user_id": [0, 1], "item_id": [0, 1], "rating": [5.0, 3.0]})
        candidates = pd.DataFrame({"user_id": [0, 1, 2], "item_id": [1, 0, 1], "rating": [4.0, 2.0, 5.0]})
        user_factors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        item_factors = np.array([[1.0, 0.5], [0.5, 1.0]])
        scores = observational_logdet_entropy_scores(
            candidates,
            observed,
            num_users=3,
            num_items=2,
            user_factors=user_factors,
            item_factors=item_factors,
            regularization=1.0,
            predicted_values=np.array([4.5, 2.0, 5.0]),
            weighting="reward_cold",
        )
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.all(scores >= 0.0))

        smoothed = np.zeros_like(user_factors)
        effective = np.array([2.0, 1.0, 0.5])
        degree = np.array([3.0, 1.0, 0.0])
        mixed, rho = shrink_dls_user_factors(
            user_factors,
            smoothed,
            effective,
            degree,
            top_k=3,
            strength=0.8,
            user_observed_counts=np.array([4.0, 1.0, 0.0]),
            mode="reliability",
        )
        self.assertEqual(mixed.shape, user_factors.shape)
        self.assertTrue(np.all((0.0 <= rho) & (rho <= 0.8)))
        self.assertTrue(np.linalg.norm(mixed - user_factors) < np.linalg.norm(smoothed - user_factors))

    def test_new_entropy_embedding_and_baselines_do_not_crash(self) -> None:
        dataset = generate_synthetic_dataset(seed=4, num_users=36, num_items=28, observed_fraction=0.32)
        split = split_dataset(dataset, seed=5, test_ratio=0.2, initial_fraction_of_train=0.1)
        for method in [
            "spectral",
            "node2vec",
            "sparse_node2vec",
            "sparse_node2vec_biased",
            "sparse_node2vec_bfs",
            "sparse_node2vec_sgns_bfs",
            "sparse_node2vec_sgns_dfs",
            "sparse_node2vec_dfs",
            "sparse_node2vec_return",
            "sparse_node2vec_outward",
            "elph",
            "elph_struct",
            "elph_role",
        ]:
            emb = user_embedding(dataset.adjacency, dim=4, method=method, seed=6)
            self.assertEqual(emb.shape, (dataset.num_users, 4))
            self.assertFalse(np.isnan(emb).any())

        config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            entropy_mode="observational_hybrid",
            entropy_coverage_weight=0.25,
            selection="topk",
            diffusion_normalization="degree",
            graph_embedding_method="spectral",
        )
        for policy in [
            "du_net",
            "linear_ucb",
            "oful",
            "mf_linear_ucb",
            "mf_oful",
            "mf_residual_ucb",
            "low_rank_bandit",
            "mf_low_rank_ucb",
            "thompson_sampling",
            "mf_thompson_sampling",
            "mf_residual_thompson_sampling",
            "logistic_ucb",
            "logistic_thompson_sampling",
            "finite_ucb",
            "epsilon_greedy",
            "adaptive_greedy",
            "explore_then_exploit",
            "prediction_uncertainty",
            "leverage_uncertainty",
            "active_learning_hybrid",
            "social_neighbor_item",
            "social_mf_hybrid",
            "graph_social",
            "linucb_mf_interaction",
            "linucb_concat",
            "linucb_bias_interaction",
        ]:
            metrics = run_policy(
                policy=policy,
                split=split,
                adjacency=dataset.adjacency,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
                acquisition_config=config,
                seed=7,
            )
            self.assertEqual(len(metrics), 2)
            self.assertFalse(metrics["w_rmse"].isna().any())
            self.assertFalse(metrics["mild_reward_w_rmse"].isna().any())
            self.assertFalse(metrics["medium_reward_w_rmse"].isna().any())
            self.assertFalse(metrics["linear_reward_norm_w_rmse"].isna().any())
            self.assertFalse(metrics["positive_reward_norm_w_rmse"].isna().any())

        logdet_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            entropy_mode="observational_logdet",
            entropy_logdet_weighting="reward",
            dls_shrinkage=0.6,
            dls_shrinkage_mode="reliability",
            selection="exp",
            selection_temperature=0.5,
        )
        metrics = run_policy(
            policy="du_net_logdet_shrinkage_exp_tau05",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=logdet_config,
            seed=8,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["w_rmse"].isna().any())

        coherence_lite_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            entropy_mode="coherence_lite",
            coherence_observed_penalty=0.5,
            selection="top_m_exp",
            selection_top_m=12,
            selection_temperature=0.1,
        )
        metrics = run_policy(
            policy="du_net",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=coherence_lite_config,
            seed=9,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["mild_reward_w_rmse"].isna().any())

        residual_coherence_divcap_config = AcquisitionConfig(
            cycles=2,
            batch_size=4,
            lambda_entropy=0.2,
            dls_shrinkage=0.75,
            dls_shrinkage_mode="fixed",
            dls_shrinkage_schedule=(
                {"start_cycle": 1, "end_cycle": 1, "alpha": 0.75, "mode": "fixed"},
                {"start_cycle": 2, "end_cycle": 2, "alpha": 0.25, "mode": "fixed"},
            ),
            entropy_coherence_blend=0.1,
            selection="diversity_cap",
            selection_max_per_user=2,
            selection_max_per_item=2,
        )
        metrics = run_policy(
            policy="du_net",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=residual_coherence_divcap_config,
            seed=10,
        )
        self.assertEqual(len(metrics), 3)
        self.assertAlmostEqual(float(metrics.loc[0, "avg_dls_shrinkage"]), 0.75)
        self.assertAlmostEqual(float(metrics.loc[1, "avg_dls_shrinkage"]), 0.25)

        adaptive_metric_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            dls_shrinkage_mode="fixed",
            adaptive_dls_shrinkage_grid=(0.0, 0.25),
            adaptive_validation_metric="linear_reward_norm",
            adaptive_prefilter_metric="acquisition_score",
            adaptive_max_refits=2,
        )
        metrics = run_policy(
            policy="du_net_adaptive_grid",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=adaptive_metric_config,
            seed=11,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["linear_reward_norm_w_rmse"].isna().any())

        paper_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.1,
            diffusion_prob=0.0,
            entropy_mode="entropy_observational_lite",
            selection="exp",
            paper_tuning_frequency=10,
        )
        metrics = run_policy(
            policy="paper_faithful_dunet",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=paper_config,
            seed=14,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["pdf_literal_norm_rmse"].isna().any())
        self.assertFalse(metrics["standard_mask_rmse"].isna().any())

        paper_schur_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.1,
            diffusion_prob=0.0,
            entropy_mode="schur_conditional_variance",
            selection="exp",
            paper_tuning_frequency=10,
        )
        metrics = run_policy(
            policy="paper_faithful_dunet",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=paper_schur_config,
            seed=15,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["standard_mask_rmse"].isna().any())

        paper_schur_no_tune_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.1,
            diffusion_prob=0.0,
            entropy_mode="schur_conditional_variance",
            selection="topk",
            paper_tuning_frequency=0,
        )
        metrics = run_policy(
            policy="paper_faithful_dunet_no_entropy",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=paper_schur_no_tune_config,
            seed=16,
        )
        self.assertEqual(len(metrics), 2)
        self.assertEqual(float(metrics.loc[0, "adaptive_lambda_entropy"]), 0.1)
        metrics = run_policy(
            policy="paper_faithful_dunet_no_reward",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=paper_schur_no_tune_config,
            seed=17,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["standard_mask_rmse"].isna().any())

        graph_edge_dls_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            dls_neighbor_mode="graph_edges",
        )
        metrics = run_policy(
            policy="du_net",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=graph_edge_dls_config,
            seed=12,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["linear_reward_norm_w_rmse"].isna().any())

        warm_start_config = AcquisitionConfig(
            cycles=2,
            batch_size=4,
            lambda_entropy=0.2,
            dls_neighbor_mode="graph_edges",
            warm_start_refit=True,
        )
        metrics = run_policy(
            policy="du_net",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="svdpp"),
            acquisition_config=warm_start_config,
            seed=13,
        )
        self.assertEqual(len(metrics), 3)
        self.assertFalse(metrics["linear_reward_norm_w_rmse"].isna().any())

        late_guard_config = AcquisitionConfig(
            cycles=2,
            batch_size=4,
            lambda_entropy=0.2,
            dls_neighbor_mode="graph_edges",
            late_cycle_guard_start=1,
            late_cycle_lambda_multiplier=0.0,
            late_cycle_frontier_quantile=0.9,
        )
        metrics = run_policy(
            policy="du_net",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=late_guard_config,
            seed=13,
        )
        self.assertEqual(len(metrics), 3)
        self.assertEqual(metrics.loc[0, "adaptive_lambda_entropy"], 0.2)
        self.assertEqual(metrics.loc[1, "adaptive_lambda_entropy"], 0.0)

        network_scores = network_observational_entropy_scores(
            split.candidate.head(12),
            split.initial,
            dataset.num_users,
            dataset.num_items,
            user_embedding(dataset.adjacency, dim=4, method="elph_struct", seed=9),
            np.ones((dataset.num_items, 4)),
            dataset.adjacency,
            diffusion_prob=0.05,
            regularization=1.0,
            max_neighbors=3,
        )
        self.assertEqual(network_scores.shape, (12,))
        self.assertTrue(np.all(network_scores >= 0.0))

        expected_zero = network_observational_expected_entropy_scores(
            split.candidate.head(12),
            split.initial,
            dataset.num_users,
            dataset.num_items,
            user_embedding(dataset.adjacency, dim=4, method="elph_struct", seed=9),
            np.ones((dataset.num_items, 4)),
            dataset.adjacency,
            diffusion_prob=0.0,
            regularization=1.0,
            max_neighbors=3,
        )
        base_logdet = observational_logdet_entropy_scores(
            split.candidate.head(12),
            split.initial,
            dataset.num_users,
            dataset.num_items,
            user_embedding(dataset.adjacency, dim=4, method="elph_struct", seed=9),
            np.ones((dataset.num_items, 4)),
            regularization=1.0,
        )
        self.assertTrue(np.allclose(expected_zero, base_logdet))

        stochastic_zero = network_observational_stochastic_entropy_scores(
            split.candidate.head(12),
            split.initial,
            dataset.num_users,
            dataset.num_items,
            user_embedding(dataset.adjacency, dim=4, method="elph_struct", seed=9),
            np.ones((dataset.num_items, 4)),
            dataset.adjacency,
            diffusion_prob=0.0,
            regularization=1.0,
            max_neighbors=3,
            samples=4,
        )
        self.assertTrue(np.allclose(stochastic_zero, base_logdet))

        weighted_zero = network_observational_weighted_entropy_scores(
            split.candidate.head(12),
            split.initial,
            dataset.num_users,
            dataset.num_items,
            user_embedding(dataset.adjacency, dim=4, method="elph_struct", seed=9),
            np.ones((dataset.num_items, 4)),
            dataset.adjacency,
            diffusion_prob=0.0,
            regularization=1.0,
            max_neighbors=3,
        )
        self.assertTrue(np.allclose(weighted_zero, base_logdet))

        linked_candidates = pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "item_id": [0, 0, 1],
                "rating": [4.0, 5.0, 3.0],
            }
        )
        linked_observed = pd.DataFrame({"user_id": [2], "item_id": [1], "rating": [3.0]})
        linked_adj = np.array(
            [
                [False, True, False],
                [True, False, False],
                [False, False, False],
            ]
        )
        linked_users = np.array([[1.0, 0.0], [0.4, 0.8], [0.0, 1.0]])
        linked_items = np.array([[1.0, 0.5], [0.5, 1.0]])
        linked_base = observational_logdet_entropy_scores(
            linked_candidates,
            linked_observed,
            3,
            2,
            linked_users,
            linked_items,
            regularization=1.0,
        )
        linked_expected = network_observational_expected_entropy_scores(
            linked_candidates,
            linked_observed,
            3,
            2,
            linked_users,
            linked_items,
            linked_adj,
            diffusion_prob=0.5,
            regularization=1.0,
            max_neighbors=2,
            normalization="none",
        )
        self.assertGreaterEqual(float(linked_expected[0]), float(linked_base[0]))

        linked_stochastic = network_observational_stochastic_entropy_scores(
            linked_candidates,
            linked_observed,
            3,
            2,
            linked_users,
            linked_items,
            linked_adj,
            diffusion_prob=1.0,
            regularization=1.0,
            max_neighbors=2,
            normalization="none",
            samples=4,
        )
        self.assertGreaterEqual(float(linked_stochastic[0]), float(linked_base[0]))

        linked_reliability_stochastic_zero = network_observational_reliability_stochastic_entropy_scores(
            linked_candidates,
            linked_observed,
            3,
            2,
            linked_users,
            linked_items,
            linked_adj,
            diffusion_prob=0.0,
            regularization=1.0,
            max_neighbors=2,
            normalization="none",
            samples=4,
        )
        self.assertTrue(np.allclose(linked_reliability_stochastic_zero, linked_base))

        linked_reliability_stochastic = network_observational_reliability_stochastic_entropy_scores(
            linked_candidates,
            linked_observed,
            3,
            2,
            linked_users,
            linked_items,
            linked_adj,
            diffusion_prob=1.0,
            regularization=1.0,
            max_neighbors=2,
            normalization="none",
            samples=4,
        )
        self.assertGreaterEqual(float(linked_reliability_stochastic[0]), float(linked_base[0]))

        linked_neighbor_gain_zero = network_neighbor_prediction_gain_entropy_scores(
            linked_candidates,
            linked_observed,
            3,
            2,
            linked_users,
            linked_items,
            linked_adj,
            diffusion_prob=0.0,
            regularization=1.0,
            max_neighbors=2,
            normalization="none",
        )
        self.assertTrue(np.allclose(linked_neighbor_gain_zero, linked_base))
        linked_neighbor_gain = network_neighbor_prediction_gain_entropy_scores(
            linked_candidates,
            linked_observed,
            3,
            2,
            linked_users,
            linked_items,
            linked_adj,
            diffusion_prob=1.0,
            regularization=1.0,
            max_neighbors=2,
            normalization="none",
        )
        self.assertGreaterEqual(float(linked_neighbor_gain[0]), float(linked_base[0]))

        network_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            entropy_mode="network_observational_expected",
            response_alpha=0.02,
            entropy_response_weighting="expected_response",
            entropy_response_weight_strength=1.0,
            graph_embedding_method="sparse_node2vec_biased",
            dls_shrinkage=0.4,
            dls_shrinkage_mode="reliability",
            network_entropy_max_neighbors=3,
        )
        metrics = run_policy(
            policy="du_net_shrinkage_dls",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=network_config,
            seed=10,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["w_rmse"].isna().any())

        neighbor_gain_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            entropy_mode="network_neighbor_prediction_gain",
            diffusion_prob=0.05,
            dls_shrinkage=0.4,
            dls_shrinkage_mode="reliability",
            network_entropy_max_neighbors=3,
            network_entropy_reliability_strength=1.0,
        )
        metrics = run_policy(
            policy="du_net_shrinkage_dls",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=neighbor_gain_config,
            seed=13,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["w_rmse"].isna().any())

        hybrid_neighbor_gain_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            entropy_mode="hybrid_neighbor_prediction_gain",
            diffusion_prob=0.05,
            network_entropy_max_neighbors=3,
            network_neighbor_gain_weight=0.2,
        )
        metrics = run_policy(
            policy="du_net",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=hybrid_neighbor_gain_config,
            seed=14,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["w_rmse"].isna().any())

        weighted_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            entropy_mode="network_observational_weighted",
            diffusion_prob=0.05,
            dls_shrinkage=0.4,
            dls_shrinkage_mode="reliability",
            network_entropy_max_neighbors=3,
            network_entropy_reliability_strength=1.0,
            diffusion_normalization="degree",
        )
        metrics = run_policy(
            policy="du_net_shrinkage_dls",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=weighted_config,
            seed=12,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["w_rmse"].isna().any())

        reliability_stochastic_config = AcquisitionConfig(
            cycles=1,
            batch_size=4,
            lambda_entropy=0.2,
            entropy_mode="network_observational_stochastic_reliability",
            diffusion_prob=0.05,
            dls_shrinkage=0.4,
            dls_shrinkage_mode="reliability",
            network_entropy_max_neighbors=3,
            network_entropy_stochastic_samples=2,
            network_entropy_reliability_strength=1.0,
        )
        metrics = run_policy(
            policy="du_net_shrinkage_dls",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=reliability_stochastic_config,
            seed=11,
        )
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics["w_rmse"].isna().any())

    def test_adaptive_late_guard_policy_runs_observed_only_choice(self) -> None:
        dataset = generate_synthetic_dataset(seed=33, num_users=40, num_items=30, observed_fraction=0.25)
        split = split_dataset(dataset, seed=3, test_ratio=0.2, initial_fraction_of_train=0.15)
        config = AcquisitionConfig(
            cycles=2,
            batch_size=8,
            lambda_entropy=0.2,
            diffusion_prob=0.02,
            sigma=0.3,
            late_cycle_guard_start=1,
            late_cycle_lambda_multiplier=0.5,
            adaptive_validation_metric="linear_reward_norm",
            adaptive_min_inner_reward_delta=0.0,
        )
        metrics = run_policy(
            policy="du_net_adaptive_late_guard",
            split=split,
            adjacency=dataset.adjacency,
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            mf_config=MFConfig(rank=4, epochs=1, algorithm="mf"),
            acquisition_config=config,
            seed=8,
        )
        self.assertEqual(len(metrics), 3)
        self.assertFalse(metrics["w_rmse"].isna().any())
        late_lambdas = set(metrics.loc[metrics["cycle"] >= 1, "adaptive_lambda_entropy"].round(6))
        self.assertTrue(late_lambdas.issubset({0.1, 0.2}))


if __name__ == "__main__":
    unittest.main()
