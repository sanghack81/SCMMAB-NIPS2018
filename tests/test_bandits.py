"""Tests for npsem.bandits — KL helpers and the bandit algorithms."""
import numpy as np
import pytest

from npsem.bandits import KL, sup_KL, kl_UCB, thompson_sampling, play_bandits


# ---- KL divergence / sup_KL ----

def test_kl_zero_when_equal():
    assert KL(0.3, 0.3) == 0


def test_kl_is_positive_off_diagonal():
    assert KL(0.2, 0.5) > 0
    assert KL(0.8, 0.5) > 0


def test_kl_infinite_at_impossible_boundary():
    assert KL(0.5, 1.0) == np.inf
    assert KL(0.5, 0.0) == np.inf


def test_sup_kl_zero_divergence_returns_reference():
    assert sup_KL(0.4, 0.0) == 0.4


def test_sup_kl_within_bounds_and_above_reference():
    mu_ref = 0.4
    out = sup_KL(mu_ref, 0.1)
    assert mu_ref <= out <= 1.0


# ---- kl_UCB optimisation faithfulness ----

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_klucb_faster_equals_slow(seed):
    # K_ = 6 (> 4) so the U_keeper look-ahead path is exercised; it must be
    # bit-identical to the naive recompute.
    mu = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    a_fast, r_fast = kl_UCB(1000, mu, seed=seed, faster=True)
    a_slow, r_slow = kl_UCB(1000, mu, seed=seed, faster=False)
    assert np.array_equal(a_fast, a_slow)
    assert np.array_equal(r_fast, r_slow)


def test_klucb_runs_with_few_arms():
    # K_ <= 4 disables the faster path internally; should still run.
    arms, rewards = kl_UCB(100, [0.2, 0.8], seed=0)
    assert arms.shape == (100,) and rewards.shape == (100,)


# ---- concentration on the best arm ----

def test_thompson_concentrates_on_best_arm():
    mu = [0.1, 0.2, 0.5, 0.5, 0.8]            # unique best = index 4
    arms, _ = thompson_sampling(500, mu, seed=3)
    most_played = int(np.argmax(np.bincount(arms, minlength=len(mu))))
    assert most_played == 4


def test_klucb_concentrates_on_best_arm():
    mu = [0.1, 0.2, 0.5, 0.5, 0.8]
    arms, _ = kl_UCB(500, mu, seed=3)
    most_played = int(np.argmax(np.bincount(arms, minlength=len(mu))))
    assert most_played == 4


# ---- reproducibility ----

def test_thompson_is_reproducible_per_seed():
    mu = [0.2, 0.5, 0.8]
    a1, r1 = thompson_sampling(200, mu, seed=11)
    a2, r2 = thompson_sampling(200, mu, seed=11)
    assert np.array_equal(a1, a2) and np.array_equal(r1, r2)


def test_klucb_is_reproducible_per_seed():
    mu = [0.2, 0.5, 0.8]
    a1, r1 = kl_UCB(200, mu, seed=11)
    a2, r2 = kl_UCB(200, mu, seed=11)
    assert np.array_equal(a1, a2) and np.array_equal(r1, r2)


# ---- play_bandits driver ----

def test_play_bandits_shapes():
    arms, rewards = play_bandits(100, [0.2, 0.5, 0.8], "TS", repeat=3, n_jobs=1)
    assert arms.shape == (3, 100)
    assert rewards.shape == (3, 100)


def test_play_bandits_rejects_unknown_algo():
    with pytest.raises(AssertionError):
        play_bandits(10, [0.5, 0.5], "NOPE", repeat=1, n_jobs=1)
