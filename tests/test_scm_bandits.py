"""Tests for npsem.scm_bandits — the SCM -> bandit-machine bridge.

The headline test is the paper's central claim: the POMIS arm set always
contains a globally optimal arm.
"""
import numpy as np
import pytest

from npsem.scm_bandits import SCM_to_bandit_machine, arms_of, arm_types


def test_arm_types():
    assert arm_types() == ["POMIS", "MIS", "Brute-force", "All-at-once"]


def test_iv_arm_machine_shape(iv_scm):
    M, _ = iv_scm
    mu, arm_setting = SCM_to_bandit_machine(M)
    assert len(mu) == 9                       # 3 vars (X,Z) subsets x binary values
    assert len(arm_setting) == 9
    assert not np.any(np.isnan(mu))
    assert arm_setting[0] == {}               # first arm is the empty intervention


def test_iv_arm_counts(iv_scm):
    M, _ = iv_scm
    _, arm_setting = SCM_to_bandit_machine(M)
    counts = {at: len(arms_of(at, arm_setting, M.G, "Y")) for at in arm_types()}
    assert counts == {"POMIS": 4, "MIS": 5, "Brute-force": 9, "All-at-once": 4}


def test_strategy_arms_are_subsets_of_bruteforce(scm):
    M, _ = scm
    _, arm_setting = SCM_to_bandit_machine(M)
    allarms = set(arms_of("Brute-force", arm_setting, M.G, "Y"))
    for at in ("POMIS", "MIS", "All-at-once"):
        assert set(arms_of(at, arm_setting, M.G, "Y")) <= allarms


def test_pomis_arms_contain_a_global_optimum(scm):
    """Paper's core claim — verified across every example SCM."""
    M, _ = scm
    mu, arm_setting = SCM_to_bandit_machine(M)
    mu = np.asarray(mu)
    mu_star = np.nanmax(mu)
    pomis_arms = arms_of("POMIS", arm_setting, M.G, "Y")
    assert max(mu[a] for a in pomis_arms) == pytest.approx(mu_star)


def test_mis_arms_contain_a_global_optimum(scm):
    M, _ = scm
    mu, arm_setting = SCM_to_bandit_machine(M)
    mu = np.asarray(mu)
    mis_arms = arms_of("MIS", arm_setting, M.G, "Y")
    assert max(mu[a] for a in mis_arms) == pytest.approx(np.nanmax(mu))


def test_all_at_once_is_suboptimal_on_iv(iv_scm):
    """Intervening on *everything* misses the optimum in the IV graph."""
    M, _ = iv_scm
    mu, arm_setting = SCM_to_bandit_machine(M)
    mu = np.asarray(mu)
    aao = arms_of("All-at-once", arm_setting, M.G, "Y")
    assert max(mu[a] for a in aao) < np.nanmax(mu)
