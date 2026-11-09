#!/usr/bin/env python3
"""Exact two-period dependence counterexample for the M5 research dossier.

No third-party packages or data files are required.
Run: python3 M5研究/experiments/joint_dependence_demo.py
"""
from statistics import mean

# Two distributions with identical Bernoulli(0.5) marginal distributions.
# A: perfectly positively dependent; B: perfectly negatively dependent.
SCENARIOS = {
    "A_comonotonic": [(0, 0), (100, 100)],
    "B_countermonotonic": [(0, 100), (100, 0)],
}


def positive_part(x):
    return max(x, 0)


def evaluate(paths, shared_capacity=100, local_capacity=50):
    totals = [x + y for x, y in paths]
    shared_shortage = [positive_part(t - shared_capacity) for t in totals]
    shared_leftover = [positive_part(shared_capacity - t) for t in totals]

    # Fixed, separable allocation: 50 units reserved for each item/region.
    local_shortage = [
        positive_part(x - local_capacity) + positive_part(y - local_capacity)
        for x, y in paths
    ]
    local_leftover = [
        positive_part(local_capacity - x) + positive_part(local_capacity - y)
        for x, y in paths
    ]
    return {
        "marginal_mean_x": mean(x for x, _ in paths),
        "marginal_mean_y": mean(y for _, y in paths),
        "shared_total_p90_left_continuous": sorted(totals)[-1],
        "shared_expected_shortage": mean(shared_shortage),
        "shared_expected_leftover": mean(shared_leftover),
        "local_expected_shortage": mean(local_shortage),
        "local_expected_leftover": mean(local_leftover),
        "shared_probability_capacity_exceeded": mean(t > shared_capacity for t in totals),
    }


def main():
    results = {name: evaluate(paths) for name, paths in SCENARIOS.items()}
    for name, result in results.items():
        print(name)
        for key, value in result.items():
            print(f"  {key}: {value}")

    a, b = results["A_comonotonic"], results["B_countermonotonic"]
    assert a["marginal_mean_x"] == b["marginal_mean_x"] == 50
    assert a["marginal_mean_y"] == b["marginal_mean_y"] == 50
    assert a["shared_expected_shortage"] == 50
    assert b["shared_expected_shortage"] == 0
    assert a["shared_expected_leftover"] == 50
    assert b["shared_expected_leftover"] == 0
    assert a["local_expected_shortage"] == b["local_expected_shortage"] == 50
    assert a["local_expected_leftover"] == b["local_expected_leftover"] == 50
    print("PASS: identical marginals; shared-pool risk differs; fixed separable costs agree.")


if __name__ == "__main__":
    main()
