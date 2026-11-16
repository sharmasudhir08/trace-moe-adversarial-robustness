"""Abstain layer (contribution C3): calibrated confidence, a reject threshold, and the risk-coverage analysis.

After training, a temperature T is fitted on the validation windows (Guo et al., 2017), so the 19-class
confidence is calibrated. The reject threshold is the validation confidence that keeps `target_coverage`
of the windows; below it the model abstains and raises an "unknown / uncertain" alert instead of a class.
The same procedure is applied to every model (neural and RF), so the comparison is fair.

Selective-classification terms (Geifman and El-Yaniv, 2017):
  coverage        share of windows the model accepts
  selective risk  error rate on the accepted windows
  AURC            area under the risk-coverage curve (lower is better), threshold-free
"""
import numpy as np
from sklearn.metrics import f1_score, normalized_mutual_info_score

from .metrics import expected_calibration_error


def _scaled(proba, T):
    z = np.log(np.clip(proba, 1e-12, 1.0)) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(proba_val, y_val, grid=None):
    """Temperature that minimises the validation negative log-likelihood (grid search, robust and fast)."""
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 60)) if grid is None else grid
    best, best_nll = 1.0, np.inf
    for T in grid:
        p = _scaled(proba_val, T)
        nll = -np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-12, 1)).mean()
        if nll < best_nll:
            best, best_nll = float(T), nll
    return best


def risk_coverage(conf, correct, n_points=20):
    """Risk-coverage curve (sorted by confidence, most confident first) and its area (AURC).

    Bug fix 2026-11-13: ties in confidence are broken in expectation. Previously
    `kind="stable"` kept tied windows in their input order, which groups them by
    file (and therefore by class). The resulting AURC measured file order rather
    than the model's ranking. For neural models whose softmax saturates at
    exactly 1.0 for 80-99 percent of windows, this produced a spurious ordering.

    New semantics: windows with the SAME confidence contribute the group mean
    correct-rate to every member of the group. This is the expected-value
    tie-break described in the fix memo and matches what a random tie-break
    would give on average.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    n = len(conf)
    order = np.argsort(-conf, kind="mergesort")
    sorted_conf = conf[order]
    sorted_correct = correct[order].astype(np.float64)
    expected_correct = sorted_correct.copy()
    # Expected-value tie-break: replace per-window correct with per-group mean
    _, inv, counts = np.unique(-sorted_conf, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        group_means = np.bincount(inv, weights=sorted_correct) / counts
        expected_correct = group_means[inv]
    k = np.arange(1, n + 1)
    risk = 1 - np.cumsum(expected_correct) / k
    cov = k / n
    pts = np.linspace(0.05, 1.0, n_points)
    idx = np.clip(np.ceil(pts * n).astype(int) - 1, 0, n - 1)
    return float(risk.mean()), {"coverage": cov[idx].round(4).tolist(),
                                 "risk": risk[idx].round(6).tolist()}


def _log_space_margin(p):
    """Confidence proxy that does not saturate at 1.0 like `p.max(1)` does.

    Returns `log(p_top1) - log(p_top2)`, i.e. the log-space margin between the
    winning class and the runner-up. For saturated softmax outputs the top1
    log-prob stays near 0 while the top2 log-prob can be arbitrarily negative,
    so the margin retains ranking information.
    """
    log_p = np.log(np.clip(p, 1e-300, 1.0))
    log_p_sorted = np.sort(log_p, axis=1)
    return log_p_sorted[:, -1] - log_p_sorted[:, -2]


def selective_report(proba_val, y_val, proba_test, y_test, target_coverage=0.95):
    """Calibrate on validation, pick the reject threshold on validation, report on test.

    Bug fix 2026-11-13: confidence is now the log-space top1 minus top2 margin
    instead of `pt.max(1)`. The softmax max saturates at 1.0 for 80-99 percent
    of neural-model test windows, which left the risk-coverage sort unable to
    rank most windows. The log-space margin does not saturate.
    """
    if len(proba_test) == 0:
        return {}
    have_val = proba_val is not None and len(proba_val) > 0
    T = fit_temperature(proba_val, y_val) if have_val else 1.0
    pt = _scaled(proba_test, T)
    conf = _log_space_margin(pt)
    pred = pt.argmax(1)
    correct = pred == y_test
    if have_val:
        pt_val = _scaled(proba_val, T)
        conf_val = _log_space_margin(pt_val)
    else:
        conf_val = conf
    thr = float(np.quantile(conf_val, 1 - target_coverage))
    acc = conf >= thr
    aurc, curve = risk_coverage(conf, correct)
    labels = np.arange(proba_test.shape[1])
    out = {"temperature": T, "threshold": thr, "target_coverage": target_coverage,
           "coverage": float(acc.mean()), "risk_full": float(1 - correct.mean()),
           "selective_risk": float(1 - correct[acc].mean()) if acc.any() else float("nan"),
           "accepted_macro_f1": float(f1_score(y_test[acc], pred[acc], labels=labels, average="macro",
                                               zero_division=0)) if acc.any() else float("nan"),
           "aurc": aurc, "ece_raw": expected_calibration_error(y_test, proba_test),
           "ece_calibrated": expected_calibration_error(y_test, pt), "curve": curve}
    return out


def routing_report(route, y6, y19, classes6, n_experts):
    """How the experts split the traffic: NMI with the 6 categories and the 19 classes, and counts per category."""
    if route is None or len(route) == 0:
        return {}
    counts = {}
    for k, name in enumerate(classes6):
        m = y6 == k
        if m.any():
            counts[name] = np.bincount(route[m], minlength=n_experts).tolist()
    return {"nmi_label6": float(normalized_mutual_info_score(y6, route)),
            "nmi_label19": float(normalized_mutual_info_score(y19, route)),
            "expert_share": (np.bincount(route, minlength=n_experts) / len(route)).round(4).tolist(),
            "counts_by_category": counts}
