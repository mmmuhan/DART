import numpy as np
import pandas as pd
from pathlib import Path
import pyabc


def save_tstop_posterior(csv_path, obs_idx, t_stop, post_df):
    row = {"obs_idx": int(obs_idx), "t_stop": int(t_stop)}
    row.update({f"p_model_{k}": float(v) for k, v in post_df["p"].items()})
    csv_path = Path(csv_path)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


class L2Distance(pyabc.Distance):
    def __call__(self, x, x0, t=None, par=None):
        d = x["x"] - x0["x"]
        return float(np.sqrt(np.dot(d, d)))


def build_empirical_models(xmat, label_all, train_idx, seed=0):
    xmat = np.asarray(xmat)
    label_all = np.asarray(label_all).astype(int)
    train_idx = np.asarray(train_idx, dtype=int)

    train_y = label_all[train_idx]
    idx1 = train_idx[train_y == 1]
    idx2 = train_idx[train_y == 2]
    lib1 = xmat[idx1]
    lib2 = xmat[idx2]

    rng = np.random.default_rng(seed)

    def sim_from(lib):
        def sim(pars):
            j = rng.integers(0, lib.shape[0])
            return {"x": lib[j]}
        return sim

    models = [sim_from(lib1), sim_from(lib2)]
    priors = [pyabc.Distribution(), pyabc.Distribution()]
    return models, priors


def run_one_obs_with_stop(
    models, priors, observed_x, db_path,
    population_size=200,
    eps_quantile=0.5,
    max_pops_cap=8,
    acc_thresh=0.01,
    dp_thresh=0.05,
    patience=1,
    verbose=True,
):
    abc = pyabc.ABCSMC(
        models=models,
        parameter_priors=priors,
        distance_function=L2Distance(),
        population_size=population_size,
        eps=pyabc.epsilon.QuantileEpsilon(alpha=eps_quantile),
        sampler=pyabc.sampler.SingleCoreSampler(),
    )

    abc.new(db_path, {"x": np.asarray(observed_x).copy()})
    history = abc.run(max_nr_populations=max_pops_cap)

    pops = history.get_all_populations().query("t>=0").copy()
    t_max = int(pops["t"].max())

    prev_pmax = None
    hit = 0
    t_stop = t_max

    for t in range(t_max + 1):
        r = pops[pops["t"] == t].iloc[0]
        acc_rate = float(r["particles"] / r["samples"])
        eps = float(r["epsilon"])

        post = history.get_model_probabilities(t=t)
        p = post["p"].values
        pmax = float(np.max(p))
        dp = None if prev_pmax is None else float(abs(pmax - prev_pmax))

        if verbose:
            if dp is None:
                print(f"[t={t}] eps={eps:.4g} acc={acc_rate:.4%} p={p}")
            else:
                print(f"[t={t}] eps={eps:.4g} acc={acc_rate:.4%} p={p}  dp={dp:.4g}")

        if (prev_pmax is not None) and (acc_rate < acc_thresh) and (dp < dp_thresh):
            hit += 1
        else:
            hit = 0

        if hit >= patience:
            t_stop = t
            if verbose:
                print(f"[STOP] Chosen t={t_stop}: acc<{acc_thresh} and dp<{dp_thresh} for {patience} consecutive pops.")
            break

        prev_pmax = pmax

    post_final = history.get_model_probabilities(t=t_stop)
    return post_final, history, t_stop


# ---- build once ----
models, priors = build_empirical_models(xmat_f, labels_all, train_idx, seed=0)

# ensure db folder exists
db_dir = Path("../../synthetic_data/svm/ABC/100_6_04")
db_dir.mkdir(parents=True, exist_ok=True)

# ---- loop multiple tests ----
hists = []
for obs_idx in test_idx[0:100]:
    obs_idx = int(obs_idx)

    post, hist, tstop = run_one_obs_with_stop(
        models, priors, xmat_f[obs_idx],
        db_path=f"sqlite:///{db_dir.resolve()}/pyabc_{obs_idx}.db",
        population_size=100,
        eps_quantile=0.4,
        max_pops_cap=6,
        acc_thresh=0.05,
        dp_thresh=0.05,
        patience=1,
        verbose=True,
    )

    save_tstop_posterior(f"{db_dir}/tstop_posterior.csv", obs_idx, tstop, post)
    hists.append(hist)
