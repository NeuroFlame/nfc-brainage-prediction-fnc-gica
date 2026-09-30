# Brain Age Prediction FNC GICA

Brain age prediction using decentralized SVR with FNC (functional network connectivity) matrices as features. Ported from [coinstac-brainage-fnc](https://github.com/trendscenter/coinstac-brainage-fnc) to the NeuroFLAME / NVFlare federated learning framework.

---

## Algorithm

The consortium leader's site is the **owner**: it holds out its data and trains the final model. All other sites are **members**.

1. **Setup** — Every site loads its FNC data, splits it into train/test sets, and reports a random token. The server resolves the owner from `consortium_leader_id` and tells the sites which token is the owner's.
2. **Member training** — Each member trains a `MinMaxScaler + LinearSVR` on all of its data and sends back the learned weight vector. The owner skips this step.
3. **Aggregation** — The server stacks the member weight vectors into `w_locals` (shape `n_features × n_members`) and sends them to all sites.
4. **Owner training** — The owner projects its data through the averaged member weights (`U = X @ mean(w_locals)`), then trains a second `LinearSVR` on the projected feature.
5. **Outputs** — The owner writes `owner_svr_result.json`; each member writes its `local_svr_result.json`. Every site writes an `index.html` report.

---

## Project Structure

This computation follows the [NeuroFLAME computation boilerplate](https://github.com/NeuroFlame/computation-nvflare-boilerplate) (version recorded in `.neuroflame.json`). Computation code lives in `app/code/computation/`:

```
app/code/computation/
├── spec.py          # Workflow: which steps run, in what order
├── types.py         # Values exchanged between steps
├── inputs.py        # FNC .mat + covariates loading, upper-triangle features
├── local_math.py    # Train/test split, member SVR, owner projected SVR
├── remote_math.py   # Owner resolution, weight stacking
├── results.py       # Output files per site
└── report.py        # index.html report
```

`app/code/framework/`, `app/code/runtime/`, `app/config/`, `system/`, the Dockerfiles, and the scripts are owned by the boilerplate. Update them with the boilerplate's `scripts/migrate_computation.py`, not by hand.

---

## Input Data

Each site directory under `test_data/` must contain:

- `coinstac-gica_postprocess_results.mat` — MATLAB file with `fnc_corrs_all` array (subjects × sessions × components × components)
- `covariates.csv` — CSV with at minimum `filename` and `age` columns

---

## Computation Parameters

Edit `test_data/server/parameters.json` to configure the run:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `consortium_leader_id` | The consortium leader's site name or NeuroFLAME user ID. That site is the owner. **Required.** | — |
| `data_file` | FNC `.mat` filename | `"coinstac-gica_postprocess_results.mat"` |
| `label_file` | Covariates CSV filename | `"covariates.csv"` |
| `input_source` | `"GICA"` or `"UKBioBank_Comp2019"` | `"GICA"` |
| `split_type` | `"random"` or `"age_range_stratified"` | `"random"` |
| `test_size` | Fraction of subjects for test set | `0.1` |
| `shuffle` | Shuffle before splitting | `true` |
| `svr_params_local` | `LinearSVR` kwargs for member sites | see file |
| `svr_params_owner` | `LinearSVR` kwargs for the owner site | see file |
| `log_level` | `debug`, `info`, `warning`, `error`, or `critical` | `"info"` |

The test parameters set `consortium_leader_id` to `site1`.

---

## Running Locally

Run a local simulation with the bundled test data (requires Docker):

```bash
./run_local_simulation.sh site1,site2,site3
```

Results are written to `test_output/simulate_job/<site>/`. Add `--no-build` to skip rebuilding the image after source-only changes.

Run lint, formatting checks, and unit tests with:

```bash
make check
```

---

## Notes on COINSTAC → NeuroFLAME Translation

| COINSTAC | NeuroFLAME |
|----------|------------|
| `local.py local_0()` | `local_math.prepare_site()` + `local_math.train_member_model()` |
| `remote.py remote_0()` | `remote_math.designate_owner()` + `remote_math.stack_member_weights()` |
| `local.py local_1()` | `local_math.train_owner_model()` |
| `remote.py remote_1()` | `remote_math.finish_run()` |
| `state['owner']` / `clientId` check | `consortium_leader_id`, matched through per-site tokens |
| `args['cache']` (disk) | Framework local state (`with_state`), holding train/test indices |
| `compspec.json` inputs | `test_data/server/parameters.json` |
