# Migration to computation-nvflare-boilerplate

This computation was migrated from the hand-written NVFlare
`Controller`/`Executor`/`Aggregator` architecture (NVFlare 2.4, Python 3.8) to
[`computation-nvflare-boilerplate`](https://github.com/NeuroFlame/computation-nvflare-boilerplate)
`0.1.0` (NVFlare 2.8, Python 3.11). Framework-owned files were applied with the
boilerplate's `scripts/migrate_computation.py --in-place --force`. The SVR math
was ported into `app/code/computation/`.

## Workflow

The old two tasks map to a `stepped_workflow` with three local/remote pairs
and a final output step:

| Old task | New local step | New remote step |
|---|---|---|
| — | `prepare_site` (with `load_inputs`) | `designate_owner` |
| `GET_LOCAL_SVR_WEIGHTS` | `train_member_model` | `stack_member_weights` |
| `ACCEPT_AGGREGATED_WEIGHTS` | `train_owner_model` | `finish_run` |
| — | `build_outputs` (output step) | — |

The extra first round exists because a site needs to know whether it is the
owner before round 0's training (see [Owner site](#owner-site)).

## Owner site

The old executor compared its own NVFlare client name with an `owner_site`
parameter. That parameter was set by a customized `system/entry_provision.py`
to the first provisioned user. The framework does not tell a site its own
identity, and `system/entry_provision.py` is owned by the boilerplate, so the
migration replaced it.

The owner is now the consortium leader's site, named by the required
`consortium_leader_id` parameter (a site display name, or a NeuroFLAME user ID
resolved through `site_id_name_map`). This follows
[nfc-dpsvm](https://github.com/NeuroFlame/nfc-dpsvm), which uses the same
owner/member algorithm:

- In the first round, each site mints a random token and keeps it in local
  state.
- The server resolves the owner's display name to its token and sends the
  token to every site. A site acts as the owner when the token matches its own.
- The run stops with an error if `consortium_leader_id` is missing, if the
  leader's site is not taking part, or if no other site takes part.

## Local state holds indices, not data

The old owner cached its train/test feature matrices in the NVFlare context
between rounds. The framework serializes local state with an 8 MiB limit per
array, and a realistic owner matrix exceeds it (for example, 2,000 subjects ×
1,378 FNC features is about 22 MB). Local state now holds only the train/test
row indices. Rounds that need the data re-read the site's input files.

## Other intentional differences

- **`data_file` and `label_file`** in `parameters.json` are now used. The old
  executor ignored them and used hardcoded filenames, which are still the
  defaults.
- **Outputs are unchanged**: `owner_svr_result.json` at the owner,
  `local_svr_result.json` at each member, and `index.html` at every site. The
  report is the old one, now built in `app/code/computation/report.py` and
  returned to the framework instead of being written directly. Each site's
  report shows its own display name.
- **Unknown `input_source`** values now raise an error. Previously the old code
  failed later with an unbound-variable error.
- **SVR parameters** fall back to the documented defaults when
  `svr_params_local` or `svr_params_owner` is omitted, matching the display
  notes, which list them as optional. Previously a missing value raised
  `KeyError`.
- **Unused code removed**: k-fold partition generators, `.npy` loading, and the
  `__main__` test stubs in the old utilities. None fed into the output.
- **Dependencies**: boilerplate pins plus h5py, scipy, scikit-learn, joblib,
  and threadpoolctl. `h5py` moved from 3.1.0 to 3.9.0 because 3.1.0 does not
  install on Python 3.11.

## Verification

- **Numeric parity**: the pre-migration and migrated implementations were run
  on the bundled test data with identical train/test splits. The member weight
  vectors, the owner weight and intercept, and all RMSE/MAE values match
  exactly.
- A 3-site `./run_local_simulation.sh site1,site2,site3` run completes. site1
  (the leader in the test parameters) writes the owner outputs and sites 2 and
  3 write the member outputs.
- `make check` (ruff, formatting, compile, unit tests) and
  `migrate_computation.py --check` (0 differing paths) pass.

## Known issues carried over

**High owner error on the bundled test data.** The owner model's test RMSE is
about 27–49 years, depending on the random split, while the member models fit
their own data almost perfectly. The pre-migration code gives the same numbers,
so the migration did not cause this. The test data is synthetic
(`synth-siteN-*.nii`) and may not relate FNC to age. The single projected
feature, with the default `LinearSVR` settings, may also underfit.

**The report's COINSTAC reference values are from a different dataset.**
`COINSTAC_REF` in `report.py` is documented as a run where site1 and site2 had
identical 20-subject data. All sites in the current test data differ, so the
"COINSTAC ref" columns in `index.html` can't be compared with runs on this
data.
