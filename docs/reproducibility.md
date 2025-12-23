# Reproducibility

Every stage is a sub-command of `echolabel` that reads a stage configuration from `configs/` and a machine paths file from `configs/paths/` and writes its outputs to one directory. It records a `manifest.json` there with the configuration hash, a description of every input, a SHA-256 hash of the package source and the SHA-256 of every output file. A stage whose manifest matches the current configuration, inputs and source is skipped unless `--force` is given. The source hash covers every Python file of the package, so any change to the code makes every earlier manifest stale, and the stage runs again when it is next invoked. The hash kept in a manifest identifies the source that produced its outputs.

## Machines

Two machines are used. The compute server holds the raw data and runs every stage. A development machine holds the repository, runs the unit tests and the stages against a small sample extracted from the raw data; its locations are in `configs/paths/laptop.yaml`, which point at `data/samples/`. The same code and configuration run on both; only the paths file changes.

The locations of the server are not part of the repository. `configs/paths/server.example.yaml` names the five roots (the two read-only data roots, the original annotation collection, the project directory and the scratch directory) as environment variables. Either the variables are exported, or the file is copied to `configs/paths/server.yaml`, which is not tracked, and the locations are written there. The launch scripts read their settings (repository directory, project and scratch directories, conda environment, paths file, login) from `scripts/server.env`, a local copy of `scripts/server.env.example`, or from the environment.

## Environments

On the compute server:

```bash
bash scripts/server/setup_env.sh
```

creates the conda environment `acoustic` (Python 3.11, CUDA-enabled PyTorch) from `requirements/server-lock.txt`, which pins every package. `environment-server.yml` describes the same environment at the level of direct dependencies. On the development machine:

```bash
conda env create -f environment-laptop.yml
```

creates `acoustic-cpu` with the same package versions and a CPU-only PyTorch. In either environment the package is installed in editable mode with `pip install -e .`.

## Running on the compute server

The repository is copied to the server with `bash scripts/sync_to_server.sh` (run on the development machine). Commands run through

```bash
bash scripts/server/run_stage.sh <command> --config <config> --paths configs/paths/server.yaml
```

which activates the environment, lowers the CPU priority and writes a log under the project directory. Long commands run inside a `tmux` session. GPU stages select a device with `scripts/server/select_gpu.sh`, which prints the index of the card with the most free memory. Small result tables are copied back into `results/` with `bash scripts/pull_results.sh <run subdirectory>`.

## Tests

```bash
pytest -q
ruff check src tests scripts
```

The unit tests use synthetic data only and run in about a minute.

## Stage 0: data audit

| Command | Configuration | Output | Runtime on the server |
|---|---|---|---|
| `echolabel audit-annotations` | `configs/stage0_audit.yaml` | `runs/audit/annotations/` | 3 minutes (32 threads) |
| `echolabel audit-transects` | `configs/stage0_audit.yaml` | `runs/audit/transects/` | 1 minute |
| `echolabel audit-zarr` | `configs/stage0_audit.yaml` | `runs/audit/zarr/` | 8 minutes (one pass over the store, 32 threads) |
| `echolabel pull-sample` | `configs/stage0_sample.yaml` | `samples/` | 1 minute |
| `echolabel audit-transect-acoustics` | `configs/stage5_acoustics.yaml` | `runs/acoustics/exploration/` | a few minutes |

Output directories are relative to `project_dir` in the paths file. The last command summarises the echosounder record around the camera transects (range profiles and coarse echograms); its tables and figures are copied into `results/acoustics/exploration/`. The CSV and JSON files of the first three audits are copied into `results/audit/`; the per-item Parquet tables (one row per annotation file, box, frame or ping) stay on the server. The sample written by `pull-sample` is copied to `data/samples/` on the development machine, where the three audit commands run unchanged with `--paths configs/paths/laptop.yaml`.

## Stage 1: annotation dataset

| Command | Configuration | Output | Runtime on the server |
|---|---|---|---|
| `echolabel build-annotation-dataset` | `configs/stage1_dataset.yaml` with `configs/taxonomy.yaml` | `runs/dataset/` | 2 minutes |

The command pairs images with annotation files, maps raw labels through the taxonomy and excludes tows and folders without any annotated frame. It then assigns tows to splits with the seeded search described in `docs/methods.md`, verifies the leakage rules and writes one COCO file per split under `runs/dataset/coco/`. The per-image split assignment is copied into the repository as `configs/splits/annotation_splits.csv`; later stages read the split from that file rather than recomputing it. The class tables are copied into `results/dataset/`.

## Stage 2: detector

The external frameworks are cloned once under `<project_dir>/external/` (RT-DETRv2 from the RT-DETR repository, D-FINE) and their pretrained checkpoints downloaded to `<scratch_dir>/pretrained/`; the commits used are named in `configs/stage2_detector.yaml` and recorded in `results/detector/frameworks.json`. One experiment and seed is trained, run on validation, test and the 2024 hold-out, evaluated and calibrated with

```bash
bash scripts/server/train_detector.sh <experiment> <seed> [MIN_FREE_MIB]
```

which selects the GPU with the most free memory and calls, in turn, `echolabel train-detector`, `predict-detector`, `evaluate-detector` and `calibrate-detector` with `configs/stage2_detector.yaml`. Experiments are named in that file. Checkpoints and framework logs are written under `<scratch_dir>/runs/detector/<experiment>/seed<seed>/`; the generated framework configuration, the training log, predictions, evaluation tables and calibration go to `<project_dir>/runs/detector/<experiment>/seed<seed>/`, and the best weights are copied to `<project_dir>/weights/detector/`. A 40-epoch run of D-FINE-L at 640 pixels takes about four hours on one A100 shared with other users; inference on all three evaluation splits takes two minutes and evaluation with 200 bootstrap resamples a few minutes.

The comparison of the trained detectors (per-year and per-class tables, paired differences, calibration, the object rule) is written by

```bash
bash scripts/server/run_stage.sh summarize-detectors --config configs/stage2_detector.yaml --paths configs/paths/server.yaml
```

to `<project_dir>/runs/detector_summary/` in a few minutes on the CPU.

## Stage 3: transect inference

```bash
bash scripts/server/run_stage3.sh <experiment> <seed> [MIN_FREE_MIB]
```

runs, for every 2018 transect, `echolabel infer-transect` on one GPU and then `review-detections` and `build-camera-series` on the CPU. Its last step, `align-transect`, belongs to the acoustic infrastructure below and needs the preprocessed transect stores; the three commands before it can also be run one by one. Inference reads the left-camera frames from the source directories, stores every detection above the score floor with its calibrated score, resolves objects across classes, removes fixed structure and writes `detections.parquet`, `frames.parquet` and a summary to `<project_dir>/runs/inference/<experiment>_seed<seed>/<transect>/<camera>/`. The 141,727 frames of the four transects take about 40 minutes at 62 frames per second with 16 data-loader workers; JPEG decoding is the bottleneck. The review command draws a fixed random sample of frames per band of calibrated confidence as contact sheets under `<project_dir>/runs/review/`; the sheets hold camera images and are not copied into the repository.

## Stage 4: camera series

```bash
bash scripts/server/run_stage.sh build-camera-series --config configs/stage2_detector.yaml --paths configs/paths/server.yaml --experiment <experiment> --seed <seed> --transect <transect>
```

joins the frames to the Deep Vision metadata by timestamp, cleans the depth, and writes the per-frame depth table and the ten-second series (`series_10s.parquet`) to `<project_dir>/runs/camera_series/<experiment>_seed<seed>/<transect>/<camera>/` in seconds.

The cleaning of the depth is examined on its own with

```bash
bash scripts/server/run_stage.sh audit-camera-depth --config configs/stage2_detector.yaml --paths configs/paths/server.yaml
```

The command cleans the depth of every inference transect with the Stage 4 settings and writes one diagnostic figure per transect and a summary to `<project_dir>/runs/camera_depth_audit/`; it needs only the Deep Vision metadata and runs in about a minute.

## Acoustic infrastructure

The commands of the infrastructure described in Section 5 of `docs/methods.md` are part of the package and follow the conventions above:

| Step | Commands | Configuration |
|---|---|---|
| Acoustic preprocessing | `preprocess-transects`, `downsample-cruise` | `configs/stage5_acoustics.yaml` |
| Alignment | `align-transect` | `configs/stage6_alignment.yaml` |
| Window construction | `build-windows`, `make-splits` | `configs/stage7_windows.yaml` |
| Acoustic models and their evaluation | `pretrain-mae`, `train-acoustic`, `evaluate-acoustic` and the analysis commands listed by `echolabel --help` | `configs/example_acoustic_model.yaml`, an example of a run specification |

They are not needed to reproduce the results in `docs/results.md`, and no outputs of them are kept in the repository.

## Seeds

Every stage takes its seed from its configuration. The selected detector is trained with seed 0 and repeated with seeds 1 and 2 for the spread. The annotation split is found by a seeded search and stored in `configs/splits/annotation_splits.csv`. The frame samples of the review are fixed by a seed. Bootstrap resampling uses fixed generators, so that a rerun of an evaluation command on the same predictions gives the same tables.

## Compute and storage

| Step | Device | Time |
|---|---|---|
| Stage 0 to 1, audits and annotation dataset | CPU, 32 threads | 15 minutes |
| Stage 2, one detector run of 40 epochs at 1,024 pixels | one A100, shared | 5 to 7 hours |
| Stage 2, comparison of the trained detectors | CPU | a few minutes |
| Stage 3, inference on the four transects | one A100 and 16 workers | 40 minutes |
| Stage 4, camera series and depth audit | CPU | about a minute |

No stage uses more than one GPU. Heavy processes run at a lowered CPU priority with at most 32 workers, and the numerical libraries are limited to eight threads per process, since the host has 256 cores shared with other users. The detector checkpoints under `scratch_dir` are the largest derived products; the inference tables of the four transects take about 0.3 GB.

## Figures

The figures under `docs/figures/` are drawn by three scripts that share the style and class colours of `scripts/figure_style.py`. `python scripts/make_detector_figures.py` reads the tables under `results/` and writes the confusion matrix, the precision-recall and F1 curves, the reliability diagram, the per-class AP50 and the framework comparison of the selected experiment. `python scripts/make_transect_series_figure.py <series_10s.parquet>` reads the ten-second series of one transect from the camera series run on the server. `python scripts/make_gallery_figure.py select ...` runs on the server, where it reads the inference tables and copies candidate frames with confident detections of the main species to a folder; `render` draws the frames listed in `docs/figures/gallery_frames.csv` from that folder. `python scripts/make_pipeline_figure.py` draws the pipeline. Each figure is written as PNG and PDF without creation time or software tags, so a rerun gives identical files.

## Verifying outputs

A manifest lists every output file of its stage with a SHA-256 hash.

```bash
python scripts/verify_manifest.py [--strict] [--paths <paths file>] <directory> [<directory> ...]
```

checks every stage directory below the given ones against its manifest, prints the number of matching files per directory and names every file that has changed. Text files are compared up to their line endings. The exit status is 1 if a file has changed; with `--strict` it is also 1 if a recorded output is absent or cannot be compared, which is the check to run on the server, where every output is kept.

The small tables and figures under `results/` are copies of stage outputs together with their manifests, without the large per-item files, which are counted as not present. In the copies the locations of the server are replaced by placeholders such as `<project_dir>` (`scripts/redact_paths.py`, called by `scripts/pull_results.sh`), and its host name in the manifests by `<host>`. A copy with a placeholder no longer has the recorded hash. With `--paths` and the paths file of the server the placeholders are put back before the comparison; without it such a file is counted as not compared. `python scripts/verify_manifest.py results` therefore verifies that the tables in the repository are the ones that the recorded commands wrote, except for the files with placeholders.
