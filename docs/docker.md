# Docker guide

This page covers the full set of Docker options for running the validation/metrics backend: permission handling, path mapping, and an interactive shell for debugging. For the single most common command, see the [README](../README.md#docker).

## Table of contents

- [Pulling the image](#pulling-the-image)
- [Permissions: the `--user` flag](#permissions-the---user-flag)
- [Mounting your data](#mounting-your-data)
- [Interactive shell](#interactive-shell)
- [Running as a CLI](#running-as-a-cli)
- [Troubleshooting](#troubleshooting)

---

## Pulling the image

```bash
docker pull dbouget/raidionics-val:v1.1.1-py39-cpu
```

---

## Permissions: the `--user` flag

Every command below includes `--user $(id -u)`. This matters in practice: without it, any files the container creates (validation results, logs, temp files) will be owned by `root` on your host, and you won't be able to read/write/delete them without `sudo`.

`$(id -u)` resolves to your current user's numeric UID at runtime — no need to hard-code it, though you can substitute your UID directly (e.g. `--user 1000`) if you prefer.

---

## Mounting your data

All commands mount a local directory into the container at `/workspace/resources`:

```bash
-v /home/<username>/<resources_path>:/workspace/resources
```

Replace `/home/<username>/<resources_path>` with a real path on your machine. This directory must contain:

- The folder(s) with data to use as input for the validation studies, following the [expected data format](data_format.md)
- A destination folder where results will be saved

The container can only see what's under this mounted path.

---

## Interactive shell

Useful for debugging, inspecting the environment, or running commands manually inside the container:

```bash
docker run --entrypoint /bin/bash \
  -v /home/<username>/<resources_path>:/workspace/resources \
  -t -i --network=host --ipc=host --user $(id -u) \
  dbouget/raidionics-val:v1.1.1-py39-cpu
```

---

## Running as a CLI

For direct, non-interactive runs:

```bash
docker run \
  -v /home/<username>/<resources_path>:/workspace/resources \
  -t -i --network=host --ipc=host --user $(id -u) \
  dbouget/raidionics-val:v1.1.1-py39-cpu \
  -c /workspace/resources/<path>/<to>/main_config.ini -v <verbose>
```

**Path notes:** the `-c` argument must point to the config file's path *inside the container*, i.e. relative to `/workspace/resources`. Concretely, if your config lives at:

```
/home/myuser/Data/Validation/main_config.ini
```

and you mounted `/home/myuser/Data` as your resources path, the correct `-c` value is:

```
/workspace/resources/Validation/main_config.ini
```

**Verbosity levels** (`-v`): `debug`, `info`, `warning`, `error`.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Output files owned by `root` | Forgot `--user $(id -u)` |
| Container can't find input files | Path passed to `-c` isn't relative to `/workspace/resources`, or the resources volume wasn't mounted correctly |
| Validation fails immediately with a folder-structure error | Data doesn't match the [expected format](data_format.md) — check the `predictions/<fold>/` layout and `cross_validation_folds.txt` naming |
