# Update the worker

Stop the worker before updating. When you finish, the worker source and dependencies are current while
your configuration, downloaded models, and runtime cache remain in place.

Use the procedure that matches how you installed the worker. If you are unsure, open the worker folder:
a `.git` entry means you have a git installation. Most other Windows installations use the managed
installer procedure. These procedures update the existing folder and have no automatic undo; see
[Go back to an earlier version](#go-back-to-an-earlier-version) before continuing if you may need a rollback.

## Update a managed installation

Use these steps for a worker installed by the Windows installer, the one-line installer, or a previous
managed release.

1. Stop the worker with **Quit** in the dashboard or `Ctrl+C` in its terminal. Continue when the dashboard
   closes or the terminal returns to its prompt.
2. Run `update.cmd` on Windows or `./update.sh` on Linux/macOS from the worker folder. Continue when it
   reports `Updated to <version>` or `Already up to date (<version>)` and finishes dependency sync.
3. Start the worker normally. Confirm the new version in the dashboard or the startup log.

You can also re-run the installer over the same folder. It preserves `bridgeData.yaml` and the sibling
`<worker>-data` folder that holds models, managed Python, and caches.

## Update a git installation

1. Stop the worker with **Quit** in the dashboard or `Ctrl+C` in its terminal. Continue when the dashboard
   closes or the terminal returns to its prompt.
2. Open a terminal in the worker folder and update the checkout:

    ```bash
    git pull --ff-only
    ```

   Continue when git reports a fast-forward or says the checkout is already up to date. If git reports
   local changes or a divergent branch, keep those changes and resolve the git state before continuing.

3. Synchronize the worker environment:

    ```bat
    update.cmd
    ```

   On Linux/macOS, run `./update.sh`. `update-runtime.cmd` and `./update-runtime.sh` remain equivalent
   dependency-only alternatives after the pull.

   Continue when the command reports `Git checkout dependencies are up to date`.

4. Start the worker. Confirm the new version in the dashboard or startup log.

This procedure also supports old installations that used micromamba. The new environment is created
alongside the old files; you do not need to delete the old environment before updating.

## Update an extracted zip

1. Stop the worker. Continue when the dashboard closes or the terminal returns to its prompt.
2. Download the [latest source zip](https://github.com/Haidra-Org/horde-worker-reGen/archive/refs/heads/main.zip).
   Continue when the browser reports that the download completed.
3. Extract it over the existing worker folder and allow matching files to be replaced. Keep
   `bridgeData.yaml` and the sibling `<worker>-data` folder. Confirm that
   `horde_worker_regen/__init__.py` contains the version from the downloaded archive.
4. Run `update-runtime.cmd` on Windows or `./update-runtime.sh` on Linux/macOS. Continue when dependency
   synchronization completes without an error.
5. Start the worker and confirm the new version in the dashboard or startup log.

## Respond to an update prompt

A managed worker can offer an update when it starts:

- **Yes** applies the update and continues starting.
- **No** skips it for this launch.
- **skip** hides that specific version. A newer release is still offered later.

You can always update later by running `update.cmd` or `./update.sh`.

## Control a large dependency download

The updater previews large dependency changes before downloading them. PyTorch is usually the largest
item. When prompted, choose **Upgrade** to continue, **Hold** to keep the installed PyTorch when compatible,
or **Cancel** to leave the environment unchanged.

To request the compatible hold directly, run:

```bat
update.cmd --hold-torch
```

On Linux/macOS, use `./update.sh --hold-torch`. The worker refuses the hold when the new release requires a
newer PyTorch, so this option cannot create an incompatible environment. See the
[update and dependency-sync command reference](../reference/cli.md#update-and-dependency-sync) for all
backend, cache, non-interactive, and release-channel controls.

## Recover from an interrupted update

Run the same update command again. Every launch and update verifies the private package manager and checks
whether the environment matches the current lockfile before importing worker code. A source update that
stopped before dependency synchronization therefore resumes safely on the next attempt.

If retrying fails:

- For a download or connection error, restore access to GitHub Releases and retry.
- For a disk-space error, free space on the worker-data drive and retry.
- For `CRYPT_E_NO_REVOCATION_CHECK`, temporarily disable the antivirus download inspection that produced
  it, retry, then re-enable protection.
- If the same bootstrap error repeats, re-run the latest installer over the same folder or ask in
  [#local-workers on Discord](https://discord.com/channels/781145214752129095/1076124012305993768).

Do not delete `bridgeData.yaml`, the sibling `<worker>-data` folder, or downloaded models to repair a
dependency update.

## Go back to an earlier version

Updates have no automatic rollback. If you must return to an older release, install that release into a
separate folder and copy your `bridgeData.yaml` into it. Point its `cache_home` at the existing model cache
if you want to avoid downloading models again. Keep the current folder until the older worker starts
successfully.

For the design and failure guarantees behind these procedures, see
[Updates and bootstrap](../explanation/updates_and_bootstrap.md).
