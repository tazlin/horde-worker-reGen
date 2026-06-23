# Getting started

This guide takes you from nothing to a running worker that is earning kudos. It assumes no command-line
experience and makes the setup choices for you. Set aside about an hour, most of which is the computer
downloading things while you do something else.

## What you need

- A supported machine. The short version: an **NVIDIA GPU** on **Windows** or **Linux**, or an AMD GPU
  on Linux. If you are not sure, check the support table on the
  [README](https://github.com/Haidra-Org/horde-worker-reGen#readme) first.
- A free **AI Horde API key**. Get one at [aihorde.net/register](https://aihorde.net/register) and keep
  it somewhere handy. Treat it like a password.
- Some free disk space. The program itself needs around 10 to 15 GB, and the AI models you choose are
  extra (a useful set is tens of GB). An SSD is strongly preferred.

You do **not** need to install Python first: the installer brings its own. It does need `git`; it uses one
you already have, fetches a small portable copy on Windows if you have none, and on Linux/macOS asks you to
install it first (one line: `sudo apt install git`, `sudo dnf install git`, or `brew install git`). Before
it downloads anything it shows a notice of what it will install and from where, and asks you to confirm.

## Step 1: Install

**Windows:** download
**[HordeWorker-Setup.exe](https://github.com/Haidra-Org/horde-worker-reGen/releases/latest/download/HordeWorker-Setup.exe)**,
double-click it, and click through the wizard. It shows what it will install (and the third-party
licenses) and asks you to accept. The shortcut checkboxes start unticked: tick them if you want Desktop or
Start Menu shortcuts. If Windows shows "Windows protected your PC", click **More info**, then
**Run anyway**; the installer simply is not code-signed yet.

**Linux:** paste this into a terminal and press Enter:

```bash
curl -LsSf https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.sh | sh
```

It prints what it will install and asks you to confirm (answer `y`). If you run it somewhere with no
terminal to answer (a script or pipe with no console), re-run it as
`curl -LsSf .../install.sh | HORDE_WORKER_ASSUME_YES=1 sh` to accept up front.

Other ways to install (a scripted PowerShell command, a manual clone) are in the
[install guide](../how-to/install.md), but the two above are the easiest.

## Step 2: Let it set itself up

The first run builds its environment: it downloads Python and PyTorch, which takes a few minutes. When
that finishes it opens a **dashboard in your web browser**. This is where you will control the worker.

If a window asks about your network or firewall, allowing local access is enough; the dashboard runs on
your own machine.

## Step 3: Follow the setup wizard

The dashboard greets you with a short wizard. It asks for:

1. **Your API key and a worker name.** Paste the key from registration. The worker name is how your
   contribution shows up on the horde, so pick something recognisable. If the name is already taken you
   will be told, just choose another.
2. **Which models to serve.** A sensible default is chosen for your GPU. You can accept it for now and
   change it later.
3. **An optional benchmark.** This briefly tests your machine and tunes the settings for you. It is
   worth doing, but you can skip it and run it later from the **Benchmark** tab.

When you finish, choose **Start**.

## Step 4: Wait for the models to download

After you start, the worker downloads the models you picked. The dashboard switches to the
**Downloads** tab so you can watch the progress bars. On a first run this can take 30 to 60 minutes
depending on what you chose and how fast your connection is.

You do not have to watch. The worker begins serving each model the moment it finishes downloading, so it
starts earning before everything is done. Just leave the window open.

## Step 5: Watch it work

Switch to the **Overview** tab. Once a model is ready and a request comes in, you will see jobs being
submitted and your **kudos per hour** tick up. That is it: you are contributing to the AI Horde.

New workers earn at a reduced rate for about a week (some kudos are held in escrow) until the horde
trusts them, so do not worry if early numbers look modest.

## Keeping it running

- The window you launched from **is** the worker. Closing it (or pressing `Ctrl+C` in it) stops the
  worker. Closing just the browser tab is fine; the worker keeps running and you can reopen the
  dashboard to reconnect.
- To start it again later: on Windows, use the **AI Horde Worker** shortcut if you created one, otherwise
  run `horde-worker.cmd` in the install folder; on Linux, run `./horde-worker.sh` in the install folder.
  The slow one-time setup only happens once, so reopening is quick, and your settings are remembered.

## Where to go next

- [Configure the worker for your GPU](../how-to/configure-for-your-gpu.md): squeeze out more performance.
- [Use the dashboard](../how-to/use-the-dashboard.md): a tour of every tab and shortcut.
- [Update the worker](../how-to/update-the-worker.md): stay current.
- [Troubleshooting](../how-to/troubleshoot.md): if something looks wrong.
- Questions? Join [#local-workers on Discord](https://discord.com/channels/781145214752129095/1076124012305993768).
