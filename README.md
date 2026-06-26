# AI Horde Worker reGen

[AI Horde](https://aihorde.net/) is a free image-generation service, kept running by people who
donate spare GPU time. No account fees, no usage credits, no waitlist. This is the software that
lets you contribute.

When your worker completes jobs for others, you earn **kudos**, which move you up the queue for
your own requests. Many contributors run it for that reason. Many others run it simply because they
think free access to this kind of tool is worth supporting.

The service is run by [Haidra](https://haidra.net/mission/), a registered nonprofit association.
It has no investors and distributes no profits; our goal is to keep AI image generation accessible
to students, artists, researchers, and anyone who cannot or prefers not to pay commercial rates.

## Get started

You do not need Python or any other software installed first.

**Windows:** Download
**[HordeWorker-Setup.exe](https://github.com/Haidra-Org/horde-worker-reGen/releases/latest/download/HordeWorker-Setup.exe)**
and double-click it.

**Linux / macOS:** Paste this into a terminal:

```bash
curl -LsSf https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.sh | sh
```

**Windows (command line):** paste
`irm https://raw.githubusercontent.com/Haidra-Org/horde-worker-reGen/main/install.ps1 | iex`
into PowerShell.

On first run the installer downloads its own Python and PyTorch, opens a dashboard in your
browser, and walks you through entering your [AI Horde API key](https://aihorde.net/register)
(free, instant registration) and picking which models to serve. Models download in the background
while the worker starts. Re-running the same command later updates the software.

Desktop shortcuts are optional - the installers ask before creating any, and the `.exe`'s
shortcut checkboxes start unticked.

The [Getting started guide](docs/tutorials/getting-started.md) has screenshots and answers
common first-run questions.

## Will it run on my machine?

8 GB or more of VRAM is recommended. CPU-only installs work but are too slow to serve real jobs for image generation, but can be used as alchemists.

| Your setup | Status | Notes |
| ------------ | -------- | ------- |
| **Windows + NVIDIA GPU** | Supported | The easiest path. Full features. |
| **Linux + NVIDIA GPU** | Supported | Full features. |
| **Linux + AMD GPU** | Experimental | Uses ROCm (installed ad-hoc). Core features; opt into the rest. See [Run on AMD ROCm](docs/how-to/run-on-amd-rocm.md). |
| **Linux + Intel Arc / XPU** | Experimental | Core features. Install the XPU torch wheel ad-hoc (see [Compute backends](docs/explanation/compute_backends.md)). |
| **Apple Silicon (macOS / MPS)** | Experimental | Core features on the default macOS PyTorch wheel; much slower than a discrete GPU. |
| **Windows + AMD GPU** | Experimental | Supported for the Radeon/Ryzen AI devices in AMD's ROCm Windows compatibility matrix. See [Run on AMD ROCm](docs/how-to/run-on-amd-rocm.md). |
| **Windows + Intel GPU** | Not straightforward | DirectML is temporarily unavailable, so the realistic route is Linux. |
| **No supported GPU (CPU)** | CPU only | Installs and runs, only alchemy recommended. |

Core image generation and the safety classifier are pure PyTorch and run on every backend above.
Two extras - **post-processing** (upscale, face-fix, background removal) and **controlnet** - depend
on native packages that have no wheels for some accelerators. On non-NVIDIA setups the worker detects
this automatically, stops advertising those features, and never accepts a job it cannot run. You can
opt in where wheels exist; see [Compute backends](docs/explanation/compute_backends.md).

If your setup is listed as experimental or not straightforward, read the relevant guide before you
start so you know what is involved.

## Guides

| I want to... | Guide |
|--------------|-------|
| Get a good setup for my GPU | [Configure for your GPU](docs/how-to/configure-for-your-gpu.md) |
| Learn the dashboard | [Use the dashboard](docs/how-to/use-the-dashboard.md) |
| Keep it running after I close the window, or reattach | [Closing and reattaching](docs/how-to/use-the-dashboard.md#closing-and-reattaching) |
| Run without a UI (servers, automation) | [Run headless](docs/how-to/run-headless.md) |
| Run in a container | [Run in Docker](docs/how-to/run-in-docker.md) |
| Use several GPUs | [Run multiple GPUs](docs/how-to/run-multiple-gpus.md) |
| Serve my own models | [Add custom models](docs/how-to/add-custom-models.md) |
| Run on AMD or without NVIDIA | [Run on AMD ROCm](docs/how-to/run-on-amd-rocm.md) |
| Keep it up to date | [Update the worker](docs/how-to/update-the-worker.md) |
| Fix a problem | [Troubleshooting](docs/how-to/troubleshoot.md) |

The full documentation site starts at [docs/](docs/index.md).

## Understand or contribute

- **[Documentation home](docs/index.md)**: start here.
- **[Architecture overview](docs/explanation/architecture.md)**: what runs where, the process model,
  and IPC.
- **[Codebase map](docs/reference/codebase-map.md)**: a file-to-responsibility quick reference.
- **[Contributing](CONTRIBUTING.md)**: development setup and guidelines.

## Support

- Help: [#local-workers on Discord](https://discord.com/channels/781145214752129095/1076124012305993768)
  or [open an issue](https://github.com/Haidra-Org/horde-worker-reGen/issues).
- News and releases: the [AI Horde Discord](https://discord.gg/3DxrhksKzn).

## Model usage and licenses

Many bundled models use the
[CreativeML OpenRAIL License](https://huggingface.co/spaces/CompVis/stable-diffusion-license). Please
review it before use.

## AI Assistance Disclosure

This project uses AI assistance in its development. Models include those provided through Github
(including its FIM "autocomplete" feature), OpenAI, Anthropic and Deepseek. We encourage any contributors
to disclose their use of AI assistance in their contributions, but it is not required. We do, however, require
that all contributions, whether AI-assisted or not, adhere to our contribution guidelines and code of conduct
as outlined in the CONTRIBUTING.md file. Further, we do not accept unattended or automated AI contributions;
a human must review and make a best effort to understand and verify the contribution before it is accepted.
If you have any questions about our AI assistance policy, please feel free to reach out to us on our Discord
server or through our issue tracker.
