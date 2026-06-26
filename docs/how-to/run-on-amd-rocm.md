# Run on AMD (ROCm)

AMD support is **experimental**. The installer follows ComfyUI's backend support:

- Linux uses the stable PyTorch ROCm wheel index.
- Windows uses AMD's ROCm Windows PyTorch wheels for supported Radeon/Ryzen AI devices.
- DirectML remains unavailable because its PyTorch line is too old for this worker.

For help, join the
[AMD discussion on Discord](https://discord.com/channels/781145214752129095/1076124012305993768).

## Linux

Use the ROCm variants of the scripts in place of the standard ones:

- `update-runtime-rocm.sh` to install or update.
- `horde-bridge-rocm.sh` to run the worker.

Everything else (config, models, the dashboard) works as on any Linux install. See
[Install](install.md) and [Configure for your GPU](configure-for-your-gpu.md).

## Windows AMD ROCm

The Windows installer detects supported AMD adapter names and installs the `rocm-windows` profile. That
profile syncs the universal base environment, then overlays AMD's official ROCm Windows wheels for
`amd_smi`, `hip_sdk`, `torch`, `torchvision`, and `torchaudio`.

Current auto-detection is intentionally conservative and covers the device names in AMD's Radeon/Ryzen
AI Windows compatibility tables, including:

- Radeon RX 7900 / 7700 and Radeon PRO W7900
- Radeon RX 9070 / 9060 and Radeon AI PRO R9700
- Ryzen AI 9 HX 370-class devices and Ryzen AI Max / Radeon 8050S / 8060S

If your supported card is not recognized, force the token before running the installer or update script:

```powershell
$env:HORDE_WORKER_BACKEND = "rocm-windows"
.\update-runtime.cmd
```

These profiles install a lean base by default. Optional feature extras can be enabled with
`HORDE_WORKER_FEATURES`, as described in [Compute backends](../explanation/compute_backends.md).

## DirectML (Windows): unavailable

DirectML let AMD and Intel GPUs run on Windows without CUDA. It is **temporarily unavailable**: the
`torch-directml` build is pinned to an older PyTorch (the 2.4 era) that is incompatible with the
version this worker now requires. The `update-runtime-directml` and `horde-bridge-directml` scripts
were removed rather than left to fail with a confusing error.

If you have an unsupported AMD or Intel GPU on Windows and no CUDA option, run the worker on Linux with
ROCm for now, either on a native Linux install or through WSL (below). This section will be restored
when a compatible DirectML build is available.

## AMD ROCm inside Windows (WSL)

> This path is **highly experimental and developer-grade.** It involves installing Windows build
> tools, compiling a bridge library from source, and editing system files. Expect it to be slower than
> native Linux, and budget time for troubleshooting. If that does not sound like you, prefer a native
> Linux install or an NVIDIA GPU.

WSL will probably be slower than a native Linux system. Unless you have a lot of RAM you may also hit
memory limits; you might need to raise the WSL memory limit or configure swap, as described in the
[WSL configuration docs](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

### Step 1: Install the Windows build tools and SDK (on the Windows host)

The Linux compiler needs the Microsoft development headers to talk to Windows.

- Download the **Visual Studio Build Tools** from Microsoft's developer website.
- Run the installer and select the **Desktop development with C++** workload.
- In the details panel on the right, ensure **Windows 11 SDK (10.0.26100.0)** is checked, then install.
- Confirm the headers exist at
  `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\`. You need this path in Step 5.

### Step 2: System setup

- Make sure Windows and your AMD drivers are up to date.
- Enable and install WSL. Open a command prompt **as Administrator** and run:

  ```text
  wsl --install -d Ubuntu-24.04
  ```

  If that errors about WSL not being installed, run `wsl --install` first, then the command above.

- If you previously used Ubuntu-24.04 in WSL, reset the image (this deletes data inside it, so save
  anything important elsewhere first):

  ```text
  wsl --unregister Ubuntu-24.04
  ```

- When prompted, set a simple unix username and a password (the password prompt shows no characters as
  you type, which is normal). To reopen Ubuntu later, search for `Ubuntu 24.04` in the Start Menu or
  run `wsl -d Ubuntu-24.04`.

### Step 3: Install Ubuntu build dependencies

In the Ubuntu 24.04 WSL terminal:

```bash
sudo apt update && sudo apt install -y build-essential cmake git ca-certificates wget gpg
```

### Step 4: Add the ROCm repo, pin it, and install

Run these blocks in order:

```bash
# 1. Download and add the AMD GPG key
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# 2. Add the ROCm repository for Ubuntu 24.04 (Noble)
sudo tee /etc/apt/sources.list.d/rocm.list << EOF
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/latest noble main
EOF

# 3. Pin apt to prefer AMD's repo over Ubuntu's defaults
echo -e "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600" | sudo tee /etc/apt/preferences.d/rocm-pin-600

# 4. Update and install the ROCm stack (this downloads several gigabytes)
sudo apt update
sudo apt install -y rocm rocm-dev
```

### Step 5: Clone and build librocdxg

This bridge library points the Linux compiler at your mounted Windows `C:` drive so it can read the
Windows SDK headers.

```bash
# 1. Clone the repository
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg

# 2. Set the Windows SDK path.
# IMPORTANT: change "10.0.26100.0" to match the folder version you found in Step 1.
export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/'

# 3. Configure the build
mkdir -p build
cd build
cmake .. -DWIN_SDK="${win_sdk}/shared"

# 4. Compile and install
make
sudo make install
```

### Step 6: Enable detection and verify

```bash
# 1. Add the environment variables to your shell profile
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc

# 2. Reload your profile
source ~/.bashrc
```

Then run the check:

```bash
rocminfo
```

It should print `WSL environment detected.` followed by HSA system attributes and a list of HSA
agents.

### Install the worker

From here, follow the standard Linux ROCm steps above (`update-runtime-rocm.sh`, then
`horde-bridge-rocm.sh`).
