# Textual TUI Interface (Experimental)

horde-worker-reGen now includes an optional Textual-based Terminal User Interface (TUI) for monitoring your worker in real-time.

## Features

The TUI provides a comprehensive dashboard displaying:

- **Worker Header**: Worker name, version, uptime, and total kudos
- **Configuration Panel**: Max threads, queue size, batch size, max power, performance mode, and safety settings
- **Process Status Panel**: Real-time status of all inference and safety processes
- **Job Queue Panel**: Current jobs pending inference, in progress, and pending safety checks
- **Session Statistics Panel**: Jobs popped, submitted, faulted, process recoveries, and estimated kudos/hour
- **System Resources Panel**: GPU and system RAM usage
- **Active Features Panel**: Visual indicators for enabled features (img2img, LoRA, ControlNet, etc.)
- **Activity Log**: Scrollable log of recent events and messages

## Installation

The TUI requires the `textual` package:

```bash
pip install -r requirements.txt
```

Or install textual separately:

```bash
pip install textual
```

## Usage

To enable the TUI, use the `--tui` flag when starting the worker:

```bash
python run_worker.py --tui
```

Or with the shell script:

```bash
./horde-bridge.sh
# Then add --tui to the python command
```

## Keyboard Shortcuts

- **q**: Quit the TUI and worker
- **d**: Toggle dark mode
- **Ctrl+C**: Gracefully shutdown the worker

## Notes

- The TUI is experimental and may have bugs
- All worker functionality remains the same; the TUI is purely a visualization layer
- The TUI runs in a separate thread and updates every 2 seconds
- If textual is not installed, the worker will start normally without the TUI and log an error

## Troubleshooting

If the TUI doesn't start:
1. Ensure textual is installed: `pip install textual`
2. Check the console logs for error messages
3. Try running without `--tui` flag to use standard console output

## Comparison with Standard Output

**Standard output**: Text-based log messages scrolling in the console
**TUI mode**: Organized dashboard with real-time updates and structured panels

Both modes provide the same functionality; the TUI simply presents the information in a more organized and visually appealing way.
