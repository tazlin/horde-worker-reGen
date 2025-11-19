# Textual UI Dashboard

The Horde Worker reGen now includes an optional Textual UI (TUI) dashboard that provides a rich, interactive terminal interface for monitoring your worker in real-time.

## Features

The TUI provides six main screens accessible via keyboard shortcuts:

### 1. Dashboard (Press `1`)
The main overview screen showing:
- Worker status and uptime
- Process overview (active/total processes)
- Job queue statistics
- Kudos earnings and performance metrics
- Real-time alerts and warnings
- Quick stats: session start, idle time, slowdowns, failures

### 2. Processes (Press `2`)
Detailed view of all worker processes showing:
- Process state (WAITING_FOR_JOB, INFERENCE_STARTING, etc.)
- Currently loaded model and model state
- Progress percentage with visual progress bars
- RAM and VRAM usage per process
- Active job ID (if processing)
- Safety process indicator

### 3. Jobs (Press `3`)
Job queue management view displaying:
- Pending inference jobs
- Jobs in progress
- Jobs pending safety check
- Jobs being safety checked
- Jobs pending submit
- Queue utilization percentage
- Performance metrics (faults, slowdowns, failures, recoveries)

### 4. Stats (Press `4`)
Detailed kudos and performance statistics:
- Session kudos earned (large display)
- Kudos per hour rate
- Total user kudos balance
- Username
- Session uptime and percentage
- Total idle time
- System health indicators

### 5. Config (Press `5`)
Worker configuration overview:
- Worker name and basic settings
- Max threads, queue size, max power
- Memory settings (RAM/VRAM to leave free)
- Performance modes (high/moderate/low memory)
- Safety and security settings
- List of all loaded models

### 6. Logs (Press `6`)
Real-time log viewer with:
- Color-coded log levels (ERROR=red, WARNING=yellow, INFO=green, DEBUG=blue)
- Timestamp for each message
- Auto-scrolling to latest messages
- Press `c` to clear the log display

## Installation

To use the TUI, you need to install the optional `textual` dependency:

```bash
# Option 1: Install with pip
pip install textual

# Option 2: Install from requirements file
pip install -r requirements.tui.txt

# Option 3: Install with optional dependencies
pip install -e .[tui]
```

## Usage

To start the worker with the TUI dashboard, simply add the `--tui` flag:

```bash
# Standard usage
python run_worker.py --tui

# With other options
python run_worker.py --tui -v

# With custom worker name
python run_worker.py --tui -n my-worker-01
```

If the `textual` package is not installed and you use the `--tui` flag, the worker will display a warning and fall back to standard console mode.

## Keyboard Shortcuts

Global shortcuts available on all screens:

- `1` - Switch to Dashboard
- `2` - Switch to Processes view
- `3` - Switch to Jobs view
- `4` - Switch to Statistics view
- `5` - Switch to Configuration view
- `6` - Switch to Logs view
- `r` - Refresh current screen manually
- `q` - Quit the application
- `c` - Clear logs (only on Logs screen)

## Update Intervals

Different screens refresh at different intervals for optimal performance:

- Dashboard: Every 2 seconds
- Processes: Every 1 second (real-time)
- Jobs: Every 1 second (real-time)
- Stats: Every 1 second (real-time)
- Config: Every 5 seconds
- Logs: Every 0.5 seconds (real-time)

## Features Inspired by Existing Logging

The TUI is designed based on the existing logging scheme in the worker:

- **StatusReporter patterns**: The dashboard layout mirrors the structure of status reports
- **Process states**: All 20+ process states from `HordeProcessState` are displayed
- **Model states**: Shows the 6 model load states (DOWNLOADING → LOADED_IN_VRAM → IN_USE)
- **Kudos logging**: Real-time kudos tracking as implemented in `KudosLogger`
- **Maintenance mode**: Displays maintenance mode alerts like `MaintenanceModeMessenger`
- **Job queues**: Shows all queue types (pending_inference, in_progress, pending_safety_check, etc.)

## Technical Details

### Data Source
All data is sourced from the `HordeWorkerProcessManager` instance:
- Process states from `_process_map`
- Job queues from `jobs_pending_inference`, `jobs_in_progress`, etc.
- Statistics from `kudos_generated_this_session`, `_num_process_recoveries`, etc.
- Configuration from `bridge_data`

### Architecture
- **TUIDataProvider**: Bridges the worker process manager and TUI screens
- **Individual Screens**: Each screen is a separate class for modularity
- **Custom Widgets**: Reusable widgets like `ProcessCard`, `StatBox`, `StatusPanel`
- **Async Integration**: Runs concurrently with the worker process manager

### Log Integration
The TUI integrates with loguru to capture and display log messages in real-time. This is done via a custom loguru sink that forwards messages to the TUI's log buffer.

## Troubleshooting

### TUI won't start
- Ensure `textual` is installed: `pip install textual`
- Check terminal compatibility (most modern terminals work)
- Try updating textual: `pip install --upgrade textual`

### Display issues
- Ensure your terminal supports 256 colors
- Try resizing the terminal window
- Some terminals work better than others (iTerm2, Windows Terminal, etc.)

### Performance concerns
- The TUI adds minimal overhead to the worker
- Data updates are throttled to reasonable intervals
- The TUI runs asynchronously and won't block worker operations

### Exiting the TUI
- Press `q` to quit gracefully
- The worker will continue shutting down normally
- `Ctrl+C` will also work as expected

## Compatibility

The TUI is compatible with:
- Python 3.10+
- Linux, macOS, and Windows
- Most modern terminal emulators
- SSH sessions
- Screen/tmux multiplexers

## Fallback Behavior

If you start the worker with `--tui` but:
- `textual` is not installed → Falls back to standard console mode with a warning
- TUI crashes or errors → Falls back to standard console mode with error details
- Terminal doesn't support TUI → Standard console mode will work normally

This ensures the worker always runs, even if the TUI can't be initialized.
