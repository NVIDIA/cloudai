# CloudAI Web UI

A simple web interface for viewing CloudAI test results.

## Overview

The CloudAI Web UI provides a basic dashboard to browse test scenarios and their results. It's designed as a minimal proof-of-concept that can be extended for more advanced functionality.

## Features

- **Main Dashboard**: Lists all test scenarios with key metadata and statistics
- **Responsive Design**: Clean, modern interface using Bootstrap 5
- **Data Abstraction**: Ready for future database integration

## Installation & Usage

1. Install UI dependencies:
   ```bash
   uv sync --extra ui
   ```

2. Start the UI (uses `./results` by default):
   ```bash
   uv run cloudai-ui
   ```

3. With custom results directory:
   ```bash
   uv run cloudai-ui --results-dir /path/to/results
   ```

4. Access in browser: `http://localhost:5000`

## Command Line Options

- `--results-dir` - Path to results directory (default: results)
- `--host` - Host to bind to (default: localhost)
- `--port` - Port to bind to (default: 5000)
- `--debug` - Enable Flask debug mode

## Architecture

### Data Layer
- `DataProvider` - Abstract base class for data access
- `LocalFileDataProvider` - Current implementation reading from local files
- Ready for future `DatabaseDataProvider` implementation

### File Structure Recognition

The UI automatically detects CloudAI result structures:

```
results/
├── scenario_name_YYYY-MM-DD_HH-MM-SS/
│   ├── test_run_name/
│   │   ├── 0/  # iteration directories
│   │   │   ├── test-run.toml
│   │   │   └── ...
│   │   └── trajectory.csv
│   └── slurm-job.toml
```

## Future Extensions

The architecture supports easy addition of:
- Individual scenario and test run detail pages
- Various report type viewers (trajectory analysis, performance results, logs, etc.)
- Database backends
- Real-time updates
- Advanced filtering and search

## Dependencies

- Flask - Web framework
- Bootstrap 5 - UI styling
- Font Awesome - Icons

## License

Apache 2.0 - See LICENSE file for details.