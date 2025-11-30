# Conductor Worktree Tracker

A terminal CLI tool that monitors [Conductor](https://conductor.app)'s SQLite database and displays worktrees you've visited with growing time indicators, PR hierarchy visualization, and CI/CD status.

## Features

- **Radial Layout**: Circular display with repos positioned around an ellipse, trees branching outward
- **PR Integration**: Shows PR titles and numbers, with hierarchy based on PR base/head relationships
- **CI/CD Status**: Live status indicators for GitHub Actions (✓ pass, ✗ fail, animated spinner for pending)
- **Session Status**: Shows which worktrees have active Claude sessions (working/idle)
- **Smart Branch Detection**: Detects actual git branches even when Conductor's database is stale

## Quick Start

```bash
# Run with defaults (radial view, today's activity)
python3 tracker.py

# Run once and exit (for testing)
python3 tracker.py --once

# 5-minute stale threshold
python3 tracker.py --stale 5

# Skip PR fetching for faster startup
python3 tracker.py --no-prs

# Show diagnostic info (terminal size, cache status)
python3 tracker.py --diag
```

## Configuration

All settings can be configured via `config.json` in the same directory as `tracker.py`. CLI arguments override config file values.

### config.json

```json
{
  "stale_minutes": 30,
  "interval_seconds": 2.0,
  "show_hierarchy": true,
  "tree_layout": false,
  "fetch_prs": true,
  "since": "today",
  "debug": false,
  "pr_refresh_minutes": 5,
  "show_status": true,
  "pomodoro_enabled": false,
  "pomodoro_work_minutes": 25,
  "pomodoro_break_minutes": 5
}
```

### Configuration Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `stale_minutes` | int | `30` | Minutes before a worktree is marked as stale (red + `!`). |
| `interval_seconds` | float | `2.0` | How often to poll the Conductor database for changes. |
| `show_hierarchy` | bool | `true` | Show PR-based hierarchy structure. When false, shows flat list. |
| `tree_layout` | bool | `false` | Use traditional tree layout instead of radial. |
| `fetch_prs` | bool | `true` | Query GitHub for open PRs to determine hierarchy and show CI status. |
| `since` | string | `"today"` | Activity lookback: `"today"` for since midnight, or hours (e.g., `"2"`). |
| `debug` | bool | `false` | Show debug output for click detection and PR hierarchy building. |
| `pr_refresh_minutes` | int | `5` | How often to refresh PR data from GitHub. |
| `show_status` | bool | `true` | Show session status indicators (working/idle/error). |
| `pomodoro_enabled` | bool | `false` | Enable pomodoro timer mode with reflection prompts. |
| `pomodoro_work_minutes` | int | `25` | Duration of work sessions in pomodoro mode. |
| `pomodoro_break_minutes` | int | `5` | Duration of breaks in pomodoro mode. |

## CLI Arguments

CLI arguments override config.json values:

```
python3 tracker.py [options]

--stale MINUTES      Minutes before worktree shows as stale
--interval SECONDS   Database polling interval in seconds
--no-hierarchy       Disable PR hierarchy, show flat list
--tree               Use tree layout instead of default radial
--no-prs             Skip GitHub PR queries (faster startup)
--once               Run once and exit (for testing)
--since VALUE        Activity lookback: "today" or hours
--debug              Show debug output
--no-status          Hide session status indicators
--pomodoro           Enable pomodoro timer mode
--diag               Show diagnostic info (terminal size, cache status)
```

## Display Format

### Radial View (default)

Circular layout with repos positioned around an ellipse. Trees branch outward based on quadrant (up/down/left/right):

```
CONDUCTOR WORKTREE TRACKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                               ┌─ STATUS ───────────┐
                                               │ █ ⠋ 3 WORKING █    │
               ┌ ⠋ san-juan-v2 █ now           │   ○ 10 idle        │
               ● etherform                     └────────────────────┘

         conductor-cli ●           ● gas-killer-router
                                   ├ ○ feat: add configu #62 ████ 7h ago !
                                   ├ ○ fix: fail on stor #65 ███ 13m ago
                                   └ ⠋ luxembourg █ now

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tracking: 12 worktrees | Session: 15m
```

### Tree View (`--tree`)

Traditional tree layout with full hierarchy:

```
CONDUCTOR WORKTREE TRACKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

gas-analyzer-rs
├── ✓ sacramento (#76)                  now  █  @ 22:45
└── ○ san-jose                          now  █  @ 22:45

conductor-cli
└── ⠋ Add Conductor Worktree (#1)    5m ago  ███  @ 22:40
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tracking: 4 worktrees | Session: 15m
```

## Visual Indicators

### Time Bars

The bar grows as time since last activity increases:

| Time | Bar | Color |
|------|-----|-------|
| 0-5 min | `█` to `██` | Green |
| 5-15 min | `███` to `█████` | Yellow |
| 15-30 min | `██████` to `████████` | Orange |
| 30+ min | `██████████` | Red + `!` |

### Session Status Icons

Each worktree shows a status indicator based on its Claude session state:

| Icon | Status | Description |
|------|--------|-------------|
| `⠋⠙⠹⠸` | Working | Claude is actively processing (animated spinner) |
| `○` | Idle | Session is waiting for user input |
| `✗` | Error | Session encountered an error |

### CI/CD Status Icons

PRs show their GitHub Actions status:

| Icon | Status | Description |
|------|--------|-------------|
| `✓` | Pass | All checks passed (green) |
| `✗` | Fail | One or more checks failed (red) |
| `⠋⠙⠹⠸` | Pending | Checks are running (animated yellow spinner) |

CI status is cached to avoid GitHub API rate limits:
- Pending PRs: refreshed every 2 minutes
- Pass/Fail PRs: refreshed every 5 minutes
- Max 5 API calls per refresh cycle

### Hierarchy Symbols

- `●` - Repository name
- `├` / `└` - Tree connectors
- `#123` - PR number
- PR titles shown instead of branch names when available
- `(branch-name)` - Intermediate branch without worktree (dimmed)

## How It Works

1. **On startup**: Loads workspaces from Conductor's SQLite database, detects actual git branches, queries GitHub for open PRs
2. **Branch detection**: Runs `git branch --show-current` in each worktree to detect the real branch (Conductor's database may be stale if you switched branches)
3. **Every N seconds**: Polls database for `updated_at` changes to detect workspace switches
4. **PR Hierarchy**: Uses GitHub PR base/head relationships to build parent-child tree
5. **CI Status**: Queries `gh pr checks` for each PR with rate-limited caching

## Dependencies

- Python 3.8+
- `tqdm` (optional, for progress bars): `pip install tqdm`
- `gh` CLI (for GitHub PR queries and CI status): `brew install gh`

## Data Sources

- **Conductor DB**: `~/Library/Application Support/com.conductor.app/conductor.db`
- **GitHub PRs**: via `gh pr list` CLI command
- **CI Status**: via `gh pr checks` CLI command
- **Git branches**: via `git branch --show-current` in each worktree
