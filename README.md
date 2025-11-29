# Conductor Worktree Tracker

A terminal CLI tool that monitors [Conductor](https://conductor.app)'s SQLite database and displays worktrees you've visited with growing time indicators and PR hierarchy visualization.

## Quick Start

```bash
# Run with defaults (radial view, today's activity)
python3 tracker.py

# Run once and exit (for testing)
python3 tracker.py --once

# Colorblind mode with 5-minute stale threshold
python3 tracker.py --colorblind --stale 5

# Skip PR fetching for faster startup
python3 tracker.py --no-prs
```

## Configuration

All settings can be configured via `config.json` in the same directory as `tracker.py`. CLI arguments override config file values.

### config.json

```json
{
  "stale_minutes": 30,
  "colorblind": false,
  "interval_seconds": 2.0,
  "show_hierarchy": true,
  "tree_layout": false,
  "fetch_prs": true,
  "since": "today",
  "debug": false,
  "pr_refresh_minutes": 5
}
```

### Configuration Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `stale_minutes` | int | `30` | Minutes before a worktree is marked as stale. Stale worktrees show a warning indicator and full bar. |
| `colorblind` | bool | `false` | Use protanopia-friendly color scheme (blue/cyan instead of green/red, ⚠ icon for stale). |
| `interval_seconds` | float | `2.0` | How often to poll the Conductor database for changes (in seconds). |
| `show_hierarchy` | bool | `true` | Show PR-based hierarchy structure. When false, shows flat list. |
| `tree_layout` | bool | `false` | Use traditional tree layout. When false (default), uses compact radial/column layout. |
| `fetch_prs` | bool | `true` | Query GitHub for open PRs to determine hierarchy. Disable for faster startup. |
| `since` | string | `"today"` | Activity lookback period. Use `"today"` for all activity since midnight, or a number for hours (e.g., `"2"` for last 2 hours). |
| `debug` | bool | `false` | Show debug output for click detection and PR hierarchy building. |
| `pr_refresh_minutes` | int | `5` | How often to refresh PR data from GitHub while running (in minutes). |

## CLI Arguments

CLI arguments override config.json values:

```
python3 tracker.py [options]

--stale MINUTES      Minutes before worktree shows as stale
--colorblind         Use protanopia-friendly color scheme
--interval SECONDS   Database polling interval in seconds
--no-hierarchy       Disable PR hierarchy, show flat list
--tree               Use tree layout instead of default radial
--no-prs             Skip GitHub PR queries (faster startup)
--once               Run once and exit (for testing)
--since VALUE        Activity lookback: "today" or hours
--debug              Show debug output
```

## Display Format

### Radial View (default)

Compact column-based layout with repos side-by-side:

```
CONDUCTOR WORKTREE TRACKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● gas-analyzer-rs                     ● conductor-cli
├ sacramento █ now                    └ port-louis-v4 ███ 5m ago
└ san-jose █ now

● specs
└ andorra-v2 █ now
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tracking: 4 worktrees | Session: 15m
```

### Tree View (`--tree`)

Traditional tree layout with full hierarchy:

```
CONDUCTOR WORKTREE TRACKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

gas-analyzer-rs
├── sacramento (#76)                  now  █  @ 22:45
└── san-jose                          now  █  @ 22:45

conductor-cli
└── port-louis-v4                  5m ago  ███  @ 22:40
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tracking: 4 worktrees | Session: 15m
```

## Visual Indicators

### Time Bars

The bar grows as time since last activity increases:

| Time | Bar | Color (normal) | Color (colorblind) |
|------|-----|----------------|-------------------|
| 0-5 min | `█` to `██` | Green | Blue |
| 5-15 min | `███` to `█████` | Yellow | Cyan |
| 15-30 min | `██████` to `████████` | Orange | White |
| 30+ min | `██████████` | Red + `!` | Magenta + `⚠` |

### Hierarchy Symbols

- `●` - Repository name
- `├` / `└` - Tree connectors
- `#123` - PR number
- `(branch-name)` - Intermediate branch without worktree (dimmed)

## How It Works

1. **On startup**: Loads workspaces from Conductor's SQLite database, queries GitHub for open PRs
2. **Every N seconds**: Polls database for `updated_at` changes to detect workspace switches
3. **PR Hierarchy**: Uses GitHub PR base/head relationships to build parent-child tree
4. **Click Detection**: When you switch to a workspace in Conductor, its `updated_at` timestamp changes

## Dependencies

- Python 3.8+
- `tqdm` (optional, for progress bars): `pip install tqdm`
- `gh` CLI (for GitHub PR queries): `brew install gh`

## Data Sources

- **Conductor DB**: `~/Library/Application Support/com.conductor.app/conductor.db`
- **GitHub PRs**: via `gh pr list` CLI command
