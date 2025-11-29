#!/usr/bin/env python3
"""
Conductor Worktree Activity Tracker

Monitors Conductor's SQLite database and displays worktrees you've visited
with growing time indicators and repository/branch hierarchy.
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


# ============================================================================
# Configuration
# ============================================================================

CONDUCTOR_DB = Path.home() / "Library/Application Support/com.conductor.app/conductor.db"
CONFIG_FILE = Path(__file__).parent / "config.json"

# Default configuration values
DEFAULT_CONFIG = {
    "stale_minutes": 30,
    "colorblind": False,
    "interval_seconds": 2.0,
    "show_hierarchy": True,
    "tree_layout": False,
    "fetch_prs": True,
    "since": "today",  # "today" or number of hours
    "debug": False,
    "pr_refresh_minutes": 5,
}


def load_config() -> dict:
    """Load configuration from config.json, with defaults as fallback."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                user_config = json.load(f)
                config.update(user_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config.json: {e}", file=sys.stderr)
    return config


def get_since_cutoff(since_value) -> datetime:
    """Convert 'since' config value to a datetime cutoff.

    Args:
        since_value: Either "today" or a number of hours

    Returns:
        datetime: The cutoff time in UTC
    """
    from datetime import timezone
    now = utc_now()

    if since_value == "today":
        # Get start of today in local time, then convert to UTC
        local_now = datetime.now()
        local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        # Convert local midnight to UTC (naive)
        utc_offset = local_now.astimezone().utcoffset()
        if utc_offset:
            return local_midnight - utc_offset
        return local_midnight
    else:
        # Treat as hours
        from datetime import timedelta
        return now - timedelta(hours=float(since_value))

# Unicode bar characters (1/8 increments)
BAR_CHARS = " ▏▎▍▌▋▊▉█"

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Traditional colors
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    ORANGE = "\033[38;5;208m"
    RED = "\033[31m"

    # Colorblind-friendly (protanopia)
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    MAGENTA = "\033[35m"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Workspace:
    id: str
    name: str  # directory_name
    branch: str
    parent_branch: Optional[str]
    repo_id: str
    repo_name: str
    remote_url: str
    default_branch: str
    updated_at: datetime


@dataclass
class ClickRecord:
    workspace_id: str
    name: str
    repo_name: str
    branch: str
    first_seen: datetime
    last_seen: datetime


def _utc_now_factory():
    """Factory for dataclass default."""
    from datetime import timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class Session:
    started_at: datetime = field(default_factory=_utc_now_factory)
    clicks: dict[str, ClickRecord] = field(default_factory=dict)
    last_updated_at: dict[str, datetime] = field(default_factory=dict)


@dataclass
class PRInfo:
    number: int
    head_branch: str
    base_branch: str


# ============================================================================
# Database Access
# ============================================================================

def utc_now() -> datetime:
    """Get current time in UTC (naive datetime for comparison with DB)."""
    from datetime import timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_workspaces() -> list[Workspace]:
    """Load all workspaces from Conductor database."""
    if not CONDUCTOR_DB.exists():
        print(f"Error: Conductor database not found at {CONDUCTOR_DB}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(f"file:{CONDUCTOR_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            w.id,
            w.directory_name,
            w.branch,
            w.parent_branch,
            w.repository_id,
            w.updated_at,
            r.name as repo_name,
            r.remote_url,
            r.default_branch
        FROM workspaces w
        JOIN repos r ON w.repository_id = r.id
        WHERE w.state = 'ready'
        ORDER BY w.updated_at DESC
    """)

    workspaces = []
    for row in cursor.fetchall():
        # Parse UTC timestamp from database (stored without timezone info)
        updated_at = datetime.fromisoformat(row["updated_at"].replace("Z", ""))

        workspaces.append(Workspace(
            id=row["id"],
            name=row["directory_name"],
            branch=row["branch"],
            parent_branch=row["parent_branch"],
            repo_id=row["repository_id"],
            repo_name=row["repo_name"],
            remote_url=row["remote_url"] or "",
            default_branch=row["default_branch"] or "main",
            updated_at=updated_at,
        ))

    conn.close()
    return workspaces


# ============================================================================
# GitHub PR Integration
# ============================================================================

def parse_github_remote(remote_url: str) -> Optional[tuple[str, str]]:
    """Extract owner/repo from GitHub remote URL."""
    # Handle SSH format: git@github.com:owner/repo.git
    ssh_match = re.match(r"git@github\.com:([^/]+)/([^.]+)(?:\.git)?", remote_url)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)

    # Handle HTTPS format: https://github.com/owner/repo.git
    https_match = re.match(r"https://github\.com/([^/]+)/([^.]+)(?:\.git)?", remote_url)
    if https_match:
        return https_match.group(1), https_match.group(2)

    return None


def get_open_prs(owner: str, repo: str) -> list[PRInfo]:
    """Query open PRs via gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "pr", "list", "--repo", f"{owner}/{repo}", "--state", "open",
             "--json", "number,headRefName,baseRefName"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        prs = json.loads(result.stdout)
        return [
            PRInfo(
                number=pr["number"],
                head_branch=pr["headRefName"],
                base_branch=pr["baseRefName"],
            )
            for pr in prs
        ]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []


def get_all_prs(
    workspaces: list[Workspace],
    show_progress: bool = False,
) -> dict[str, list[PRInfo]]:
    """Get PRs for all unique repos."""
    # Build list of unique repos first
    repos_to_fetch: list[tuple[str, str, str]] = []  # (repo_name, owner, repo)
    repos_seen = set()

    for ws in workspaces:
        parsed = parse_github_remote(ws.remote_url)
        if parsed and ws.repo_name not in repos_seen:
            repos_seen.add(ws.repo_name)
            owner, repo = parsed
            repos_to_fetch.append((ws.repo_name, owner, repo))

    prs_by_repo: dict[str, list[PRInfo]] = {}

    if show_progress and HAS_TQDM and repos_to_fetch:
        iterator = tqdm(
            repos_to_fetch,
            desc="  Fetching PRs",
            unit="repo",
            ncols=60,
            bar_format="{desc}: {bar} {n_fmt}/{total_fmt}",
            leave=True,
        )
    else:
        iterator = repos_to_fetch

    for repo_name, owner, repo in iterator:
        prs_by_repo[repo_name] = get_open_prs(owner, repo)

    return prs_by_repo


# ============================================================================
# Hierarchy Building
# ============================================================================

@dataclass
class TreeNode:
    name: str
    branch: str
    workspace_id: Optional[str] = None
    pr_number: Optional[int] = None
    click_record: Optional[ClickRecord] = None
    children: list["TreeNode"] = field(default_factory=list)
    is_default_branch: bool = False


def build_hierarchy(
    workspaces: list[Workspace],
    session: Session,
    prs_by_repo: dict[str, list[PRInfo]],
    include_pr_branches: bool = True,
    debug: bool = False,
) -> dict[str, TreeNode]:
    """Build hierarchy tree for each repo with active workspaces.

    If include_pr_branches is True, also include branches from PRs that don't
    have worktrees (shown without timers).
    """
    # Group workspaces by repo - include all workspaces for repos with clicks
    repos_with_clicks = set()
    for ws in workspaces:
        if ws.id in session.clicks:
            repos_with_clicks.add(ws.repo_name)

    by_repo: dict[str, list[Workspace]] = {}
    for ws in workspaces:
        if ws.repo_name in repos_with_clicks:
            by_repo.setdefault(ws.repo_name, []).append(ws)

    trees: dict[str, TreeNode] = {}

    for repo_name, repo_workspaces in by_repo.items():
        if not repo_workspaces:
            continue

        default_branch = repo_workspaces[0].default_branch
        prs = prs_by_repo.get(repo_name, [])

        # Build PR lookup: head_branch -> (base_branch, pr_number)
        pr_lookup: dict[str, tuple[str, int]] = {}
        for pr in prs:
            pr_lookup[pr.head_branch] = (pr.base_branch, pr.number)

        if debug and prs:
            print(f"[DEBUG] {repo_name}: {len(prs)} PRs")
            for pr in prs[:5]:  # Show first 5
                print(f"  PR #{pr.number}: {pr.head_branch} → {pr.base_branch}")

        # Create root node for repo
        root = TreeNode(name=repo_name, branch="", is_default_branch=False)

        # Create nodes for each workspace that was clicked
        nodes: dict[str, TreeNode] = {}
        workspace_by_branch: dict[str, Workspace] = {}

        for ws in repo_workspaces:
            workspace_by_branch[ws.branch] = ws
            if ws.id in session.clicks:
                click = session.clicks.get(ws.id)
                pr_info = pr_lookup.get(ws.branch)
                node = TreeNode(
                    name=ws.name,
                    branch=ws.branch,
                    workspace_id=ws.id,
                    pr_number=pr_info[1] if pr_info else None,
                    click_record=click,
                )
                nodes[ws.branch] = node

        # If include_pr_branches, add intermediate PR branches without worktrees
        if include_pr_branches:
            # Find all branches in the PR chain that we need
            branches_to_add = set()
            for branch in list(nodes.keys()):
                # Walk up the PR chain
                current = branch
                while current in pr_lookup:
                    parent_branch = pr_lookup[current][0]
                    if parent_branch not in nodes and parent_branch != default_branch:
                        branches_to_add.add(parent_branch)
                    current = parent_branch

            # Create nodes for PR branches without worktrees
            for branch in branches_to_add:
                pr_info = pr_lookup.get(branch)
                # Use branch name as display name (no worktree), truncate to fit
                display_name = branch[:18] if len(branch) > 18 else branch
                node = TreeNode(
                    name=f"({display_name})",  # Parens indicate no worktree
                    branch=branch,
                    workspace_id=None,
                    pr_number=pr_info[1] if pr_info else None,
                    click_record=None,  # No timer for branches without worktrees
                )
                nodes[branch] = node

        # Determine parent for each node
        attached = set()
        for branch, node in nodes.items():
            # Check for PR relationship first
            pr_info = pr_lookup.get(branch)
            if pr_info:
                parent_branch = pr_info[0]
                if parent_branch in nodes:
                    if debug:
                        print(f"  [HIERARCHY] {branch} → child of {parent_branch} (PR #{pr_info[1]})")
                    nodes[parent_branch].children.append(node)
                    attached.add(branch)
                    continue
                elif debug:
                    print(f"  [HIERARCHY] {branch} has PR to {parent_branch} but parent not in nodes")

            # Fall back to parent_branch from Conductor
            ws = workspace_by_branch.get(branch)
            if ws and ws.parent_branch and ws.parent_branch in nodes:
                if debug:
                    print(f"  [HIERARCHY] {branch} → child of {ws.parent_branch} (Conductor)")
                nodes[ws.parent_branch].children.append(node)
                attached.add(branch)
                continue

        # Attach unattached nodes to root
        for branch, node in nodes.items():
            if branch not in attached:
                root.children.append(node)

        if root.children:
            trees[repo_name] = root

    return trees


# ============================================================================
# Rendering
# ============================================================================

def format_time_ago(dt: datetime) -> str:
    """Format time as 'Xm ago' or 'Xh ago'."""
    delta = utc_now() - dt
    minutes = int(delta.total_seconds() / 60)

    if minutes < 1:
        return "now"
    elif minutes < 60:
        return f"{minutes}m ago"
    else:
        hours = minutes // 60
        return f"{hours}h ago"


def generate_bar(minutes: int, stale_threshold: int, colorblind: bool) -> tuple[str, str]:
    """Generate a Unicode bar with color based on time."""
    # Calculate bar width (max 10 blocks)
    if minutes < 5:
        width = max(1, minutes)
    elif minutes < 15:
        width = 3 + (minutes - 5) // 3
    elif minutes < stale_threshold:
        width = 6 + (minutes - 15) // 5
    else:
        width = 10

    # Clamp to max
    width = min(width, 10)

    # Generate bar string
    full_blocks = width
    bar = BAR_CHARS[8] * full_blocks

    # Determine color
    if colorblind:
        if minutes < 5:
            color = Colors.BLUE
        elif minutes < 15:
            color = Colors.CYAN
        elif minutes < stale_threshold:
            color = Colors.WHITE
        else:
            color = Colors.MAGENTA
    else:
        if minutes < 5:
            color = Colors.GREEN
        elif minutes < 15:
            color = Colors.YELLOW
        elif minutes < stale_threshold:
            color = Colors.ORANGE
        else:
            color = Colors.RED

    return bar, color


def render_tree(
    trees: dict[str, TreeNode],
    session: Session,
    stale_threshold: int,
    colorblind: bool,
    show_hierarchy: bool = True,
) -> str:
    """Render the tree view."""
    lines = []
    lines.append(f"{Colors.BOLD}CONDUCTOR WORKTREE TRACKER{Colors.RESET}")
    lines.append("━" * 60)

    if not trees:
        lines.append(f"{Colors.DIM}No worktrees visited yet. Click on workspaces in Conductor.{Colors.RESET}")
    else:
        for repo_name, root in sorted(trees.items()):
            lines.append(f"\n{Colors.BOLD}{repo_name}{Colors.RESET}")

            def render_node(node: TreeNode, prefix: str = "", is_last: bool = True):
                connector = "└── " if is_last else "├── "
                pr_str = f" (#{node.pr_number})" if node.pr_number else ""

                if node.click_record:
                    # Has a worktree that was clicked - show timer
                    click = node.click_record
                    minutes = int((utc_now() - click.last_seen).total_seconds() / 60)
                    time_str = format_time_ago(click.last_seen)
                    datetime_str = format_datetime(click.last_seen)
                    bar, color = generate_bar(minutes, stale_threshold, colorblind)

                    stale_str = " STALE" if minutes >= stale_threshold else ""
                    if colorblind and minutes >= stale_threshold:
                        stale_str = " \u26a0"  # Warning icon

                    line = f"{prefix}{connector}{node.name}{pr_str}"
                    line = f"{line:<35} {time_str:>8}  {color}{bar}{stale_str}{Colors.RESET}  {Colors.DIM}@ {datetime_str}{Colors.RESET}"
                    lines.append(line)
                elif node.branch:  # Has a branch but no worktree click
                    # Show branch without timer (dimmed)
                    line = f"{prefix}{connector}{Colors.DIM}{node.name}{pr_str}{Colors.RESET}"
                    lines.append(line)

                # Render children
                child_prefix = prefix + ("    " if is_last else "│   ")
                for i, child in enumerate(node.children):
                    render_node(child, child_prefix, i == len(node.children) - 1)

            # Render root's children directly
            for i, child in enumerate(root.children):
                render_node(child, "", i == len(root.children) - 1)

    lines.append("\n" + "━" * 60)
    session_duration = utc_now() - session.started_at
    session_mins = int(session_duration.total_seconds() / 60)
    if session_mins >= 60:
        session_str = f"{session_mins // 60}h {session_mins % 60}m"
    else:
        session_str = f"{session_mins}m"

    lines.append(f"Tracking: {len(session.clicks)} worktrees | Session: {session_str}")

    return "\n".join(lines)


def format_datetime(dt: datetime) -> str:
    """Format datetime as HH:MM (today) or Mon HH:MM (other days)."""
    now = utc_now()
    if dt.date() == now.date():
        return dt.strftime("%H:%M")
    else:
        return dt.strftime("%a %H:%M")


def get_visible_length(text: str) -> int:
    """Get visible length of text, excluding ANSI codes."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', text))


def render_radial(
    trees: dict[str, TreeNode],
    session: Session,
    stale_threshold: int,
    colorblind: bool,
) -> str:
    """Render repos in columns with proper hierarchy preserved."""
    lines = []

    # Get terminal size for responsive layout
    try:
        import shutil
        term_cols, term_rows = shutil.get_terminal_size((80, 24))
    except:
        term_cols, term_rows = 80, 24

    # Calculate column width and repos per row based on terminal width
    if term_cols >= 140:
        repos_per_row = 3
        col_width = 44
        name_width = 18
    elif term_cols >= 100:
        repos_per_row = 2
        col_width = 48
        name_width = 20
    elif term_cols >= 80:
        repos_per_row = 2
        col_width = 38
        name_width = 14
    else:
        repos_per_row = 1
        col_width = term_cols - 2
        name_width = min(18, term_cols - 20)

    header_width = min(term_cols - 2, repos_per_row * col_width)
    lines.append(f"{Colors.BOLD}CONDUCTOR WORKTREE TRACKER{Colors.RESET}")
    lines.append("━" * header_width)

    if not trees:
        lines.append(f"{Colors.DIM}No worktrees visited yet. Click on workspaces in Conductor.{Colors.RESET}")
        return "\n".join(lines)

    repo_list = list(trees.items())
    n_repos = len(repo_list)

    def render_node_compact(node: TreeNode, prefix: str, is_last: bool, col_lines: list, depth: int = 0):
        """Render a node with hierarchy preserved, single-line per node."""
        if not node.branch:  # Skip root node, render children directly
            for i, child in enumerate(node.children):
                render_node_compact(child, "", i == len(node.children) - 1, col_lines, 0)
            return

        connector = "└" if is_last else "├"
        pr_str = f" #{node.pr_number}" if node.pr_number else ""
        click = node.click_record
        max_name = name_width - depth * 2
        # Smart truncation: preserve closing paren/bracket if present
        if len(node.name) > max_name:
            if node.name.endswith(")"):
                node_name = node.name[:max_name-1] + ")"
            elif node.name.endswith("]"):
                node_name = node.name[:max_name-1] + "]"
            else:
                node_name = node.name[:max_name]
        else:
            node_name = node.name

        if click:
            minutes = int((utc_now() - click.last_seen).total_seconds() / 60)
            time_str = format_time_ago(click.last_seen)
            bar, color = generate_bar(minutes, stale_threshold, colorblind)
            stale_str = ""
            if minutes >= stale_threshold:
                stale_str = " ⚠" if colorblind else "!"

            # Single line: connector name PR bar time
            col_lines.append(f"{prefix}{connector} {node_name}{pr_str} {color}{bar}{Colors.RESET} {time_str}{stale_str}")
        elif node.branch:
            # Branch without worktree - show dimmed
            col_lines.append(f"{prefix}{Colors.DIM}{connector} {node_name}{pr_str}{Colors.RESET}")

        # Render children with proper indentation
        child_prefix = prefix + ("  " if is_last else "│ ")
        for i, child in enumerate(node.children):
            render_node_compact(child, child_prefix, i == len(node.children) - 1, col_lines, depth + 1)

    row_data = []
    for i in range(0, n_repos, repos_per_row):
        row_data.append(repo_list[i:i + repos_per_row])

    for row_repos in row_data:
        # Build each column for this row
        columns = []
        max_lines = 0

        for repo_name, root in row_repos:
            col_lines = []
            repo_display = repo_name[:name_width + 6] if len(repo_name) > name_width + 6 else repo_name
            col_lines.append(f"● {Colors.BOLD}{repo_display}{Colors.RESET}")

            # Render the tree with hierarchy
            render_node_compact(root, "", True, col_lines)

            columns.append(col_lines)
            max_lines = max(max_lines, len(col_lines))

        # Pad columns to same height
        for col in columns:
            while len(col) < max_lines:
                col.append("")

        # Render row with columns side by side
        for line_idx in range(max_lines):
            row_line = ""
            for col in columns:
                cell = col[line_idx] if line_idx < len(col) else ""
                visible_len = get_visible_length(cell)
                padding = col_width - visible_len
                row_line += cell + " " * max(0, padding)
            lines.append(row_line.rstrip())

        lines.append("")  # Blank line between rows

    lines.append("━" * header_width)
    session_duration = utc_now() - session.started_at
    session_mins = int(session_duration.total_seconds() / 60)
    if session_mins >= 60:
        session_str = f"{session_mins // 60}h {session_mins % 60}m"
    else:
        session_str = f"{session_mins}m"

    lines.append(f"Tracking: {len(session.clicks)} worktrees | Session: {session_str}")

    return "\n".join(lines)


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


# ============================================================================
# Main Loop
# ============================================================================

def main():
    # Load configuration from config.json
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Track Conductor worktree activity with visual time indicators"
    )
    parser.add_argument(
        "--stale", type=int, default=config["stale_minutes"],
        help=f"Minutes before a worktree is marked as stale (default: {config['stale_minutes']})"
    )
    parser.add_argument(
        "--colorblind", action="store_true", default=config["colorblind"],
        help="Use protanopia-friendly color scheme"
    )
    parser.add_argument(
        "--interval", type=float, default=config["interval_seconds"],
        help=f"Database polling interval in seconds (default: {config['interval_seconds']})"
    )
    parser.add_argument(
        "--no-hierarchy", action="store_true", default=not config["show_hierarchy"],
        help="Show flat list instead of PR hierarchy"
    )
    parser.add_argument(
        "--tree", action="store_true", default=config["tree_layout"],
        help="Use tree layout instead of default radial"
    )
    parser.add_argument(
        "--no-prs", action="store_true", default=not config["fetch_prs"],
        help="Skip GitHub PR queries (faster startup)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit (for testing)"
    )
    parser.add_argument(
        "--since", type=str, default=config["since"],
        help="Activity lookback: 'today' or hours (default: today)"
    )
    parser.add_argument(
        "--debug", action="store_true", default=config["debug"],
        help="Show debug output for click detection"
    )

    args = parser.parse_args()

    # Store pr_refresh_minutes from config
    pr_refresh_seconds = config["pr_refresh_minutes"] * 60

    session = Session()
    prs_by_repo: dict[str, list[PRInfo]] = {}
    last_pr_refresh = datetime.min

    print(f"{Colors.BOLD}Conductor Worktree Tracker{Colors.RESET}")
    print(f"Database: {CONDUCTOR_DB}")
    print()

    # Progress indicator helper
    def progress(msg: str, done: bool = False):
        icon = "✓" if done else "⋯"
        color = Colors.GREEN if done else Colors.YELLOW
        print(f"  {color}{icon}{Colors.RESET} {msg}", flush=True)

    # Step 1: Load workspaces from database
    if HAS_TQDM:
        with tqdm(total=1, desc="  Loading workspaces", ncols=60,
                  bar_format="{desc}: {bar} {n_fmt}/{total_fmt}", leave=True) as pbar:
            initial_workspaces = get_workspaces()
            pbar.update(1)
    else:
        progress("Loading workspaces from Conductor database...")
        initial_workspaces = get_workspaces()
    for ws in initial_workspaces:
        session.last_updated_at[ws.id] = ws.updated_at
    print(f"  {Colors.GREEN}✓{Colors.RESET} Loaded {len(initial_workspaces)} workspaces")

    # Step 2: Backfill recent activity
    cutoff = get_since_cutoff(args.since)
    since_desc = "today" if args.since == "today" else f"last {args.since}h"
    recent_workspaces = [ws for ws in initial_workspaces if ws.updated_at >= cutoff]

    if recent_workspaces:
        if HAS_TQDM:
            iterator = tqdm(
                recent_workspaces,
                desc="  Scanning activity",
                ncols=60,
                bar_format="{desc}: {bar} {n_fmt}/{total_fmt}",
                leave=True,
            )
        else:
            progress(f"Scanning activity from {since_desc}...")
            iterator = recent_workspaces

        for ws in iterator:
            session.clicks[ws.id] = ClickRecord(
                workspace_id=ws.id,
                name=ws.name,
                repo_name=ws.repo_name,
                branch=ws.branch,
                first_seen=ws.updated_at,
                last_seen=ws.updated_at,
            )
        print(f"  {Colors.GREEN}✓{Colors.RESET} Found {len(session.clicks)} active worktrees ({since_desc})")
    else:
        progress(f"No activity {since_desc}", done=True)

    # Step 3: Load PRs for active repos
    if not args.no_prs:
        active_repos = set(ws.repo_name for ws in recent_workspaces)
        if active_repos:
            prs_by_repo = get_all_prs(initial_workspaces, show_progress=True)
            last_pr_refresh = utc_now()
            total_prs = sum(len(prs) for prs in prs_by_repo.values())
            print(f"  {Colors.GREEN}✓{Colors.RESET} Loaded {total_prs} open PRs from {len(prs_by_repo)} repos")
        else:
            progress("No active repos, skipping PR fetch", done=True)
    else:
        progress("PR fetching disabled (--no-prs)", done=True)

    # Re-sync timestamps right before entering main loop
    # (PR fetching may have taken time during which DB was updated)
    fresh_workspaces = get_workspaces()
    for ws in fresh_workspaces:
        session.last_updated_at[ws.id] = ws.updated_at
        # Also update last_seen for tracked workspaces to reflect actual DB time
        if ws.id in session.clicks:
            session.clicks[ws.id].last_seen = ws.updated_at

    print()
    print(f"Press Ctrl+C to exit. Refreshing every {args.interval}s...\n")

    try:
        while True:
            # Load workspaces
            workspaces = get_workspaces()

            # Refresh PRs periodically (configured in config.json)
            if not args.no_prs and (utc_now() - last_pr_refresh).total_seconds() > pr_refresh_seconds:
                prs_by_repo = get_all_prs(workspaces)
                last_pr_refresh = utc_now()

            # Detect clicks (workspace updated_at changed since last check)
            # Use a 1-second tolerance to avoid false positives from precision issues
            for ws in workspaces:
                last_known = session.last_updated_at.get(ws.id)
                if last_known is not None:
                    delta_seconds = (ws.updated_at - last_known).total_seconds()
                    if delta_seconds > 1.0:  # Only trigger if timestamp changed by >1 second
                        if args.debug:
                            print(f"[DEBUG] Click: {ws.name} delta={delta_seconds:.1f}s")
                        # This workspace was clicked/updated since last poll
                        if ws.id in session.clicks:
                            session.clicks[ws.id].last_seen = utc_now()
                        else:
                            session.clicks[ws.id] = ClickRecord(
                                workspace_id=ws.id,
                                name=ws.name,
                                repo_name=ws.repo_name,
                                branch=ws.branch,
                                first_seen=utc_now(),
                                last_seen=utc_now(),
                            )
                session.last_updated_at[ws.id] = ws.updated_at

            # Build hierarchy
            trees = build_hierarchy(workspaces, session, prs_by_repo, debug=args.debug)

            # Render
            if not args.once:
                clear_screen()
            if args.tree:
                output = render_tree(
                    trees, session, args.stale, args.colorblind,
                    show_hierarchy=not args.no_hierarchy
                )
            else:
                # Radial is default
                output = render_radial(trees, session, args.stale, args.colorblind)
            print(output)

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
