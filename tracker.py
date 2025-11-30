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
import select
import shutil
import sqlite3
import subprocess
import sys
import termios
import time
import tty
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
    "interval_seconds": 2.0,
    "show_hierarchy": True,
    "tree_layout": False,
    "fetch_prs": True,
    "since": "today",  # "today" or number of hours
    "debug": False,
    "pr_refresh_minutes": 5,
    "show_status": True,  # Show session status (working/idle) indicators
    "pomodoro_enabled": False,
    "pomodoro_work_minutes": 25,
    "pomodoro_break_minutes": 5,
    "pomodoro_reflections_file": "reflections.md",
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

# Spinner frames for animated indicators
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_spinner_idx = 0

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
    root_path: str = ""  # Repo root path for detecting actual git branch


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
    hidden_workspaces: set[str] = field(default_factory=set)  # Set of workspace IDs to hide
    hidden_repos: set[str] = field(default_factory=set)  # Set of repo names to hide


@dataclass
class PRInfo:
    number: int
    head_branch: str
    base_branch: str
    title: str = ""
    ci_status: str = ""  # "pass", "fail", "pending", "skipping", "" (no checks)


@dataclass
class SessionStatus:
    workspace_id: str
    status: str  # 'idle', 'working', 'error'
    model: Optional[str] = None
    is_compacting: bool = False


@dataclass
class PomodoroState:
    """Track pomodoro timer state."""
    enabled: bool = False
    work_minutes: int = 25
    break_minutes: int = 5
    phase: str = "work"  # "work" or "break"
    phase_started_at: datetime = field(default_factory=_utc_now_factory)
    session_count: int = 0  # Number of completed work sessions
    reflections_file: str = "reflections.md"

    def minutes_remaining(self) -> int:
        """Get minutes remaining in current phase."""
        elapsed = (utc_now() - self.phase_started_at).total_seconds() / 60
        phase_duration = self.work_minutes if self.phase == "work" else self.break_minutes
        return max(0, int(phase_duration - elapsed))

    def is_phase_complete(self) -> bool:
        """Check if current phase is complete."""
        return self.minutes_remaining() <= 0

    def start_break(self):
        """Transition to break phase."""
        self.phase = "break"
        self.phase_started_at = utc_now()

    def start_work(self):
        """Transition to work phase."""
        self.phase = "work"
        self.phase_started_at = utc_now()
        self.session_count += 1

    def format_timer(self) -> str:
        """Format remaining time as MM:SS."""
        remaining = self.minutes_remaining()
        mins = remaining
        secs = int((self.minutes_remaining() * 60) % 60)
        return f"{mins:02d}:{secs:02d}"


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
            r.default_branch,
            r.root_path
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
            root_path=row["root_path"] or "",
        ))

    conn.close()
    return workspaces


def get_actual_git_branch(ws: Workspace) -> str:
    """Get the actual git branch for a workspace by running git command.

    Returns the stored branch if git command fails.
    """
    if not ws.root_path:
        return ws.branch

    worktree_path = Path(ws.root_path) / ".conductor" / ws.name
    if not worktree_path.exists():
        return ws.branch

    try:
        result = subprocess.run(
            ["git", "-C", str(worktree_path), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return ws.branch


def detect_actual_branches(workspaces: list[Workspace], show_progress: bool = False) -> None:
    """Update workspace branches with actual git branches.

    This detects cases where the user switched branches after creating the worktree.
    Modifies workspaces in-place.
    """
    if show_progress and HAS_TQDM:
        iterator = tqdm(
            workspaces,
            desc="  Detecting branches",
            ncols=60,
            bar_format="{desc}: {bar} {n_fmt}/{total_fmt}",
            leave=True,
        )
    else:
        iterator = workspaces

    for ws in iterator:
        actual_branch = get_actual_git_branch(ws)
        if actual_branch != ws.branch:
            ws.branch = actual_branch


def get_session_statuses() -> dict[str, SessionStatus]:
    """Get current session status for all workspaces with active sessions."""
    if not CONDUCTOR_DB.exists():
        return {}

    conn = sqlite3.connect(f"file:{CONDUCTOR_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get the most recent session for each workspace
    cursor.execute("""
        SELECT
            s.workspace_id,
            s.status,
            s.model,
            s.is_compacting
        FROM sessions s
        WHERE s.workspace_id IS NOT NULL
          AND s.is_hidden = 0
        ORDER BY s.updated_at DESC
    """)

    statuses: dict[str, SessionStatus] = {}
    for row in cursor.fetchall():
        ws_id = row["workspace_id"]
        # Only keep the most recent session per workspace
        if ws_id not in statuses:
            statuses[ws_id] = SessionStatus(
                workspace_id=ws_id,
                status=row["status"] or "idle",
                model=row["model"],
                is_compacting=bool(row["is_compacting"]),
            )

    conn.close()
    return statuses


def get_last_user_message_times() -> dict[str, datetime]:
    """Get last_user_message_at for all workspaces with sessions.

    This is more accurate than updated_at for determining when the user
    was actually active in a workspace (vs background session updates).
    """
    if not CONDUCTOR_DB.exists():
        return {}

    conn = sqlite3.connect(f"file:{CONDUCTOR_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.workspace_id, MAX(s.last_user_message_at) as last_user_message_at
        FROM sessions s
        WHERE s.workspace_id IS NOT NULL
          AND s.is_hidden = 0
          AND s.last_user_message_at IS NOT NULL
        GROUP BY s.workspace_id
    """)

    result: dict[str, datetime] = {}
    for row in cursor.fetchall():
        ws_id = row["workspace_id"]
        timestamp = row["last_user_message_at"]
        if timestamp:
            # Parse UTC timestamp from database
            result[ws_id] = datetime.fromisoformat(timestamp.replace("Z", ""))

    conn.close()
    return result


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
             "--json", "number,headRefName,baseRefName,title"],
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
                title=pr.get("title", ""),
            )
            for pr in prs
        ]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []


def get_pr_ci_status(owner: str, repo: str, pr_number: int) -> str:
    """Get CI status for a PR using gh pr checks.

    Returns: "pass", "fail", "pending", "skipping", or "" (no checks/error)
    """
    try:
        result = subprocess.run(
            ["gh", "pr", "checks", str(pr_number), "--repo", f"{owner}/{repo}",
             "--json", "bucket"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            # Exit code 1 often means no checks
            return ""

        checks = json.loads(result.stdout)
        if not checks:
            return ""

        # Aggregate status: fail > pending > pass
        has_fail = any(c.get("bucket") == "fail" for c in checks)
        has_pending = any(c.get("bucket") == "pending" for c in checks)
        has_pass = any(c.get("bucket") == "pass" for c in checks)

        if has_fail:
            return "fail"
        elif has_pending:
            return "pending"
        elif has_pass:
            return "pass"
        else:
            return ""
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return ""


# Cache for CI status: {(owner, repo, pr_number): (status, timestamp)}
_ci_cache: dict[tuple[str, str, int], tuple[str, float]] = {}

# Rate-limiting settings to avoid hitting GitHub API limits
CI_CACHE_PENDING_SECONDS = 120   # Pending PRs - check every 2 min (active CI)
CI_CACHE_STABLE_SECONDS = 300    # Pass/fail - stable states, check every 5 min
CI_CACHE_UNKNOWN_SECONDS = 120   # Unknown/empty - check every 2 min
CI_MAX_REQUESTS_PER_CYCLE = 5    # Max API calls per refresh cycle


def get_ci_cache_duration(status: str) -> int:
    """Get cache duration based on CI status."""
    if status == "pending":
        return CI_CACHE_PENDING_SECONDS
    elif status in ("pass", "fail"):
        return CI_CACHE_STABLE_SECONDS
    else:
        return CI_CACHE_UNKNOWN_SECONDS


def get_pr_ci_status_cached(owner: str, repo: str, pr_number: int) -> str:
    """Get CI status from cache only (no API call)."""
    key = (owner, repo, pr_number)
    if key in _ci_cache:
        status, _ = _ci_cache[key]
        return status
    return ""


def is_ci_cache_stale(key: tuple[str, str, int]) -> bool:
    """Check if a cache entry needs refreshing."""
    if key not in _ci_cache:
        return True
    status, timestamp = _ci_cache[key]
    age = time.time() - timestamp
    return age > get_ci_cache_duration(status)


def get_ci_icon(status: str) -> str:
    """Get visual icon for CI status."""
    global _spinner_idx
    if status == "pass":
        return f"{Colors.GREEN}✓{Colors.RESET}"
    elif status == "fail":
        return f"{Colors.RED}✗{Colors.RESET}"
    elif status == "pending":
        # Animated spinner for active CI
        spinner = SPINNER_FRAMES[_spinner_idx % len(SPINNER_FRAMES)]
        return f"{Colors.YELLOW}{spinner}{Colors.RESET}"
    else:
        return ""


def refresh_ci_statuses(workspaces: list, prs_by_repo: dict) -> None:
    """Refresh CI status for PRs with rate limiting.

    - Only refreshes stale cache entries
    - Prioritizes pending PRs (active CI)
    - Limited to CI_MAX_REQUESTS_PER_CYCLE API calls per cycle
    """
    # Build owner/repo lookup from workspaces
    repo_to_owner: dict[str, tuple[str, str]] = {}
    for ws in workspaces:
        parsed = parse_github_remote(ws.remote_url)
        if parsed and ws.repo_name not in repo_to_owner:
            repo_to_owner[ws.repo_name] = parsed  # (owner, repo)

    # Collect all PRs that need refreshing, with their keys
    pending_prs = []  # (owner, repo, pr) - prioritize these
    other_prs = []    # (owner, repo, pr)

    for repo_name, prs in prs_by_repo.items():
        if repo_name not in repo_to_owner:
            continue
        owner, repo = repo_to_owner[repo_name]
        for pr in prs:
            key = (owner, repo, pr.number)
            # Always update from cache first (no API call)
            pr.ci_status = get_pr_ci_status_cached(owner, repo, pr.number)

            # Check if this entry needs refreshing
            if is_ci_cache_stale(key):
                if pr.ci_status == "pending":
                    pending_prs.append((owner, repo, pr))
                else:
                    other_prs.append((owner, repo, pr))

    # Refresh up to CI_MAX_REQUESTS_PER_CYCLE PRs, prioritizing pending
    to_refresh = pending_prs[:CI_MAX_REQUESTS_PER_CYCLE]
    remaining_slots = CI_MAX_REQUESTS_PER_CYCLE - len(to_refresh)
    if remaining_slots > 0:
        to_refresh.extend(other_prs[:remaining_slots])

    for owner, repo, pr in to_refresh:
        status = get_pr_ci_status(owner, repo, pr.number)
        _ci_cache[(owner, repo, pr.number)] = (status, time.time())
        pr.ci_status = status


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
    name: str  # Display name (PR title if has PR, else worktree name)
    branch: str
    workspace_id: Optional[str] = None
    pr_number: Optional[int] = None
    pr_title: Optional[str] = None  # Original PR title for reference
    click_record: Optional[ClickRecord] = None
    children: list["TreeNode"] = field(default_factory=list)
    is_default_branch: bool = False
    session_status: Optional[SessionStatus] = None
    ci_status: str = ""  # "pass", "fail", "pending", "" (no checks)


def build_hierarchy(
    workspaces: list[Workspace],
    session: Session,
    prs_by_repo: dict[str, list[PRInfo]],
    session_statuses: dict[str, SessionStatus] = None,
    include_pr_branches: bool = True,
    debug: bool = False,
) -> dict[str, TreeNode]:
    """Build hierarchy tree for each repo with active workspaces.

    If include_pr_branches is True, also include branches from PRs that don't
    have worktrees (shown without timers).

    Hidden workspaces (in session.hidden_workspaces) and hidden repos
    (in session.hidden_repos) are filtered out.
    """
    # Group workspaces by repo - include all workspaces for repos with clicks
    # but filter out hidden workspaces and hidden repos
    repos_with_clicks = set()
    for ws in workspaces:
        if ws.id in session.clicks and ws.id not in session.hidden_workspaces:
            # Also check if the repo itself is hidden
            if ws.repo_name not in session.hidden_repos:
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

        # Build PR lookup: head_branch -> (base_branch, pr_number, title, ci_status)
        pr_lookup: dict[str, tuple[str, int, str, str]] = {}
        for pr in prs:
            pr_lookup[pr.head_branch] = (pr.base_branch, pr.number, pr.title, pr.ci_status)

        if debug and prs:
            print(f"[DEBUG] {repo_name}: {len(prs)} PRs")
            for pr in prs[:5]:  # Show first 5
                print(f"  PR #{pr.number}: {pr.head_branch} → {pr.base_branch} ({pr.title[:30]}...)")

        # Create root node for repo
        root = TreeNode(name=repo_name, branch="", is_default_branch=False)

        # Create nodes for each workspace that was clicked
        nodes: dict[str, TreeNode] = {}
        workspace_by_branch: dict[str, Workspace] = {}

        for ws in repo_workspaces:
            workspace_by_branch[ws.branch] = ws
            # Skip hidden workspaces
            if ws.id in session.clicks and ws.id not in session.hidden_workspaces:
                click = session.clicks.get(ws.id)
                pr_info = pr_lookup.get(ws.branch)
                status = session_statuses.get(ws.id) if session_statuses else None
                # Use PR title as display name if available, else worktree name
                if pr_info and pr_info[2]:
                    display_name = pr_info[2]  # PR title
                else:
                    display_name = ws.name  # Worktree name
                node = TreeNode(
                    name=display_name,
                    branch=ws.branch,
                    workspace_id=ws.id,
                    pr_number=pr_info[1] if pr_info else None,
                    pr_title=pr_info[2] if pr_info else None,
                    click_record=click,
                    session_status=status,
                    ci_status=pr_info[3] if pr_info else "",
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
                # Use PR title as display name if available, else branch name
                if pr_info and pr_info[2]:
                    display_name = pr_info[2]  # PR title
                else:
                    display_name = branch[:18] if len(branch) > 18 else branch
                node = TreeNode(
                    name=f"({display_name})",  # Parens indicate no worktree
                    branch=branch,
                    workspace_id=None,
                    pr_number=pr_info[1] if pr_info else None,
                    pr_title=pr_info[2] if pr_info else None,
                    click_record=None,  # No timer for branches without worktrees
                    ci_status=pr_info[3] if pr_info else "",
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


def get_status_icon(status: Optional[SessionStatus]) -> str:
    """Get visual icon for session status.

    Returns:
        str: Status icon with color codes
        - Working: animated spinner (green)
        - Idle: ○ (dim)
        - Error: ✗ (red)
        - Compacting: ⟳ (yellow)
        - None: empty string
    """
    global _spinner_idx

    if status is None:
        return ""

    if status.is_compacting:
        return f"{Colors.YELLOW}⟳{Colors.RESET}"

    if status.status == "working":
        spinner = SPINNER_FRAMES[_spinner_idx % len(SPINNER_FRAMES)]
        return f"{Colors.GREEN}{spinner}{Colors.RESET}"
    elif status.status == "error":
        return f"{Colors.RED}✗{Colors.RESET}"
    else:  # idle
        return f"{Colors.DIM}○{Colors.RESET}"


def advance_spinner():
    """Advance spinner to next frame."""
    global _spinner_idx
    _spinner_idx = (_spinner_idx + 1) % len(SPINNER_FRAMES)


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


def generate_bar(minutes: int, stale_threshold: int) -> tuple[str, str]:
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

    # Determine color (green -> yellow -> orange -> red)
    if minutes < 5:
        color = Colors.GREEN
    elif minutes < 15:
        color = Colors.YELLOW
    elif minutes < stale_threshold:
        color = Colors.ORANGE
    else:
        color = Colors.RED

    return bar, color


def format_tracking_line(session: Session, session_mins: int) -> str:
    """Format the tracking status line with visible/hidden counts.

    Args:
        session: Current session with clicks and hidden sets.
        session_mins: Session duration in minutes.

    Returns:
        Formatted status line string.
    """
    if session_mins >= 60:
        session_str = f"{session_mins // 60}h {session_mins % 60}m"
    else:
        session_str = f"{session_mins}m"

    # Count visible workspaces - exclude both hidden workspaces AND
    # workspaces belonging to hidden repos
    visible_count = sum(
        1 for ws_id, click in session.clicks.items()
        if ws_id not in session.hidden_workspaces
        and click.repo_name not in session.hidden_repos
    )

    hidden_ws_count = len(session.hidden_workspaces)
    hidden_repo_count = len(session.hidden_repos)

    hidden_parts = []
    if hidden_ws_count > 0:
        hidden_parts.append(f"{hidden_ws_count} ws")
    if hidden_repo_count > 0:
        hidden_parts.append(f"{hidden_repo_count} repos")
    hidden_str = f" ({', '.join(hidden_parts)} hidden)" if hidden_parts else ""

    return f"Tracking: {visible_count} worktrees{hidden_str} | Session: {session_str} | {Colors.DIM}Press ` to hide/show{Colors.RESET}"


def render_tree(
    trees: dict[str, TreeNode],
    session: Session,
    stale_threshold: int,
    show_hierarchy: bool = True,
    show_status: bool = True,
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
                    bar, color = generate_bar(minutes, stale_threshold)

                    stale_str = " !" if minutes >= stale_threshold else ""

                    # Status icon (working/idle/error) - only if show_status enabled
                    status_icon = get_status_icon(node.session_status) if show_status else ""
                    status_prefix = f"{status_icon} " if status_icon else ""

                    line = f"{prefix}{connector}{status_prefix}{node.name}{pr_str}"
                    line = f"{line:<37} {time_str:>8}  {color}{bar}{stale_str}{Colors.RESET}  {Colors.DIM}@ {datetime_str}{Colors.RESET}"
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
    lines.append(format_tracking_line(session, session_mins))

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
    session_statuses: dict[str, SessionStatus] = None,
    show_status: bool = True,
    pomodoro: Optional[PomodoroState] = None,
) -> str:
    """Render repos in a circular layout with worktrees radiating outward."""
    import math

    try:
        term_cols, term_rows = shutil.get_terminal_size((80, 24))
    except:
        term_cols, term_rows = 80, 24

    # Reserve space for header/footer
    header_lines = 2
    footer_lines = 7
    available_rows = max(20, term_rows - header_lines - footer_lines)
    available_cols = term_cols - 2

    # Create 2D buffer for drawing (list of lists of chars/strings)
    # Each cell stores (char, color_code or None)
    buffer = [[(' ', None) for _ in range(available_cols)] for _ in range(available_rows)]

    def draw_text_raw(x: int, y: int, text: str):
        """Draw raw text with embedded ANSI codes."""
        if y < 0 or y >= available_rows:
            return
        if 0 <= x < available_cols:
            buffer[y][x] = (text, 'RAW')

    if not trees:
        lines = []
        lines.append(f"{Colors.BOLD}CONDUCTOR WORKTREE TRACKER{Colors.RESET}")
        lines.append("━" * available_cols)
        lines.append(f"{Colors.DIM}No worktrees visited yet. Click on workspaces in Conductor.{Colors.RESET}")
        return "\n".join(lines)

    repo_list = list(trees.items())
    n_repos = len(repo_list)

    # Center of the circle
    cx = available_cols // 2
    cy = available_rows // 2

    # Maximum width for each repo's content - scale with terminal width
    # Keep narrow enough that left/right sides don't overlap (< half of available_cols)
    max_content_width = max(30, min(50, available_cols // 2 - 5))
    name_width = min(20, max_content_width // 2)

    def render_node_lines(node: TreeNode, prefix: str, is_last: bool, lines_out: list, depth: int = 0, direction: str = "down"):
        """Render node to list of lines.

        direction: "down" (normal), "up" (inverted for top of circle),
                   "left" (mirrored connectors for left side)
        """
        if not node.branch:
            # Root node - just render children
            children = node.children
            if direction == "up":
                # For upward trees, reverse children order
                children = list(reversed(children))
            for i, child in enumerate(children):
                is_last_child = (i == len(children) - 1)
                render_node_lines(child, "", is_last_child, lines_out, 0, direction)
            return

        ci_icon = get_ci_icon(node.ci_status) if node.ci_status else ""
        pr_str = f" #{node.pr_number}" if node.pr_number else ""
        if ci_icon:
            pr_str = f"{pr_str} {ci_icon}" if pr_str else ci_icon
        click = node.click_record
        max_name = name_width - depth * 2

        if len(node.name) > max_name:
            if node.name.endswith(")"):
                node_name = node.name[:max_name-1] + ")"
            else:
                node_name = node.name[:max_name]
        else:
            node_name = node.name

        # Choose connector based on direction
        if direction == "up":
            connector = "┌" if is_last else "├"
        elif direction == "left":
            # Mirrored connectors for left-side expansion
            connector = "┘" if is_last else "┤"
        else:
            connector = "└" if is_last else "├"

        # For upward trees, render children FIRST (they appear above parent)
        if direction == "up":
            children = list(reversed(node.children))
            child_prefix = prefix + ("  " if is_last else "│ ")
            for i, child in enumerate(children):
                is_last_child = (i == 0)  # First in reversed = last originally
                render_node_lines(child, child_prefix, is_last_child, lines_out, depth + 1, direction)

        # Render this node
        if click:
            minutes = int((utc_now() - click.last_seen).total_seconds() / 60)
            time_str = format_time_ago(click.last_seen)
            bar, color = generate_bar(minutes, stale_threshold)
            stale_str = " !" if minutes >= stale_threshold else ""

            status_icon = get_status_icon(node.session_status) if show_status else ""
            status_pre = f"{status_icon} " if status_icon else ""

            if direction == "left":
                # Right-align for left expansion: time bar name connector prefix
                line = f"{time_str} {color}{bar}{Colors.RESET}{stale_str} {status_pre}{node_name}{pr_str} {connector}{prefix}"
            else:
                line = f"{prefix}{connector} {status_pre}{node_name}{pr_str} {color}{bar}{Colors.RESET} {time_str}{stale_str}"
            lines_out.append(line)
        elif node.branch:
            if direction == "left":
                line = f"{Colors.DIM}{node_name}{pr_str} {connector}{prefix}{Colors.RESET}"
            else:
                line = f"{prefix}{Colors.DIM}{connector} {node_name}{pr_str}{Colors.RESET}"
            lines_out.append(line)

        # For downward/left trees, render children AFTER (they appear below parent)
        if direction != "up":
            if direction == "left":
                child_prefix = ("  " if is_last else " │") + prefix
            else:
                child_prefix = prefix + ("  " if is_last else "│ ")
            for i, child in enumerate(node.children):
                render_node_lines(child, child_prefix, i == len(node.children) - 1, lines_out, depth + 1, direction)

    # ==========================================================================
    # Expanded Circle Layout - repos around ellipse, all using simple rendering
    # ==========================================================================
    # Large ellipse: repos positioned around the perimeter
    # All trees use simple "down" direction for clarity

    # EXPANDED circle radius - scale with terminal size, no hard caps
    rx = available_cols // 3  # Horizontal radius scales with width
    ry = available_rows // 3  # Vertical radius scales with height

    # Calculate max lines per repo - scale with available space
    # With more vertical space, show more branches per repo
    # Use a generous formula: at least 6 lines, up to 1/3 of screen height
    max_lines_per_repo = max(6, available_rows // 3)

    # Track placed repos for collision avoidance: list of (x_start, x_end, y_start, y_end)
    placed_regions = []

    def find_free_position(target_x, target_y, n_lines, width):
        """Find a free y position, shifting down if needed to avoid collisions."""
        y = target_y
        for _ in range(available_rows):
            collision = False
            for px_start, px_end, py_start, py_end in placed_regions:
                # Check x overlap
                my_x_end = target_x + width
                x_overlaps = not (my_x_end < px_start or target_x > px_end)
                if not x_overlaps:
                    continue
                # Check y overlap
                my_y_end = y + n_lines - 1
                y_overlaps = not (my_y_end < py_start or y > py_end)
                if y_overlaps:
                    # Shift below this region
                    y = py_end + 1
                    collision = True
                    break
            if not collision:
                break
        return max(0, min(available_rows - n_lines, y))

    # For each repo, calculate position and render
    for i, (repo_name, root) in enumerate(repo_list):
        # Distribute repos evenly around the circle
        # Start from top (-π/2) and go clockwise
        theta = (2 * math.pi * i / n_repos) - (math.pi / 2)

        # Calculate position on ellipse
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        circle_x = int(cx + rx * cos_t)
        circle_y = int(cy + ry * sin_t)

        # Determine direction based on quadrant - trees branch OUTWARD from circle
        if abs(sin_t) > abs(cos_t):
            # Primarily top or bottom
            if sin_t < 0:
                direction = "up"  # Top - branch upward
            else:
                direction = "down"  # Bottom - branch downward
        else:
            # Primarily left or right
            if cos_t < 0:
                direction = "left"  # Left - branch leftward
            else:
                direction = "down"  # Right - branch rightward (down works for right)

        # Pre-render this repo with directional branching
        repo_lines = []
        repo_display = repo_name[:20] if len(repo_name) > 20 else repo_name

        if direction == "up":
            # For upward trees, repo name comes LAST (at bottom, closest to center)
            render_node_lines(root, "", True, repo_lines, direction=direction)
            repo_lines.append(f"● {Colors.BOLD}{repo_display}{Colors.RESET}")
        elif direction == "left":
            # For left trees, repo name on RIGHT (closest to center), tree extends left
            repo_lines.append(f"{Colors.BOLD}{repo_display}{Colors.RESET} ●")
            render_node_lines(root, "", True, repo_lines, direction=direction)
        else:
            # For down/right trees, repo name comes FIRST (closest to center)
            repo_lines.append(f"● {Colors.BOLD}{repo_display}{Colors.RESET}")
            render_node_lines(root, "", True, repo_lines, direction=direction)

        # Truncate if needed
        if len(repo_lines) > max_lines_per_repo:
            if direction == "up":
                # Keep bottom lines (repo name and closest children)
                truncated = [f"  {Colors.DIM}... +{len(repo_lines) - max_lines_per_repo + 1} more{Colors.RESET}"]
                truncated.extend(repo_lines[-(max_lines_per_repo - 1):])
                repo_lines = truncated
            else:
                truncated = repo_lines[:max_lines_per_repo - 1]
                truncated.append(f"  {Colors.DIM}... +{len(repo_lines) - max_lines_per_repo + 1} more{Colors.RESET}")
                repo_lines = truncated

        n_lines = len(repo_lines)

        # Position content based on quadrant - radiating OUTWARD from circle
        if abs(sin_t) > abs(cos_t):
            # Top or bottom quadrant - center horizontally
            content_x = circle_x - max_content_width // 2
            if sin_t < 0:
                # Top - content extends upward from circle point
                content_y = max(0, circle_y - n_lines + 1)
            else:
                # Bottom - content extends downward from circle point
                content_y = circle_y
        else:
            # Left or right quadrant
            if cos_t < 0:
                # Left - RIGHT edge near circle, content extends leftward
                content_x = max(1, circle_x - max_content_width)
            else:
                # Right - LEFT edge near circle, content extends rightward
                content_x = min(circle_x + 2, available_cols - max_content_width - 1)
            # Y position follows the circle
            content_y = circle_y - n_lines // 2

        # Clamp to screen bounds
        content_x = max(0, min(available_cols - max_content_width, content_x))
        content_y = max(0, min(available_rows - n_lines, content_y))

        # Find collision-free position
        content_y = find_free_position(content_x, content_y, n_lines, max_content_width)

        # Mark this region as occupied
        placed_regions.append((content_x, content_x + max_content_width, content_y, content_y + n_lines - 1))

        # Draw each line of the repo content
        for j, line in enumerate(repo_lines):
            line_y = content_y + j
            if 0 <= line_y < available_rows:
                if direction == "left":
                    # Right-align left-direction content
                    visible_len = get_visible_length(line)
                    x_offset = max_content_width - visible_len
                    draw_text_raw(content_x + x_offset, line_y, line)
                else:
                    draw_text_raw(content_x, line_y, line)

    # Convert buffer to output lines
    output_lines = []
    output_lines.append(f"{Colors.BOLD}CONDUCTOR WORKTREE TRACKER{Colors.RESET}")
    output_lines.append("━" * available_cols)

    for row in buffer:
        line_parts = []
        i = 0
        while i < len(row):
            ch, color = row[i]
            if color == 'RAW':
                # This is a raw string with embedded ANSI
                line_parts.append(ch)
                # Skip the visible length of this string
                visible_len = get_visible_length(ch)
                i += max(1, visible_len)
            elif color:
                line_parts.append(f"{color}{ch}{Colors.RESET}")
                i += 1
            else:
                line_parts.append(ch)
                i += 1
        output_lines.append(''.join(line_parts).rstrip())

    # Remove trailing empty lines but keep some structure
    while len(output_lines) > 3 and not output_lines[-1].strip():
        output_lines.pop()

    output_lines.append("")  # Blank before footer

    # Build status box for top-right overlay
    status_box_lines = []
    if show_status and session_statuses:
        working_count = 0
        idle_count = 0
        error_count = 0

        for ws_id, status in session_statuses.items():
            if ws_id in session.clicks:
                if status.status == "working":
                    working_count += 1
                elif status.status == "idle":
                    idle_count += 1
                elif status.status == "error":
                    error_count += 1

        if working_count > 0 or idle_count > 0 or error_count > 0:
            status_width = 22
            spinner_icon = get_status_icon(SessionStatus("", "working"))

            status_box_lines.append(f"┌─ STATUS {'─' * (status_width - 11)}┐")

            if working_count > 0:
                w_count_str = str(working_count)
                # Prominent: green, bold, filled blocks around it
                w_text = f"{Colors.GREEN}{Colors.BOLD}█ {spinner_icon} {w_count_str} WORKING █{Colors.RESET}"
                w_visible = 4 + len(w_count_str) + 10  # "█ ⠋ N WORKING █"
                w_pad = status_width - 2 - w_visible
                status_box_lines.append(f"│{w_text}{' ' * max(0, w_pad)}│")

            if idle_count > 0:
                i_count_str = str(idle_count)
                # Subdued: dim gray
                i_text = f"{Colors.DIM}○ {i_count_str} idle{Colors.RESET}"
                i_visible = 2 + len(i_count_str) + 5  # "○ N idle"
                i_pad = status_width - 4 - i_visible
                status_box_lines.append(f"│  {i_text}{' ' * max(0, i_pad)} │")

            if error_count > 0:
                e_count_str = str(error_count)
                e_text = f"{Colors.RED}✗ {e_count_str} error{Colors.RESET}"
                e_visible = 2 + len(e_count_str) + 6
                e_pad = status_width - 4 - e_visible
                status_box_lines.append(f"│  {e_text}{' ' * max(0, e_pad)} │")

            status_box_lines.append(f"└{'─' * (status_width - 2)}┘")

    # Overlay status box in top-right corner
    if status_box_lines:
        status_width = 22
        for i, status_line in enumerate(status_box_lines):
            target_idx = 2 + i  # Start after header lines
            if target_idx < len(output_lines):
                existing = output_lines[target_idx]
                existing_visible = get_visible_length(existing)
                target_x = available_cols - status_width - 1

                if existing_visible < target_x:
                    # Pad existing line and append status box
                    output_lines[target_idx] = existing + ' ' * (target_x - existing_visible) + status_line
                else:
                    # Truncate existing and add status box with ellipsis
                    # Find where to cut (accounting for ANSI codes)
                    cut_point = 0
                    visible_count = 0
                    in_ansi = False
                    for j, ch in enumerate(existing):
                        if ch == '\x1b':
                            in_ansi = True
                        elif in_ansi and ch == 'm':
                            in_ansi = False
                        elif not in_ansi:
                            visible_count += 1
                            if visible_count >= target_x - 4:
                                cut_point = j
                                break
                        cut_point = j + 1
                    output_lines[target_idx] = existing[:cut_point] + f"{Colors.RESET}... " + status_line
            else:
                # Add new line for status box
                output_lines.append(' ' * (available_cols - status_width - 1) + status_line)

    lines = output_lines

    # Pomodoro timer display
    if pomodoro and pomodoro.enabled:
        remaining = pomodoro.minutes_remaining()
        phase_icon = "🍅" if pomodoro.phase == "work" else "☕"
        phase_label = "WORK" if pomodoro.phase == "work" else "BREAK"

        # Color based on urgency
        if pomodoro.phase == "work":
            if remaining <= 2:
                timer_color = Colors.RED
            elif remaining <= 5:
                timer_color = Colors.ORANGE
            else:
                timer_color = Colors.GREEN
        else:
            timer_color = Colors.CYAN

        # Build progress bar for pomodoro
        phase_duration = pomodoro.work_minutes if pomodoro.phase == "work" else pomodoro.break_minutes
        elapsed = phase_duration - remaining
        progress_pct = min(1.0, elapsed / phase_duration) if phase_duration > 0 else 0
        bar_width = 20
        filled = int(progress_pct * bar_width)
        pomo_bar = "█" * filled + "░" * (bar_width - filled)

        lines.append("")
        lines.append(f"┌─ {phase_icon} Pomodoro {'─' * (available_cols - 15)}┐")
        lines.append(f"│ {timer_color}{phase_label}: {remaining}m remaining{Colors.RESET}  [{pomo_bar}]")
        lines.append(f"│ {Colors.DIM}Session #{pomodoro.session_count}{Colors.RESET}")
        lines.append(f"└{'─' * (available_cols - 2)}┘")

    lines.append("━" * available_cols)
    session_duration = utc_now() - session.started_at
    session_mins = int(session_duration.total_seconds() / 60)
    lines.append(format_tracking_line(session, session_mins))

    return "\n".join(lines)


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


# ============================================================================
# Pomodoro Functions
# ============================================================================

REFLECTIONS_FILE = Path(__file__).parent / "reflections.md"


def prompt_reflection() -> str:
    """Prompt user for a quick reflection before break.

    Returns:
        str: The user's reflection text
    """
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}🍅 POMODORO BREAK TIME!{Colors.RESET}")
    print("=" * 60)
    print()
    print("Before you take a break, write a quick reflection:")
    print(f"{Colors.DIM}(What did you work on? Any insights or blockers?){Colors.RESET}")
    print()

    try:
        reflection = input(f"{Colors.CYAN}> {Colors.RESET}")
        return reflection.strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def save_reflection(reflection: str, pomodoro: PomodoroState):
    """Save reflection to markdown file.

    Args:
        reflection: The reflection text to save
        pomodoro: Current pomodoro state for context
    """
    if not reflection:
        return

    reflections_path = Path(__file__).parent / pomodoro.reflections_file

    # Create file with header if it doesn't exist
    if not reflections_path.exists():
        with open(reflections_path, "w") as f:
            f.write("# Pomodoro Reflections\n\n")
            f.write("Quick notes from each work session.\n\n")
            f.write("---\n\n")

    # Append reflection with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M")

    with open(reflections_path, "a") as f:
        f.write(f"## {timestamp} (Session #{pomodoro.session_count})\n\n")
        f.write(f"{reflection}\n\n")
        f.write("---\n\n")


def show_break_screen(pomodoro: PomodoroState):
    """Display break countdown screen.

    Args:
        pomodoro: Current pomodoro state
    """
    remaining = pomodoro.minutes_remaining()
    print("\n" + "=" * 60)
    print(f"{Colors.GREEN}{Colors.BOLD}☕ BREAK TIME - {remaining}m remaining{Colors.RESET}")
    print("=" * 60)
    print()
    print("Take a break! Stretch, hydrate, look away from the screen.")
    print()
    print(f"{Colors.DIM}Press Enter when ready to resume work...{Colors.RESET}")


# ============================================================================
# Non-blocking Input Handling
# ============================================================================

class TerminalInput:
    """Handle non-blocking terminal input for interactive mode."""

    def __init__(self):
        self.old_settings = None
        self.enabled = False

    def enable(self):
        """Enable raw terminal mode for non-blocking input."""
        if sys.stdin.isatty():
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                self.enabled = True
            except termios.error:
                self.enabled = False

    def disable(self):
        """Restore terminal to normal mode."""
        if self.old_settings and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except termios.error:
                pass
        self.enabled = False

    def get_key(self) -> Optional[str]:
        """Get a key if one is available (non-blocking).

        Returns:
            Single character if a regular key is pressed, None otherwise.
            Escape sequences (arrow keys, etc.) are consumed and return None.
        """
        if not self.enabled or not sys.stdin.isatty():
            return None

        try:
            # Check if input is available
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                ch = sys.stdin.read(1)
                # Handle escape sequences (arrow keys send \x1b[A, \x1b[B, etc.)
                if ch == '\x1b':
                    # Check if more characters follow (escape sequence)
                    # Use a small timeout to distinguish bare Escape from sequences
                    more_readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if more_readable:
                        # This is an escape sequence - consume and ignore it
                        # Read up to 8 more chars to clear the sequence
                        for _ in range(8):
                            ready, _, _ = select.select([sys.stdin], [], [], 0)
                            if ready:
                                sys.stdin.read(1)
                            else:
                                break
                        return None  # Ignore escape sequences
                    # Bare escape key - return it
                    return '\x1b'
                return ch
        except (OSError, select.error):
            pass
        return None


@dataclass
class HideMenuState:
    """State for the workspace hide/show menu."""
    active: bool = False


# Keys a-z for menu selection (26 items max)
MENU_KEYS = "abcdefghijklmnopqrstuvwxyz"


def build_hide_menu(
    session: Session,
    prs_by_repo: Optional[dict[str, list[PRInfo]]] = None,
) -> tuple[list[tuple[str, str, str, str, bool]], list[tuple[str, str, bool]]]:
    """Build list of workspaces and repos for hide menu.

    Args:
        session: Current session with clicks.
        prs_by_repo: PR info by repo for display names.

    Returns:
        Tuple of (workspace_items, repo_items) where:
        - workspace_items: List of (key, workspace_id, display_name, repo_name, is_hidden)
        - repo_items: List of (key, repo_name, is_hidden)
    """
    # Build PR lookup for display names: (repo_name, branch) -> PR title
    pr_titles: dict[tuple[str, str], str] = {}
    if prs_by_repo:
        for repo_name, prs in prs_by_repo.items():
            for pr in prs:
                pr_titles[(repo_name, pr.head_branch)] = pr.title

    # Build workspace items sorted by name
    workspace_items = []
    sorted_clicks = sorted(session.clicks.items(), key=lambda x: x[1].name)

    for i, (ws_id, click) in enumerate(sorted_clicks):
        if i >= len(MENU_KEYS):
            break  # Max 26 items

        key = MENU_KEYS[i]
        is_hidden = ws_id in session.hidden_workspaces

        # Use PR title if available, otherwise workspace name
        pr_title = pr_titles.get((click.repo_name, click.branch))
        display_name = pr_title if pr_title else click.name
        # Truncate long names
        if len(display_name) > 40:
            display_name = display_name[:37] + "..."

        workspace_items.append((key, ws_id, display_name, click.repo_name, is_hidden))

    # Build unique repo list
    repos = sorted(set(click.repo_name for click in session.clicks.values()))
    repo_items = []
    # Use uppercase letters for repos (after workspaces use lowercase)
    repo_keys = MENU_KEYS.upper()

    for i, repo_name in enumerate(repos):
        if i >= len(repo_keys):
            break
        key = repo_keys[i]
        is_hidden = repo_name in session.hidden_repos
        repo_items.append((key, repo_name, is_hidden))

    return workspace_items, repo_items


def render_hide_menu(
    session: Session,
    available_cols: int,
    prs_by_repo: Optional[dict[str, list[PRInfo]]] = None,
) -> str:
    """Render the hide/show workspace menu overlay.

    Args:
        session: Current session with clicks and hidden_workspaces.
        available_cols: Terminal width for formatting.
        prs_by_repo: PR info for display names.

    Returns:
        Formatted menu string.
    """
    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD}━━━ HIDE/SHOW WORKSPACES & REPOS ━━━{Colors.RESET}")
    lines.append(f"{Colors.DIM}a-z: toggle workspace, A-Z: toggle repo, `: close, 0: show all{Colors.RESET}")
    lines.append("")

    workspace_items, repo_items = build_hide_menu(session, prs_by_repo)

    # Render repos section first
    if repo_items:
        lines.append(f"{Colors.BOLD}Repositories:{Colors.RESET}")
        for key, repo_name, is_hidden in repo_items:
            if is_hidden:
                status = f"{Colors.DIM}[hidden]{Colors.RESET}"
                name_display = f"{Colors.DIM}{repo_name}{Colors.RESET}"
            else:
                status = f"{Colors.GREEN}[visible]{Colors.RESET}"
                name_display = repo_name

            key_hint = f"{Colors.MAGENTA}{key}{Colors.RESET}"
            lines.append(f"  {key_hint}  {name_display:<35} {status}")
        lines.append("")

    # Render workspaces section
    if workspace_items:
        lines.append(f"{Colors.BOLD}Workspaces:{Colors.RESET}")
        for key, ws_id, display_name, repo_name, is_hidden in workspace_items:
            if is_hidden:
                status = f"{Colors.DIM}[hidden]{Colors.RESET}"
                name_display = f"{Colors.DIM}{display_name}{Colors.RESET}"
            else:
                status = f"{Colors.GREEN}[visible]{Colors.RESET}"
                name_display = display_name

            key_hint = f"{Colors.CYAN}{key}{Colors.RESET}"
            # Show repo name in dim if different workspaces from same repo
            repo_hint = f" {Colors.DIM}({repo_name[:15]}){Colors.RESET}" if len(repo_items) > 1 else ""
            lines.append(f"  {key_hint}  {name_display:<35}{repo_hint} {status}")
    else:
        lines.append(f"  {Colors.DIM}No workspaces to hide/show{Colors.RESET}")

    lines.append("")
    hidden_ws_count = len(session.hidden_workspaces)
    hidden_repo_count = len(session.hidden_repos)
    if hidden_ws_count > 0 or hidden_repo_count > 0:
        parts = []
        if hidden_ws_count > 0:
            parts.append(f"{hidden_ws_count} workspace(s)")
        if hidden_repo_count > 0:
            parts.append(f"{hidden_repo_count} repo(s)")
        lines.append(f"  {Colors.YELLOW}{', '.join(parts)} hidden{Colors.RESET}")
    lines.append("━" * min(60, available_cols))

    return "\n".join(lines)


def handle_hide_menu_input(
    key: str,
    session: Session,
    menu_state: HideMenuState,
    prs_by_repo: Optional[dict[str, list[PRInfo]]] = None,
) -> bool:
    """Handle keyboard input for hide menu.

    Args:
        key: The key that was pressed.
        session: Current session to modify.
        menu_state: Current menu state.
        prs_by_repo: PR info for building menu.

    Returns:
        True if menu should stay open, False to close it.
    """
    if key == '`' or key == '\x1b':  # backtick or Escape
        return False

    if key == '0':
        # Show all - clear both hidden sets
        session.hidden_workspaces.clear()
        session.hidden_repos.clear()
        return True

    workspace_items, repo_items = build_hide_menu(session, prs_by_repo)

    # Handle lowercase a-z for workspaces
    if key.islower() and key in MENU_KEYS:
        for item_key, ws_id, display_name, repo_name, is_hidden in workspace_items:
            if item_key == key:
                if is_hidden:
                    session.hidden_workspaces.discard(ws_id)
                else:
                    session.hidden_workspaces.add(ws_id)
                break
        return True

    # Handle uppercase A-Z for repos
    if key.isupper() and key.lower() in MENU_KEYS:
        for item_key, repo_name, is_hidden in repo_items:
            if item_key == key:
                if is_hidden:
                    session.hidden_repos.discard(repo_name)
                else:
                    session.hidden_repos.add(repo_name)
                break
        return True

    return True  # Keep menu open for unrecognized keys


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
    parser.add_argument(
        "--no-status", action="store_true", default=not config["show_status"],
        help="Hide session status indicators (working/idle)"
    )
    parser.add_argument(
        "--pomodoro", action="store_true", default=config["pomodoro_enabled"],
        help="Enable pomodoro timer mode with reflection prompts"
    )
    parser.add_argument(
        "--diag", action="store_true",
        help="Show diagnostic info (terminal size, timing)"
    )

    args = parser.parse_args()

    # Store pr_refresh_minutes from config
    pr_refresh_seconds = config["pr_refresh_minutes"] * 60

    session = Session()
    prs_by_repo: dict[str, list[PRInfo]] = {}
    last_pr_refresh = datetime.min

    # Initialize pomodoro timer if enabled
    pomodoro = PomodoroState(
        enabled=args.pomodoro,
        work_minutes=config["pomodoro_work_minutes"],
        break_minutes=config["pomodoro_break_minutes"],
        reflections_file=config["pomodoro_reflections_file"],
    )
    if pomodoro.enabled:
        pomodoro.session_count = 1  # Start at session 1

    # Track last_user_message_at for more accurate activity timing
    last_user_message_times: dict[str, datetime] = {}

    print(f"{Colors.BOLD}Conductor Worktree Tracker{Colors.RESET}")
    print(f"Database: {CONDUCTOR_DB}")
    if pomodoro.enabled:
        print(f"Pomodoro: {pomodoro.work_minutes}m work / {pomodoro.break_minutes}m break")
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

    # Detect actual git branches (Conductor DB may be stale if user switched branches)
    detect_actual_branches(initial_workspaces, show_progress=HAS_TQDM)

    # Step 2: Backfill recent activity using last_user_message_at for accuracy
    cutoff = get_since_cutoff(args.since)
    since_desc = "today" if args.since == "today" else f"last {args.since}h"
    recent_workspaces = [ws for ws in initial_workspaces if ws.updated_at >= cutoff]

    # Get last_user_message_at times for more accurate activity tracking
    last_user_message_times = get_last_user_message_times()

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
            # Use last_user_message_at if available, otherwise fall back to updated_at
            last_activity = last_user_message_times.get(ws.id, ws.updated_at)
            session.clicks[ws.id] = ClickRecord(
                workspace_id=ws.id,
                name=ws.name,
                repo_name=ws.repo_name,
                branch=ws.branch,
                first_seen=ws.updated_at,
                last_seen=last_activity,
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

    # Re-sync the baseline timestamps right before entering main loop
    # (PR fetching may have taken time during which DB was updated)
    # NOTE: We only sync last_updated_at (for click detection baseline),
    # NOT last_seen (which should only update on actual user clicks)
    fresh_workspaces = get_workspaces()
    for ws in fresh_workspaces:
        session.last_updated_at[ws.id] = ws.updated_at

    print()
    print(f"Press Ctrl+C to exit. Refreshing every {args.interval}s...")
    print(f"{Colors.DIM}Press ` (backtick) to hide/show workspaces & repos{Colors.RESET}\n")

    # Initialize terminal input handler and hide menu state
    term_input = TerminalInput()
    hide_menu = HideMenuState()

    try:
        # Enable raw input mode for non-blocking keyboard handling
        term_input.enable()

        while True:
            # Handle pomodoro phase transitions
            if pomodoro.enabled and pomodoro.is_phase_complete():
                if pomodoro.phase == "work":
                    # Work phase complete - prompt for reflection
                    clear_screen()
                    reflection = prompt_reflection()
                    save_reflection(reflection, pomodoro)
                    pomodoro.start_break()
                    show_break_screen(pomodoro)
                    try:
                        input()  # Wait for user to press Enter
                    except (EOFError, KeyboardInterrupt):
                        pass
                else:
                    # Break phase complete - start new work session
                    pomodoro.start_work()

            # Load workspaces and session statuses
            workspaces = get_workspaces()
            detect_actual_branches(workspaces)  # Get real git branches for PR matching
            session_statuses = get_session_statuses()

            # Refresh last_user_message_at for accurate activity timing
            current_msg_times = get_last_user_message_times()

            # Refresh PRs periodically (configured in config.json)
            if not args.no_prs and (utc_now() - last_pr_refresh).total_seconds() > pr_refresh_seconds:
                prs_by_repo = get_all_prs(workspaces)
                last_pr_refresh = utc_now()

            # Update activity times based on last_user_message_at changes
            for ws_id, msg_time in current_msg_times.items():
                old_msg_time = last_user_message_times.get(ws_id)
                if old_msg_time is None or msg_time > old_msg_time:
                    # User sent a message - update last_seen
                    if ws_id in session.clicks:
                        if args.debug:
                            print(f"[DEBUG] Message: {session.clicks[ws_id].name} at {msg_time}")
                        session.clicks[ws_id].last_seen = msg_time
            last_user_message_times.update(current_msg_times)

            # Detect new workspaces (updated_at changed since last check)
            for ws in workspaces:
                last_known = session.last_updated_at.get(ws.id)
                if last_known is not None:
                    delta_seconds = (ws.updated_at - last_known).total_seconds()
                    if delta_seconds > 1.0:  # Only trigger if timestamp changed by >1 second
                        existing = session.clicks.get(ws.id)
                        if not existing:
                            # New workspace - add it
                            last_activity = current_msg_times.get(ws.id, ws.updated_at)
                            if args.debug:
                                print(f"[DEBUG] New: {ws.name} delta={delta_seconds:.1f}s")
                            session.clicks[ws.id] = ClickRecord(
                                workspace_id=ws.id,
                                name=ws.name,
                                repo_name=ws.repo_name,
                                branch=ws.branch,
                                first_seen=ws.updated_at,
                                last_seen=last_activity,
                            )
                session.last_updated_at[ws.id] = ws.updated_at

            # Build hierarchy with session statuses
            trees = build_hierarchy(
                workspaces, session, prs_by_repo,
                session_statuses=session_statuses,
                debug=args.debug
            )

            # Render
            if not args.once:
                clear_screen()
            if args.tree:
                output = render_tree(
                    trees, session, args.stale,
                    show_hierarchy=not args.no_hierarchy,
                    show_status=not args.no_status
                )
            else:
                # Radial is default
                output = render_radial(
                    trees, session, args.stale,
                    session_statuses=session_statuses,
                    show_status=not args.no_status,
                    pomodoro=pomodoro if pomodoro.enabled else None,
                )
            print(output)

            # Show hide menu if active
            if hide_menu.active:
                term_cols, _ = shutil.get_terminal_size((80, 24))
                menu_output = render_hide_menu(session, term_cols, prs_by_repo)
                print(menu_output)

            # Show diagnostic info if requested
            if args.diag:
                term_cols, term_rows = shutil.get_terminal_size((80, 24))
                print(f"[DIAG] Terminal: {term_cols}x{term_rows}, repos: {len(trees)}, PRs cached: {len(_ci_cache)}")

            # Advance spinner for next frame
            advance_spinner()

            if args.once:
                break

            # Refresh CI status AFTER render (so display appears immediately)
            # CI icons will update on next render cycle
            if not args.no_prs and prs_by_repo:
                refresh_ci_statuses(workspaces, prs_by_repo)

            # Handle keyboard input - check multiple times during the interval
            # to be responsive to user input
            interval_remaining = args.interval
            check_interval = 0.1  # Check for input every 100ms

            while interval_remaining > 0:
                key = term_input.get_key()
                if key:
                    if hide_menu.active:
                        # Handle input for hide menu
                        hide_menu.active = handle_hide_menu_input(key, session, hide_menu, prs_by_repo)
                        # Immediately redraw after menu interaction
                        break
                    else:
                        # Check for backtick to open hide menu
                        if key == '`':
                            hide_menu.active = True
                            # Immediately redraw to show menu
                            break
                        elif key == 'q' or key == 'Q':
                            # Allow 'q' to quit
                            raise KeyboardInterrupt

                time.sleep(min(check_interval, interval_remaining))
                interval_remaining -= check_interval

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        # Always restore terminal settings
        term_input.disable()
    sys.exit(0)


if __name__ == "__main__":
    main()
