#!/usr/bin/env python3
"""
Conductor Worktree Tracker - HTTP API Server

Provides a REST API for the 3D Three.js visualization frontend.
Serves the tracker data including workspaces, PRs, session statuses, and hierarchy.
"""

import argparse
import http.server
import json
import os
import re
import socketserver
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse


# ============================================================================
# Configuration
# ============================================================================

CONDUCTOR_DB = Path.home() / "Library/Application Support/com.conductor.app/conductor.db"
CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "stale_minutes": 30,
    "fetch_prs": True,
    "since": "today",
    "pr_refresh_minutes": 15,
    "ci_cache_pending_seconds": 300,    # Pending PRs - check every 5 min
    "ci_cache_stable_seconds": 900,     # Pass/fail - check every 15 min
    "ci_cache_unknown_seconds": 600,    # Unknown - check every 10 min
    "ci_max_requests_per_cycle": 2,     # Max API calls per refresh cycle
}


def load_config() -> dict:
    """Load configuration from config.json."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                user_config = json.load(f)
                config.update(user_config)
        except (json.JSONDecodeError, IOError):
            pass
    return config


def utc_now() -> datetime:
    """Get current time in UTC (naive datetime for comparison with DB)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_since_cutoff(since_value) -> datetime:
    """Convert 'since' config value to a datetime cutoff."""
    now = utc_now()
    if since_value == "today":
        local_now = datetime.now()
        local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        utc_offset = local_now.astimezone().utcoffset()
        if utc_offset:
            return local_midnight - utc_offset
        return local_midnight
    else:
        return now - timedelta(hours=float(since_value))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Workspace:
    id: str
    name: str
    branch: str
    parent_branch: Optional[str]
    repo_id: str
    repo_name: str
    remote_url: str
    default_branch: str
    updated_at: datetime
    root_path: str = ""


@dataclass
class ClickRecord:
    workspace_id: str
    name: str
    repo_name: str
    branch: str
    first_seen: datetime
    last_seen: datetime


@dataclass
class PRInfo:
    number: int
    head_branch: str
    base_branch: str
    title: str = ""
    ci_status: str = ""


@dataclass
class SessionStatus:
    workspace_id: str
    status: str
    model: Optional[str] = None
    is_compacting: bool = False


@dataclass
class CommitInfo:
    sha: str
    short_sha: str
    message: str
    author: str
    date: str
    github_url: str = ""


@dataclass
class TreeNode:
    name: str
    branch: str
    workspace_id: Optional[str] = None
    pr_number: Optional[int] = None
    pr_title: Optional[str] = None
    click_record: Optional[dict] = None
    session_status: Optional[dict] = None
    ci_status: str = ""
    children: list = field(default_factory=list)
    commits_from_parent: list = field(default_factory=list)  # Commits between this and parent branch
    parent_branch_name: Optional[str] = None  # The branch this branches from


# ============================================================================
# Database Access
# ============================================================================

def get_workspaces() -> list[Workspace]:
    """Load all workspaces from Conductor database."""
    if not CONDUCTOR_DB.exists():
        return []

    conn = sqlite3.connect(f"file:{CONDUCTOR_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            w.id,
            w.directory_name,
            w.branch,
            w.initialization_parent_branch as parent_branch,
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
    """Get the actual git branch for a workspace."""
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


def detect_actual_branches(workspaces: list[Workspace]) -> None:
    """Update workspace branches with actual git branches."""
    for ws in workspaces:
        actual_branch = get_actual_git_branch(ws)
        if actual_branch != ws.branch:
            ws.branch = actual_branch


def get_commits_between_branches(
    repo_path: str,
    base_branch: str,
    head_branch: str,
    remote_url: str,
    max_commits: int = 20
) -> list[dict]:
    """Get commits between two branches.

    Returns commits that are in head_branch but not in base_branch.
    Uses git log base_branch..head_branch
    """
    if not repo_path or not Path(repo_path).exists():
        return []

    # Parse GitHub URL for commit links
    github_info = parse_github_remote(remote_url)
    github_base_url = ""
    if github_info:
        owner, repo = github_info
        github_base_url = f"https://github.com/{owner}/{repo}/commit"

    try:
        # Use git log with a format that's easy to parse
        # Format: sha|short_sha|author|date|message
        result = subprocess.run(
            [
                "git", "-C", repo_path, "log",
                f"origin/{base_branch}..origin/{head_branch}",
                f"--max-count={max_commits}",
                "--format=%H|%h|%an|%ad|%s",
                "--date=short"
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            # Try without origin/ prefix (local branches)
            result = subprocess.run(
                [
                    "git", "-C", repo_path, "log",
                    f"{base_branch}..{head_branch}",
                    f"--max-count={max_commits}",
                    "--format=%H|%h|%an|%ad|%s",
                    "--date=short"
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

        if result.returncode != 0:
            return []

        commits = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split('|', 4)
            if len(parts) >= 5:
                sha, short_sha, author, date, message = parts
                github_url = f"{github_base_url}/{sha}" if github_base_url else ""
                commits.append({
                    "sha": sha,
                    "short_sha": short_sha,
                    "author": author,
                    "date": date,
                    "message": message[:80],  # Truncate long messages
                    "github_url": github_url,
                })

        return commits

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def get_commits_from_default_branch(
    repo_path: str,
    branch: str,
    default_branch: str,
    remote_url: str,
    max_commits: int = 20
) -> list[dict]:
    """Get commits between default branch and a feature branch."""
    return get_commits_between_branches(
        repo_path, default_branch, branch, remote_url, max_commits
    )


def get_session_statuses() -> dict[str, SessionStatus]:
    """Get current session status for all workspaces."""
    if not CONDUCTOR_DB.exists():
        return {}

    conn = sqlite3.connect(f"file:{CONDUCTOR_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

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
    """Get last_user_message_at for all workspaces."""
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
            result[ws_id] = datetime.fromisoformat(timestamp.replace("Z", ""))

    conn.close()
    return result


# ============================================================================
# GitHub PR Integration
# ============================================================================

def parse_github_remote(remote_url: str) -> Optional[tuple[str, str]]:
    """Extract owner/repo from GitHub remote URL."""
    ssh_match = re.match(r"git@github\.com:([^/]+)/([^.]+)(?:\.git)?", remote_url)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)

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


# CI status cache: (owner, repo, pr_number) -> (status, timestamp)
_ci_cache: dict[tuple[str, str, int], tuple[str, float]] = {}


def get_ci_cache_duration(status: str, config: dict) -> int:
    """Get cache duration based on CI status."""
    if status == "pending":
        return config.get("ci_cache_pending_seconds", 300)
    elif status in ("pass", "fail"):
        return config.get("ci_cache_stable_seconds", 900)
    else:
        return config.get("ci_cache_unknown_seconds", 600)


def get_pr_ci_status_cached(owner: str, repo: str, pr_number: int, config: dict) -> str:
    """Get CI status with caching to reduce API calls."""
    import time
    cache_key = (owner, repo, pr_number)
    now = time.time()

    # Check cache
    if cache_key in _ci_cache:
        cached_status, cached_time = _ci_cache[cache_key]
        cache_duration = get_ci_cache_duration(cached_status, config)
        if now - cached_time < cache_duration:
            return cached_status

    # Fetch fresh status
    status = get_pr_ci_status(owner, repo, pr_number)
    _ci_cache[cache_key] = (status, now)
    return status


def get_pr_ci_status(owner: str, repo: str, pr_number: int) -> str:
    """Get CI status for a PR."""
    try:
        result = subprocess.run(
            ["gh", "pr", "checks", str(pr_number), "--repo", f"{owner}/{repo}",
             "--json", "bucket"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return ""

        checks = json.loads(result.stdout)
        if not checks:
            return ""

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


def get_all_prs(workspaces: list[Workspace]) -> dict[str, list[PRInfo]]:
    """Get PRs for all unique repos."""
    repos_to_fetch: list[tuple[str, str, str]] = []
    repos_seen = set()

    for ws in workspaces:
        parsed = parse_github_remote(ws.remote_url)
        if parsed and ws.repo_name not in repos_seen:
            repos_seen.add(ws.repo_name)
            owner, repo = parsed
            repos_to_fetch.append((ws.repo_name, owner, repo))

    prs_by_repo: dict[str, list[PRInfo]] = {}
    for repo_name, owner, repo in repos_to_fetch:
        prs_by_repo[repo_name] = get_open_prs(owner, repo)

    return prs_by_repo


# ============================================================================
# Hierarchy Building
# ============================================================================

def build_hierarchy(
    workspaces: list[Workspace],
    clicks: dict[str, ClickRecord],
    prs_by_repo: dict[str, list[PRInfo]],
    session_statuses: dict[str, SessionStatus],
    fetch_commits: bool = True,
) -> dict[str, dict]:
    """Build hierarchy tree for each repo."""

    repos_with_clicks = set()
    for ws in workspaces:
        if ws.id in clicks:
            repos_with_clicks.add(ws.repo_name)

    by_repo: dict[str, list[Workspace]] = {}
    for ws in workspaces:
        if ws.repo_name in repos_with_clicks:
            by_repo.setdefault(ws.repo_name, []).append(ws)

    trees: dict[str, dict] = {}

    for repo_name, repo_workspaces in by_repo.items():
        if not repo_workspaces:
            continue

        default_branch = repo_workspaces[0].default_branch
        remote_url = repo_workspaces[0].remote_url
        root_path = repo_workspaces[0].root_path
        prs = prs_by_repo.get(repo_name, [])

        pr_lookup: dict[str, tuple[str, int, str, str]] = {}
        for pr in prs:
            pr_lookup[pr.head_branch] = (pr.base_branch, pr.number, pr.title, pr.ci_status)

        # Create root node with default branch info
        root = TreeNode(name=repo_name, branch=default_branch)
        nodes: dict[str, TreeNode] = {}
        workspace_by_branch: dict[str, Workspace] = {}

        for ws in repo_workspaces:
            workspace_by_branch[ws.branch] = ws
            if ws.id in clicks:
                click = clicks.get(ws.id)
                pr_info = pr_lookup.get(ws.branch)
                status = session_statuses.get(ws.id)

                if pr_info and pr_info[2]:
                    display_name = pr_info[2]
                else:
                    display_name = ws.name

                click_dict = None
                if click:
                    click_dict = {
                        "workspace_id": click.workspace_id,
                        "name": click.name,
                        "repo_name": click.repo_name,
                        "branch": click.branch,
                        "first_seen": click.first_seen.isoformat(),
                        "last_seen": click.last_seen.isoformat(),
                    }

                status_dict = None
                if status:
                    status_dict = {
                        "workspace_id": status.workspace_id,
                        "status": status.status,
                        "model": status.model,
                        "is_compacting": status.is_compacting,
                    }

                node = TreeNode(
                    name=display_name,
                    branch=ws.branch,
                    workspace_id=ws.id,
                    pr_number=pr_info[1] if pr_info else None,
                    pr_title=pr_info[2] if pr_info else None,
                    click_record=click_dict,
                    session_status=status_dict,
                    ci_status=pr_info[3] if pr_info else "",
                )
                nodes[ws.branch] = node

        # Include PR branches without worktrees
        branches_to_add = set()
        for branch in list(nodes.keys()):
            current = branch
            visited = set()  # Prevent infinite loops from cycles
            while current in pr_lookup and current not in visited:
                visited.add(current)
                parent_branch = pr_lookup[current][0]
                if parent_branch not in nodes and parent_branch != default_branch:
                    branches_to_add.add(parent_branch)
                current = parent_branch

        for branch in branches_to_add:
            pr_info = pr_lookup.get(branch)
            if pr_info and pr_info[2]:
                display_name = pr_info[2]
            else:
                display_name = branch[:18] if len(branch) > 18 else branch
            node = TreeNode(
                name=f"({display_name})",
                branch=branch,
                workspace_id=None,
                pr_number=pr_info[1] if pr_info else None,
                pr_title=pr_info[2] if pr_info else None,
                click_record=None,
                ci_status=pr_info[3] if pr_info else "",
            )
            nodes[branch] = node

        # Determine parents and fetch commits
        attached = set()
        for branch, node in nodes.items():
            pr_info = pr_lookup.get(branch)
            parent_branch_name = None

            if pr_info:
                parent_branch_name = pr_info[0]
                if parent_branch_name in nodes:
                    nodes[parent_branch_name].children.append(node)
                    attached.add(branch)
                    node.parent_branch_name = parent_branch_name
                elif parent_branch_name == default_branch:
                    # Parent is default branch (root)
                    node.parent_branch_name = default_branch

            if branch not in attached:
                ws = workspace_by_branch.get(branch)
                if ws and ws.parent_branch and ws.parent_branch in nodes:
                    nodes[ws.parent_branch].children.append(node)
                    attached.add(branch)
                    node.parent_branch_name = ws.parent_branch
                elif ws and ws.parent_branch:
                    node.parent_branch_name = ws.parent_branch

            # Fetch commits between this branch and its parent
            if fetch_commits and root_path and node.parent_branch_name:
                commits = get_commits_between_branches(
                    root_path,
                    node.parent_branch_name,
                    branch,
                    remote_url,
                    max_commits=15
                )
                node.commits_from_parent = commits
            elif fetch_commits and root_path and branch not in attached:
                # Branch attached to root - get commits from default branch
                commits = get_commits_from_default_branch(
                    root_path,
                    branch,
                    default_branch,
                    remote_url,
                    max_commits=15
                )
                node.commits_from_parent = commits
                node.parent_branch_name = default_branch

        for branch, node in nodes.items():
            if branch not in attached:
                root.children.append(node)

        if root.children:
            trees[repo_name] = tree_node_to_dict(root, remote_url)

    return trees


def tree_node_to_dict(node: TreeNode, remote_url: str = "") -> dict:
    """Convert TreeNode to serializable dict."""
    # Parse GitHub info for repo URL
    github_url = ""
    if remote_url:
        github_info = parse_github_remote(remote_url)
        if github_info:
            owner, repo = github_info
            github_url = f"https://github.com/{owner}/{repo}"

    return {
        "name": node.name,
        "branch": node.branch,
        "workspace_id": node.workspace_id,
        "pr_number": node.pr_number,
        "pr_title": node.pr_title,
        "click_record": node.click_record,
        "session_status": node.session_status,
        "ci_status": node.ci_status,
        "commits_from_parent": node.commits_from_parent,
        "parent_branch_name": node.parent_branch_name,
        "github_url": github_url,
        "children": [tree_node_to_dict(child, remote_url) for child in node.children],
    }


# ============================================================================
# API Server State
# ============================================================================

class TrackerState:
    """Maintains tracker state across API requests."""

    def __init__(self, config: dict):
        self.config = config
        self.session_start = utc_now()
        self.clicks: dict[str, ClickRecord] = {}
        self.last_updated_at: dict[str, datetime] = {}
        self.prs_by_repo: dict[str, list[PRInfo]] = {}
        self.last_pr_refresh = datetime.min
        self.last_user_message_times: dict[str, datetime] = {}
        self._initialized = False

    def initialize(self):
        """Initial data load."""
        if self._initialized:
            return

        workspaces = get_workspaces()

        for ws in workspaces:
            self.last_updated_at[ws.id] = ws.updated_at

        cutoff = get_since_cutoff(self.config["since"])
        recent = [ws for ws in workspaces if ws.updated_at >= cutoff]

        # Only detect branches for recent workspaces (much faster)
        detect_actual_branches(recent)

        msg_times = get_last_user_message_times()
        self.last_user_message_times = msg_times

        for ws in recent:
            last_activity = msg_times.get(ws.id, ws.updated_at)
            self.clicks[ws.id] = ClickRecord(
                workspace_id=ws.id,
                name=ws.name,
                repo_name=ws.repo_name,
                branch=ws.branch,
                first_seen=ws.updated_at,
                last_seen=last_activity,
            )

        if self.config.get("fetch_prs", True):
            # Only fetch PRs for repos with recent activity (in clicks)
            self.prs_by_repo = get_all_prs(recent)
            self.last_pr_refresh = utc_now()

        # Update CI statuses only for recent workspaces
        self._refresh_ci_statuses(recent)

        self._initialized = True

    def _refresh_ci_statuses(self, workspaces: list[Workspace]):
        """Refresh CI status for PRs with rate limiting."""
        repo_to_owner: dict[str, tuple[str, str]] = {}
        for ws in workspaces:
            parsed = parse_github_remote(ws.remote_url)
            if parsed and ws.repo_name not in repo_to_owner:
                repo_to_owner[ws.repo_name] = parsed

        max_requests = self.config.get("ci_max_requests_per_cycle", 2)
        requests_made = 0

        for repo_name, prs in self.prs_by_repo.items():
            if repo_name not in repo_to_owner:
                continue
            owner, repo = repo_to_owner[repo_name]
            for pr in prs:
                if requests_made >= max_requests:
                    break
                # Use cached version - only counts as request if cache miss
                old_status = pr.ci_status
                pr.ci_status = get_pr_ci_status_cached(owner, repo, pr.number, self.config)
                # Count only actual API calls (cache misses)
                if pr.ci_status != old_status or not old_status:
                    requests_made += 1
            if requests_made >= max_requests:
                break

    def force_refresh_ci(self):
        """Force refresh all CI statuses, bypassing rate limit and cache."""
        global _ci_cache
        _ci_cache.clear()  # Clear the cache to force fresh fetches

        workspaces = get_workspaces()
        recent = [ws for ws in workspaces if ws.id in self.clicks]

        # Also refresh the PR list to get latest PRs
        self.prs_by_repo = get_all_prs(recent)

        repo_to_owner: dict[str, tuple[str, str]] = {}
        for ws in recent:
            parsed = parse_github_remote(ws.remote_url)
            if parsed and ws.repo_name not in repo_to_owner:
                repo_to_owner[ws.repo_name] = parsed

        # Fetch CI status for all PRs without rate limiting
        for repo_name, prs in self.prs_by_repo.items():
            if repo_name not in repo_to_owner:
                continue
            owner, repo = repo_to_owner[repo_name]
            for pr in prs:
                pr.ci_status = get_pr_ci_status(owner, repo, pr.number)

        # Update last refresh time to prevent immediate re-fetch overwriting our results
        self.last_pr_refresh = utc_now()

    def refresh(self):
        """Refresh data from database."""
        workspaces = get_workspaces()
        # Only detect branches for workspaces with recent activity (in clicks)
        recent_ws = [ws for ws in workspaces if ws.id in self.clicks]
        detect_actual_branches(recent_ws)
        session_statuses = get_session_statuses()

        current_msg_times = get_last_user_message_times()

        # Update activity based on message times
        for ws_id, msg_time in current_msg_times.items():
            old_msg_time = self.last_user_message_times.get(ws_id)
            if old_msg_time is None or msg_time > old_msg_time:
                if ws_id in self.clicks:
                    self.clicks[ws_id].last_seen = msg_time
        self.last_user_message_times.update(current_msg_times)

        # Detect new workspaces
        for ws in workspaces:
            last_known = self.last_updated_at.get(ws.id)
            if last_known is not None:
                delta = (ws.updated_at - last_known).total_seconds()
                if delta > 1.0:
                    if ws.id not in self.clicks:
                        last_activity = current_msg_times.get(ws.id, ws.updated_at)
                        self.clicks[ws.id] = ClickRecord(
                            workspace_id=ws.id,
                            name=ws.name,
                            repo_name=ws.repo_name,
                            branch=ws.branch,
                            first_seen=ws.updated_at,
                            last_seen=last_activity,
                        )
            self.last_updated_at[ws.id] = ws.updated_at

        # Refresh PRs periodically - only for workspaces with recent activity
        pr_refresh_seconds = self.config.get("pr_refresh_minutes", 5) * 60
        if self.config.get("fetch_prs", True):
            if (utc_now() - self.last_pr_refresh).total_seconds() > pr_refresh_seconds:
                recent = [ws for ws in workspaces if ws.id in self.clicks]
                self.prs_by_repo = get_all_prs(recent)
                self._refresh_ci_statuses(recent)
                self.last_pr_refresh = utc_now()

        return workspaces, session_statuses

    def get_api_data(self) -> dict:
        """Get data for API response."""
        self.initialize()
        workspaces, session_statuses = self.refresh()

        trees = build_hierarchy(
            workspaces,
            self.clicks,
            self.prs_by_repo,
            session_statuses,
            fetch_commits=self.config.get("fetch_commits", True),
        )

        session_duration = (utc_now() - self.session_start).total_seconds()

        statuses_dict = {
            ws_id: {
                "workspace_id": s.workspace_id,
                "status": s.status,
                "model": s.model,
                "is_compacting": s.is_compacting,
            }
            for ws_id, s in session_statuses.items()
        }

        return {
            "trees": trees,
            "session_statuses": statuses_dict,
            "session_start": self.session_start.isoformat(),
            "session_duration_seconds": session_duration,
            "worktree_count": len(self.clicks),
            "stale_minutes": self.config.get("stale_minutes", 30),
        }


# ============================================================================
# HTTP Server
# ============================================================================

class TrackerHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for tracker API."""

    tracker_state: TrackerState = None

    def do_GET(self):
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)

        if parsed.path == "/api/tracker":
            # Check if CI refresh is requested
            if query_params.get("refresh_ci", [""])[0] == "true":
                self.tracker_state.force_refresh_ci()
            self.send_json(self.tracker_state.get_api_data())
        elif parsed.path == "/" or parsed.path == "/index.html":
            self.serve_html()
        else:
            super().do_GET()

    def send_json(self, data: dict):
        """Send JSON response with CORS headers."""
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def serve_html(self):
        """Serve the tracker3d.html file."""
        html_path = Path(__file__).parent / "tracker3d.html"
        if html_path.exists():
            content = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, "tracker3d.html not found")

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class ReusableTCPServer(socketserver.TCPServer):
    """TCP server that allows address reuse to avoid 'Address already in use' errors."""
    allow_reuse_address = True


def run_server(port: int = 8765):
    """Run the HTTP server."""
    config = load_config()
    TrackerHandler.tracker_state = TrackerState(config)

    # Set the directory for serving static files
    os.chdir(Path(__file__).parent)

    with ReusableTCPServer(("", port), TrackerHandler) as httpd:
        print(f"Conductor Worktree Tracker 3D Server")
        print(f"=" * 40)
        print(f"Server running at: http://localhost:{port}")
        print(f"API endpoint: http://localhost:{port}/api/tracker")
        print(f"Database: {CONDUCTOR_DB}")
        print()
        print(f"Press Ctrl+C to stop")
        print()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


def main():
    parser = argparse.ArgumentParser(
        description="Conductor Worktree Tracker 3D - HTTP API Server"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port to run the server on (default: 8765)"
    )
    args = parser.parse_args()

    run_server(args.port)


if __name__ == "__main__":
    main()
