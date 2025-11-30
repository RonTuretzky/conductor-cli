#!/usr/bin/env python3
"""
Conductor Worktree Activity Tracker - 3D Visualization

A 3D interactive visualization of Conductor worktree activity using VisPy.
Displays PR hierarchies as a 3D graph with animated nodes and connections.

Requirements:
    pip install vispy PyQt6 numpy

Usage:
    python3 tracker3d.py
    python3 tracker3d.py --tree  # Use tree layout instead of radial
    python3 tracker3d.py --once  # Render once and exit
"""

import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from vispy import app, scene, color
    from vispy.scene import visuals
    from vispy.visuals.transforms import STTransform
except ImportError:
    print("Error: VisPy is required for 3D visualization.")
    print("Install with: pip install vispy PyQt6 numpy")
    sys.exit(1)

# ============================================================================
# Configuration (shared with tracker.py)
# ============================================================================

CONDUCTOR_DB = Path.home() / "Library/Application Support/com.conductor.app/conductor.db"
CONFIG_FILE = Path(__file__).parent / "config.json"

DEFAULT_CONFIG = {
    "stale_minutes": 30,
    "interval_seconds": 2.0,
    "show_hierarchy": True,
    "tree_layout": False,
    "fetch_prs": True,
    "since": "today",
    "debug": False,
    "pr_refresh_minutes": 5,
    "show_status": True,
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


@dataclass
class ClickRecord:
    workspace_id: str
    name: str
    repo_name: str
    branch: str
    first_seen: datetime
    last_seen: datetime


def _utc_now_factory():
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
    title: str = ""


@dataclass
class SessionStatus:
    workspace_id: str
    status: str  # 'idle', 'working', 'error'
    model: Optional[str] = None
    is_compacting: bool = False


@dataclass
class TreeNode:
    name: str
    branch: str
    workspace_id: Optional[str] = None
    pr_number: Optional[int] = None
    pr_title: Optional[str] = None
    click_record: Optional[ClickRecord] = None
    children: list["TreeNode"] = field(default_factory=list)
    is_default_branch: bool = False
    session_status: Optional[SessionStatus] = None


# ============================================================================
# Database & GitHub Access (from tracker.py)
# ============================================================================

def utc_now() -> datetime:
    from datetime import timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)


def get_since_cutoff(since_value) -> datetime:
    from datetime import timezone
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


def get_workspaces() -> list[Workspace]:
    if not CONDUCTOR_DB.exists():
        print(f"Error: Conductor database not found at {CONDUCTOR_DB}", file=sys.stderr)
        return []

    conn = sqlite3.connect(f"file:{CONDUCTOR_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            w.id, w.directory_name, w.branch, w.parent_branch,
            w.repository_id, w.updated_at,
            r.name as repo_name, r.remote_url, r.default_branch
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
        ))

    conn.close()
    return workspaces


def get_session_statuses() -> dict[str, SessionStatus]:
    if not CONDUCTOR_DB.exists():
        return {}

    conn = sqlite3.connect(f"file:{CONDUCTOR_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT s.workspace_id, s.status, s.model, s.is_compacting
        FROM sessions s
        WHERE s.workspace_id IS NOT NULL AND s.is_hidden = 0
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


def parse_github_remote(remote_url: str) -> Optional[tuple[str, str]]:
    ssh_match = re.match(r"git@github\.com:([^/]+)/([^.]+)(?:\.git)?", remote_url)
    if ssh_match:
        return ssh_match.group(1), ssh_match.group(2)
    https_match = re.match(r"https://github\.com/([^/]+)/([^.]+)(?:\.git)?", remote_url)
    if https_match:
        return https_match.group(1), https_match.group(2)
    return None


def get_open_prs(owner: str, repo: str) -> list[PRInfo]:
    try:
        result = subprocess.run(
            ["gh", "pr", "list", "--repo", f"{owner}/{repo}", "--state", "open",
             "--json", "number,headRefName,baseRefName,title"],
            capture_output=True, text=True, timeout=10,
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


def get_all_prs(workspaces: list[Workspace]) -> dict[str, list[PRInfo]]:
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
    session: Session,
    prs_by_repo: dict[str, list[PRInfo]],
    session_statuses: dict[str, SessionStatus] = None,
    include_pr_branches: bool = True,
) -> dict[str, TreeNode]:
    """Build hierarchy tree for each repo with active workspaces."""
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

        pr_lookup: dict[str, tuple[str, int, str]] = {}
        for pr in prs:
            pr_lookup[pr.head_branch] = (pr.base_branch, pr.number, pr.title)

        root = TreeNode(name=repo_name, branch="", is_default_branch=False)
        nodes: dict[str, TreeNode] = {}
        workspace_by_branch: dict[str, Workspace] = {}

        for ws in repo_workspaces:
            workspace_by_branch[ws.branch] = ws
            if ws.id in session.clicks:
                click = session.clicks.get(ws.id)
                pr_info = pr_lookup.get(ws.branch)
                status = session_statuses.get(ws.id) if session_statuses else None
                if pr_info and pr_info[2]:
                    display_name = pr_info[2]
                else:
                    display_name = ws.name
                node = TreeNode(
                    name=display_name,
                    branch=ws.branch,
                    workspace_id=ws.id,
                    pr_number=pr_info[1] if pr_info else None,
                    pr_title=pr_info[2] if pr_info else None,
                    click_record=click,
                    session_status=status,
                )
                nodes[ws.branch] = node

        if include_pr_branches:
            branches_to_add = set()
            for branch in list(nodes.keys()):
                current = branch
                while current in pr_lookup:
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
                )
                nodes[branch] = node

        attached = set()
        for branch, node in nodes.items():
            pr_info = pr_lookup.get(branch)
            if pr_info:
                parent_branch = pr_info[0]
                if parent_branch in nodes:
                    nodes[parent_branch].children.append(node)
                    attached.add(branch)
                    continue

            ws = workspace_by_branch.get(branch)
            if ws and ws.parent_branch and ws.parent_branch in nodes:
                nodes[ws.parent_branch].children.append(node)
                attached.add(branch)
                continue

        for branch, node in nodes.items():
            if branch not in attached:
                root.children.append(node)

        if root.children:
            trees[repo_name] = root

    return trees


# ============================================================================
# 3D Graph Layout
# ============================================================================

@dataclass
class Node3D:
    """A node in the 3D graph."""
    id: str
    label: str
    position: np.ndarray  # [x, y, z]
    color: np.ndarray  # [r, g, b, a]
    size: float
    status: str  # 'idle', 'working', 'error', 'compacting', 'none'
    minutes_since_activity: int
    children: list[str] = field(default_factory=list)
    parent: Optional[str] = None
    is_repo: bool = False
    # Additional info for display
    pr_number: Optional[int] = None
    pr_title: Optional[str] = None
    worktree_name: Optional[str] = None
    time_str: str = ""  # "5m ago", "now", etc.
    last_seen_time: Optional[str] = None  # "14:30" format
    is_compacting: bool = False
    has_worktree: bool = True  # False for intermediate PR branches


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


def format_datetime(dt: datetime) -> str:
    """Format datetime as HH:MM."""
    return dt.strftime("%H:%M")


def calculate_3d_layout(
    trees: dict[str, TreeNode],
    stale_threshold: int,
) -> tuple[list[Node3D], list[tuple[str, str]]]:
    """
    Calculate 3D positions for all nodes in the hierarchy.

    Returns:
        nodes: List of Node3D objects
        edges: List of (parent_id, child_id) tuples
    """
    nodes: list[Node3D] = []
    edges: list[tuple[str, str]] = []

    repo_list = list(trees.items())
    n_repos = len(repo_list)

    if n_repos == 0:
        return nodes, edges

    # Place repos in a circle on the XZ plane (Y is up)
    repo_radius = 8.0

    def get_color_for_minutes(minutes: int, status: str, is_compacting: bool = False) -> np.ndarray:
        """Get color based on activity time and status."""
        if is_compacting:
            # Yellow/gold for compacting
            return np.array([1.0, 0.85, 0.2, 1.0])
        elif status == 'working':
            # Bright green for working
            return np.array([0.2, 1.0, 0.3, 1.0])
        elif status == 'error':
            # Red for error
            return np.array([1.0, 0.2, 0.2, 1.0])
        elif status == 'idle':
            # Color based on time since activity
            if minutes < 5:
                return np.array([0.3, 0.8, 0.4, 1.0])  # Green
            elif minutes < 15:
                return np.array([0.9, 0.9, 0.3, 1.0])  # Yellow
            elif minutes < stale_threshold:
                return np.array([1.0, 0.6, 0.2, 1.0])  # Orange
            else:
                return np.array([0.8, 0.2, 0.2, 1.0])  # Red (stale)
        else:
            # Default gray for no status / intermediate branches
            return np.array([0.5, 0.5, 0.6, 0.6])

    def add_node_recursive(
        tree_node: TreeNode,
        parent_id: Optional[str],
        base_pos: np.ndarray,
        angle: float,
        angle_span: float,
        depth: int,
        repo_name: str,
    ):
        """Recursively add nodes to the 3D graph."""
        # Create unique ID
        if tree_node.workspace_id:
            node_id = tree_node.workspace_id
        elif tree_node.branch:
            node_id = f"{repo_name}:{tree_node.branch}"
        else:
            node_id = f"repo:{repo_name}"

        # Calculate position
        # Depth increases Y (height), angle determines XZ position
        distance = 2.0 + depth * 1.5
        y = depth * 1.2
        x = base_pos[0] + distance * math.cos(angle)
        z = base_pos[2] + distance * math.sin(angle)
        position = np.array([x, y, z])

        # Determine status and color
        status = 'none'
        minutes = 0
        time_str = ""
        last_seen_time = None
        is_compacting = False
        has_worktree = tree_node.click_record is not None

        if tree_node.session_status:
            status = tree_node.session_status.status
            is_compacting = tree_node.session_status.is_compacting

        if tree_node.click_record:
            minutes = int((utc_now() - tree_node.click_record.last_seen).total_seconds() / 60)
            time_str = format_time_ago(tree_node.click_record.last_seen)
            last_seen_time = format_datetime(tree_node.click_record.last_seen)
            if status == 'none':
                status = 'idle'

        color = get_color_for_minutes(minutes, status, is_compacting)

        # Size based on activity - working nodes are larger
        if status == 'working':
            size = 0.6
        elif tree_node.click_record:
            size = 0.45 + min(0.3, minutes / 60)  # Grows with time
        else:
            size = 0.25  # Smaller for branches without worktrees

        # Create label - show PR title if available
        if tree_node.pr_title:
            label = tree_node.pr_title[:30] if len(tree_node.pr_title) > 30 else tree_node.pr_title
        else:
            label = tree_node.name[:25] if len(tree_node.name) > 25 else tree_node.name

        node = Node3D(
            id=node_id,
            label=label,
            position=position,
            color=color,
            size=size,
            status=status,
            minutes_since_activity=minutes,
            parent=parent_id,
            is_repo=False,
            pr_number=tree_node.pr_number,
            pr_title=tree_node.pr_title,
            worktree_name=tree_node.name if tree_node.click_record else None,
            time_str=time_str,
            last_seen_time=last_seen_time,
            is_compacting=is_compacting,
            has_worktree=has_worktree,
        )
        nodes.append(node)

        if parent_id:
            edges.append((parent_id, node_id))

        # Add children
        n_children = len(tree_node.children)
        if n_children > 0:
            child_angle_span = angle_span / max(n_children, 1)
            start_angle = angle - angle_span / 2 + child_angle_span / 2

            for i, child in enumerate(tree_node.children):
                child_angle = start_angle + i * child_angle_span
                add_node_recursive(
                    child,
                    node_id,
                    position,
                    child_angle,
                    child_angle_span * 0.8,  # Narrow as we go deeper
                    depth + 1,
                    repo_name,
                )

    # Process each repo
    for i, (repo_name, root) in enumerate(repo_list):
        # Position repo at the center of its sector
        repo_angle = (2 * math.pi * i / n_repos) - math.pi / 2
        repo_x = repo_radius * math.cos(repo_angle)
        repo_z = repo_radius * math.sin(repo_angle)
        repo_pos = np.array([repo_x, 0.0, repo_z])

        # Create repo node
        repo_id = f"repo:{repo_name}"
        repo_node = Node3D(
            id=repo_id,
            label=repo_name,
            position=repo_pos,
            color=np.array([0.3, 0.5, 0.9, 1.0]),  # Blue for repos
            size=0.6,
            status='none',
            minutes_since_activity=0,
            is_repo=True,
        )
        nodes.append(repo_node)

        # Add children (worktrees) radiating outward
        n_children = len(root.children)
        if n_children > 0:
            # Spread children in a cone
            angle_span = math.pi / 2  # 90 degrees per repo
            start_angle = repo_angle - angle_span / 2
            child_angle_span = angle_span / max(n_children, 1)

            for j, child in enumerate(root.children):
                child_angle = start_angle + (j + 0.5) * child_angle_span
                add_node_recursive(
                    child,
                    repo_id,
                    repo_pos,
                    child_angle,
                    child_angle_span * 0.6,
                    1,
                    repo_name,
                )

    return nodes, edges


# ============================================================================
# VisPy 3D Visualization
# ============================================================================

class WorktreeVisualization:
    """3D visualization of Conductor worktree activity."""

    def __init__(self, config: dict, args):
        self.config = config
        self.args = args
        self.stale_threshold = args.stale

        # Session state
        self.session = Session()
        self.prs_by_repo: dict[str, list[PRInfo]] = {}
        self.last_pr_refresh = datetime.min

        # VisPy setup
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            title='Conductor Worktree Tracker 3D',
            size=(1200, 800),
            show=True,
            bgcolor='#1a1a2e',
        )

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=60,
            distance=25,
            elevation=30,
            azimuth=45,
        )

        # Visual elements
        self.node_markers = None
        self.edge_lines = None
        self.time_rings = []  # Time bar rings around nodes
        self.status_rings = []  # Status indicator rings
        self.labels = []
        self.node_data: list[Node3D] = []

        # Animation state
        self.animation_phase = 0.0
        self.spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.spinner_idx = 0

        # Create status text
        self.status_text = scene.Text(
            "Loading...",
            color='white',
            font_size=12,
            anchor_x='left',
            anchor_y='top',
            parent=self.canvas.scene,
        )
        self.status_text.transform = STTransform(translate=(10, 20, 0))

        # Timer for updates
        self.timer = app.Timer(interval=config["interval_seconds"], connect=self.on_timer)

        # Initial data load
        self._initial_load()

    def _initial_load(self):
        """Load initial data from database."""
        print("Loading workspaces...")
        workspaces = get_workspaces()

        # Initialize timestamps
        for ws in workspaces:
            self.session.last_updated_at[ws.id] = ws.updated_at

        # Backfill recent activity
        cutoff = get_since_cutoff(self.args.since)
        for ws in workspaces:
            if ws.updated_at >= cutoff:
                self.session.clicks[ws.id] = ClickRecord(
                    workspace_id=ws.id,
                    name=ws.name,
                    repo_name=ws.repo_name,
                    branch=ws.branch,
                    first_seen=ws.updated_at,
                    last_seen=ws.updated_at,
                )

        print(f"Found {len(self.session.clicks)} active worktrees")

        # Load PRs
        if not self.args.no_prs:
            print("Fetching PRs...")
            self.prs_by_repo = get_all_prs(workspaces)
            self.last_pr_refresh = utc_now()
            total_prs = sum(len(prs) for prs in self.prs_by_repo.values())
            print(f"Loaded {total_prs} open PRs")

        # Update visualization
        self._update_visualization()

        # Start timer
        self.timer.start()

    def _update_visualization(self):
        """Update the 3D visualization with current data."""
        workspaces = get_workspaces()
        session_statuses = get_session_statuses()

        # Refresh PRs periodically
        pr_refresh_seconds = self.config["pr_refresh_minutes"] * 60
        if not self.args.no_prs and (utc_now() - self.last_pr_refresh).total_seconds() > pr_refresh_seconds:
            self.prs_by_repo = get_all_prs(workspaces)
            self.last_pr_refresh = utc_now()

        # Detect new clicks
        for ws in workspaces:
            last_known = self.session.last_updated_at.get(ws.id)
            if last_known is not None:
                delta_seconds = (ws.updated_at - last_known).total_seconds()
                if delta_seconds > 1.0:
                    if ws.id in self.session.clicks:
                        self.session.clicks[ws.id].last_seen = utc_now()
                    else:
                        self.session.clicks[ws.id] = ClickRecord(
                            workspace_id=ws.id,
                            name=ws.name,
                            repo_name=ws.repo_name,
                            branch=ws.branch,
                            first_seen=utc_now(),
                            last_seen=utc_now(),
                        )
            self.session.last_updated_at[ws.id] = ws.updated_at

        # Build hierarchy
        trees = build_hierarchy(
            workspaces,
            self.session,
            self.prs_by_repo,
            session_statuses=session_statuses,
        )

        # Calculate 3D layout
        nodes, edges = calculate_3d_layout(trees, self.stale_threshold)
        self.node_data = nodes

        # Clear old labels and rings
        for label in self.labels:
            label.parent = None
        self.labels.clear()

        for ring in self.time_rings:
            ring.parent = None
        self.time_rings.clear()

        for ring in self.status_rings:
            ring.parent = None
        self.status_rings.clear()

        if not nodes:
            self.status_text.text = "No worktrees visited. Click on workspaces in Conductor."
            if self.node_markers:
                self.node_markers.parent = None
                self.node_markers = None
            if self.edge_lines:
                self.edge_lines.parent = None
                self.edge_lines = None
            return

        # Create node lookup
        node_lookup = {n.id: n for n in nodes}

        # Prepare node data for markers
        positions = np.array([n.position for n in nodes])
        colors = np.array([n.color for n in nodes])
        sizes = np.array([n.size * 50 for n in nodes])  # Scale for visibility

        # Create/update markers
        if self.node_markers:
            self.node_markers.parent = None

        self.node_markers = visuals.Markers()
        self.node_markers.set_data(
            positions,
            face_color=colors,
            edge_color=colors * 0.7,
            edge_width=2,
            size=sizes,
            symbol='o',
        )
        self.view.add(self.node_markers)

        # Create edges
        if edges:
            edge_positions = []
            edge_colors = []
            for parent_id, child_id in edges:
                parent = node_lookup.get(parent_id)
                child = node_lookup.get(child_id)
                if parent and child:
                    edge_positions.append(parent.position)
                    edge_positions.append(child.position)
                    # Edge color is blend of parent and child
                    edge_color = (parent.color + child.color) / 2
                    edge_color[3] = 0.6  # Semi-transparent
                    edge_colors.append(edge_color)
                    edge_colors.append(edge_color)

            if edge_positions:
                if self.edge_lines:
                    self.edge_lines.parent = None

                self.edge_lines = visuals.Line(
                    pos=np.array(edge_positions),
                    color=np.array(edge_colors),
                    width=2,
                    connect='segments',
                )
                self.view.add(self.edge_lines)

        # Create time indicator rings around nodes (like time bars in terminal)
        for node in nodes:
            if node.is_repo or not node.has_worktree:
                continue

            # Calculate ring size based on time (like the bar width in terminal)
            minutes = node.minutes_since_activity
            if minutes < 5:
                ring_scale = 0.3 + (minutes / 5) * 0.2
            elif minutes < 15:
                ring_scale = 0.5 + ((minutes - 5) / 10) * 0.2
            elif minutes < self.stale_threshold:
                ring_scale = 0.7 + ((minutes - 15) / (self.stale_threshold - 15)) * 0.2
            else:
                ring_scale = 0.9 + min(0.1, (minutes - self.stale_threshold) / 30)

            ring_scale = min(1.0, ring_scale)

            # Create a ring using a circle of points
            n_points = 32
            theta = np.linspace(0, 2 * np.pi, n_points)
            ring_radius = node.size * (1.5 + ring_scale)

            # Ring in XZ plane around the node
            ring_points = np.zeros((n_points, 3))
            ring_points[:, 0] = node.position[0] + ring_radius * np.cos(theta)
            ring_points[:, 1] = node.position[1]
            ring_points[:, 2] = node.position[2] + ring_radius * np.sin(theta)

            # Ring color matches node color
            ring_color = node.color.copy()
            ring_color[3] = 0.6  # Semi-transparent

            ring = visuals.Line(
                pos=ring_points,
                color=ring_color,
                width=3,
                connect='strip',
            )
            self.view.add(ring)
            self.time_rings.append(ring)

        # Create status indicator rings for working/error nodes
        for node in nodes:
            if node.is_repo:
                continue

            if node.status == 'working' or node.is_compacting:
                # Outer pulsing ring for working status
                n_points = 32
                theta = np.linspace(0, 2 * np.pi, n_points)
                ring_radius = node.size * 2.5

                ring_points = np.zeros((n_points, 3))
                ring_points[:, 0] = node.position[0] + ring_radius * np.cos(theta)
                ring_points[:, 1] = node.position[1]
                ring_points[:, 2] = node.position[2] + ring_radius * np.sin(theta)

                if node.is_compacting:
                    ring_color = np.array([1.0, 0.85, 0.2, 0.8])  # Yellow
                else:
                    ring_color = np.array([0.2, 1.0, 0.3, 0.8])  # Green

                ring = visuals.Line(
                    pos=ring_points,
                    color=ring_color,
                    width=4,
                    connect='strip',
                )
                self.view.add(ring)
                self.status_rings.append((ring, node))

        # Create labels for nodes with full info
        for node in nodes:
            if node.is_repo:
                # Repo label - just the name
                label_text = node.label
                label_color = (0.6, 0.8, 1.0, 1.0)  # Light blue
            else:
                # Build multi-line label with all info
                lines = []

                # Status icon prefix
                if node.is_compacting:
                    status_icon = "⟳"  # Compacting
                elif node.status == 'working':
                    status_icon = "●"  # Working (will pulse)
                elif node.status == 'error':
                    status_icon = "✗"  # Error
                elif node.status == 'idle':
                    status_icon = "○"  # Idle
                else:
                    status_icon = ""

                # PR number and title
                if node.pr_number:
                    pr_line = f"#{node.pr_number}"
                    if node.pr_title:
                        title = node.pr_title[:25] + "..." if len(node.pr_title) > 25 else node.pr_title
                        pr_line += f": {title}"
                    lines.append(pr_line)
                elif node.label:
                    lines.append(node.label)

                # Time info (only for worktrees with activity)
                if node.has_worktree and node.time_str:
                    time_line = f"{status_icon} {node.time_str}"
                    if node.last_seen_time:
                        time_line += f" @ {node.last_seen_time}"
                    lines.append(time_line)
                elif not node.has_worktree:
                    lines.append("(no worktree)")

                label_text = "\n".join(lines)

                # Color based on status
                if node.is_compacting:
                    label_color = (1.0, 0.85, 0.2, 1.0)  # Yellow
                elif node.status == 'working':
                    label_color = (0.4, 1.0, 0.5, 1.0)  # Bright green
                elif node.status == 'error':
                    label_color = (1.0, 0.4, 0.4, 1.0)  # Red
                elif node.minutes_since_activity >= self.stale_threshold:
                    label_color = (1.0, 0.5, 0.5, 1.0)  # Red-ish for stale
                elif not node.has_worktree:
                    label_color = (0.6, 0.6, 0.7, 0.7)  # Dimmed
                else:
                    label_color = (1.0, 1.0, 1.0, 0.95)  # White

            label = scene.Text(
                label_text,
                color=label_color,
                font_size=10 if node.is_repo else 8,
                bold=node.is_repo,
                anchor_x='center',
                anchor_y='bottom',
                parent=self.view.scene,
            )
            # Position label above node
            label_pos = node.position.copy()
            label_pos[1] += node.size + 0.4
            label.transform = STTransform(translate=label_pos)
            self.labels.append(label)

        # Update status panel with detailed info like terminal version
        session_duration = utc_now() - self.session.started_at
        session_mins = int(session_duration.total_seconds() / 60)
        if session_mins >= 60:
            session_str = f"{session_mins // 60}h {session_mins % 60}m"
        else:
            session_str = f"{session_mins}m"

        # Count by status
        working_nodes = [n for n in nodes if n.status == 'working' and not n.is_repo]
        idle_nodes = [n for n in nodes if n.status == 'idle' and not n.is_repo]
        error_nodes = [n for n in nodes if n.status == 'error' and not n.is_repo]
        compacting_nodes = [n for n in nodes if n.is_compacting and not n.is_repo]
        stale_nodes = [n for n in nodes if n.minutes_since_activity >= self.stale_threshold and not n.is_repo]

        # Build status lines
        status_lines = [
            "━━━ CONDUCTOR WORKTREE TRACKER 3D ━━━",
            "",
        ]

        # Status counts with icons
        status_parts = []
        if working_nodes:
            status_parts.append(f"● Working: {len(working_nodes)}")
        if idle_nodes:
            status_parts.append(f"○ Idle: {len(idle_nodes)}")
        if error_nodes:
            status_parts.append(f"✗ Error: {len(error_nodes)}")
        if compacting_nodes:
            status_parts.append(f"⟳ Compacting: {len(compacting_nodes)}")

        if status_parts:
            status_lines.append(" | ".join(status_parts))

        # List working worktrees
        if working_nodes:
            names = [n.pr_title or n.label for n in working_nodes[:3]]
            if len(names) > 3:
                names_str = ", ".join(names) + f" +{len(working_nodes) - 3} more"
            else:
                names_str = ", ".join(names)
            status_lines.append(f"  Active: {names_str}")

        # Stale warning
        if stale_nodes:
            stale_names = [n.pr_title or n.label for n in stale_nodes[:2]]
            if len(stale_nodes) > 2:
                stale_str = ", ".join(stale_names) + f" +{len(stale_nodes) - 2} more"
            else:
                stale_str = ", ".join(stale_names)
            status_lines.append(f"  ⚠ Stale ({self.stale_threshold}m+): {stale_str}")

        status_lines.append("")
        status_lines.append(f"Tracking: {len(self.session.clicks)} worktrees | Session: {session_str}")
        status_lines.append("Controls: Drag=Rotate, Scroll=Zoom, Shift+Drag=Pan")

        self.status_text.text = "\n".join(status_lines)

        self.canvas.update()

    def on_timer(self, event):
        """Timer callback for periodic updates."""
        self.animation_phase += 0.15
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)

        # Animate working nodes (pulsing)
        if self.node_markers and self.node_data:
            colors = []
            sizes = []
            for node in self.node_data:
                c = node.color.copy()
                s = node.size * 50

                if node.status == 'working' or node.is_compacting:
                    # Pulsing effect for working/compacting
                    pulse = 0.5 + 0.5 * math.sin(self.animation_phase * 4)
                    c[3] = 0.7 + 0.3 * pulse
                    s = s * (0.85 + 0.3 * pulse)

                colors.append(c)
                sizes.append(s)

            self.node_markers.set_data(
                np.array([n.position for n in self.node_data]),
                face_color=np.array(colors),
                edge_color=np.array(colors) * 0.7,
                edge_width=2,
                size=np.array(sizes),
                symbol='o',
            )

        # Animate status rings (expanding/pulsing)
        for ring, node in self.status_rings:
            pulse = 0.5 + 0.5 * math.sin(self.animation_phase * 4)

            # Expand/contract the ring
            n_points = 32
            theta = np.linspace(0, 2 * np.pi, n_points)
            ring_radius = node.size * (2.2 + 0.6 * pulse)

            ring_points = np.zeros((n_points, 3))
            ring_points[:, 0] = node.position[0] + ring_radius * np.cos(theta)
            ring_points[:, 1] = node.position[1]
            ring_points[:, 2] = node.position[2] + ring_radius * np.sin(theta)

            # Update ring color alpha for pulsing
            if node.is_compacting:
                ring_color = np.array([1.0, 0.85, 0.2, 0.4 + 0.5 * pulse])
            else:
                ring_color = np.array([0.2, 1.0, 0.3, 0.4 + 0.5 * pulse])

            ring.set_data(pos=ring_points, color=ring_color)

        # Periodic data refresh
        self._update_visualization()

    def run(self):
        """Run the visualization."""
        if self.args.once:
            self._update_visualization()
            # Just show for a moment
            app.run()
        else:
            app.run()


# ============================================================================
# Main
# ============================================================================

def main():
    config = load_config()

    parser = argparse.ArgumentParser(
        description="3D visualization of Conductor worktree activity"
    )
    parser.add_argument(
        "--stale", type=int, default=config["stale_minutes"],
        help=f"Minutes before a worktree is marked as stale (default: {config['stale_minutes']})"
    )
    parser.add_argument(
        "--no-prs", action="store_true", default=not config["fetch_prs"],
        help="Skip GitHub PR queries (faster startup)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Render once and exit"
    )
    parser.add_argument(
        "--since", type=str, default=config["since"],
        help="Activity lookback: 'today' or hours (default: today)"
    )

    args = parser.parse_args()

    print("Conductor Worktree Tracker - 3D Visualization")
    print("=" * 50)

    viz = WorktreeVisualization(config, args)
    viz.run()


if __name__ == "__main__":
    main()
