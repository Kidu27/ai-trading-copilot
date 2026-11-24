"""
github_sync.py — GitHub API State Persistence for Render.com

Persists trading memory (trades_log.csv, pattern_history.json, lessons_learned.json)
to a dedicated 'memory' git branch on GitHub using the GitHub REST API.

Features:
  - Zero local impact: If GITHUB_TOKEN is not set, silently no-ops.
  - Automatic branch initialization: Creates the 'memory' branch from default branch if missing.
  - pull_memory_files(): Downloads remote memory files on startup to restore bot state.
  - push_memory_files(): Base64-encodes and commits local memory files to the 'memory' branch.
"""

import base64
import os
from typing import Any

import requests

import broker_executor
import journal
import notifier
import pattern_stats
import post_mortem

_BASE_DIR = os.path.dirname(__file__)


def _get_config() -> tuple[str, str, str]:
    """Return (token, repo, branch)."""
    token = os.getenv("GITHUB_TOKEN", "").strip()
    repo = os.getenv("GITHUB_REPO", "Kidu27/ai-trading-copilot").strip()
    branch = os.getenv("GITHUB_MEMORY_BRANCH", "memory").strip()
    return token, repo, branch


def is_enabled() -> bool:
    """Check if GitHub sync is configured with an API token."""
    token, _, _ = _get_config()
    return bool(token)


def _get_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-trading-copilot-sync",
    }


def _ensure_memory_branch(token: str, repo: str, branch: str) -> bool:
    """Verify that the target memory branch exists; if missing, create it from default branch."""
    headers = _get_headers(token)
    ref_url = f"https://api.github.com/repos/{repo}/git/ref/heads/{branch}"

    try:
        resp = requests.get(ref_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            print(f"  [github_sync] Branch '{branch}' does not exist on {repo}. Creating from default branch...")
            # 1. Get default branch name and head SHA
            repo_info_resp = requests.get(f"https://api.github.com/repos/{repo}", headers=headers, timeout=15)
            default_branch = repo_info_resp.json().get("default_branch", "main") if repo_info_resp.status_code == 200 else "main"

            head_resp = requests.get(f"https://api.github.com/repos/{repo}/git/ref/heads/{default_branch}", headers=headers, timeout=15)
            if head_resp.status_code != 200:
                print(f"  [github_sync] ❌ Could not find base branch '{default_branch}': {head_resp.status_code} {head_resp.text}")
                return False

            base_sha = head_resp.json().get("object", {}).get("sha")
            if not base_sha:
                print(f"  [github_sync] ❌ Missing base SHA for '{default_branch}'.")
                return False

            # 2. Create the branch reference
            create_payload = {
                "ref": f"refs/heads/{branch}",
                "sha": base_sha,
            }
            create_resp = requests.post(
                f"https://api.github.com/repos/{repo}/git/refs",
                headers=headers,
                json=create_payload,
                timeout=15,
            )
            if create_resp.status_code in (200, 201):
                print(f"  [github_sync] ✅ Successfully created '{branch}' branch on {repo}.")
                return True
            else:
                print(f"  [github_sync] ❌ Error creating '{branch}' branch: {create_resp.status_code} {create_resp.text}")
                return False
    except Exception as exc:
        print(f"  [github_sync] Exception checking/creating branch: {exc}")
        return False


def _get_target_files() -> list[tuple[str, str]]:
    """Return list of (local_absolute_path, repo_relative_filename)."""
    return [
        (journal.LOG_FILE, "trades_log.csv"),
        (pattern_stats.PATTERN_HISTORY_FILE, "pattern_history.json"),
        (post_mortem.LESSONS_FILE, "lessons_learned.json"),
        (notifier.PENDING_SIGNALS_FILE, "pending_signals.json"),
        (broker_executor.ACTIVE_TRADES_FILE, "active_trades.json"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public Operations
# ─────────────────────────────────────────────────────────────────────────────

def pull_memory_files() -> bool:
    """
    On startup, verify the memory branch and download existing state files
    (trades_log.csv, pattern_history.json, lessons_learned.json) to disk.
    """
    if not is_enabled():
        return False

    token, repo, branch = _get_config()
    headers = _get_headers(token)

    if not _ensure_memory_branch(token, repo, branch):
        return False

    files = _get_target_files()
    pulled_count = 0

    for local_path, filename in files:
        url = f"https://api.github.com/repos/{repo}/contents/{filename}?ref={branch}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                content_b64 = data.get("content", "")
                encoding = data.get("encoding", "base64")
                if encoding == "base64" and content_b64:
                    raw_bytes = base64.b64decode(content_b64.encode("ascii"))
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    with open(local_path, "wb") as fh:
                        fh.write(raw_bytes)
                    pulled_count += 1
                    print(f"  [github_sync] 📥 Restored {filename} from '{branch}' branch ({len(raw_bytes)} bytes).")
            elif resp.status_code == 404:
                print(f"  [github_sync] ℹ️ {filename} not yet on '{branch}' branch (initial run).")
            else:
                print(f"  [github_sync] ⚠️ Failed to fetch {filename}: {resp.status_code}")
        except Exception as exc:
            print(f"  [github_sync] Error pulling {filename}: {exc}")

    print(f"  [github_sync] Memory pull complete: {pulled_count}/{len(files)} files restored.")
    return pulled_count > 0


def push_memory_files(commit_msg: str = "Update trading memory state") -> bool:
    """
    Commit and push current local memory files to the dedicated memory branch.
    Fetches the current remote file SHA first to ensure safe updates.
    """
    if not is_enabled():
        return False

    token, repo, branch = _get_config()
    headers = _get_headers(token)

    files = _get_target_files()
    success_count = 0

    for local_path, filename in files:
        if not os.path.exists(local_path):
            continue

        try:
            with open(local_path, "rb") as fh:
                raw_bytes = fh.read()

            b64_content = base64.b64encode(raw_bytes).decode("ascii")

            # 1. Fetch current remote SHA if file exists on memory branch
            url = f"https://api.github.com/repos/{repo}/contents/{filename}?ref={branch}"
            get_resp = requests.get(url, headers=headers, timeout=15)
            current_sha = None
            if get_resp.status_code == 200:
                current_sha = get_resp.json().get("sha")

            # 2. Put file to commit on memory branch
            put_url = f"https://api.github.com/repos/{repo}/contents/{filename}"
            put_payload: dict[str, Any] = {
                "message": f"[memory] {commit_msg} ({filename})",
                "content": b64_content,
                "branch": branch,
            }
            if current_sha:
                put_payload["sha"] = current_sha

            put_resp = requests.put(put_url, headers=headers, json=put_payload, timeout=20)

            # Retry once on 409 conflict
            if put_resp.status_code == 409:
                latest_resp = requests.get(url, headers=headers, timeout=15)
                if latest_resp.status_code == 200:
                    put_payload["sha"] = latest_resp.json().get("sha")
                    put_resp = requests.put(put_url, headers=headers, json=put_payload, timeout=20)

            if put_resp.status_code in (200, 201):
                success_count += 1
                print(f"  [github_sync] 📤 Pushed {filename} to '{branch}' branch ({len(raw_bytes)} bytes).")
            else:
                print(f"  [github_sync] ⚠️ Failed to commit {filename}: {put_resp.status_code} {put_resp.text}")
        except Exception as exc:
            print(f"  [github_sync] Error pushing {filename}: {exc}")

    return success_count > 0
