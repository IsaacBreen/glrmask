#!/usr/bin/env python3
"""Register the GLRMask GitHub Actions trusted publisher on every publishable crate.

crates.io Trusted Publishing is configured per crate. This lists each crate's existing
GitHub configurations and creates the missing GLRMask one, skipping crates that already
have an equivalent configuration, so reruns are safe.

Without `--apply` it only prints what it would do. Checking for existing configurations
requires a crates.io API token, because listing trusted publishers is restricted to crate
owners; without `CRATES_IO_TOKEN` the dry run cannot rule out duplicates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Base URL, overridable so tests can point at a stub server.
API = os.environ.get("CRATES_IO_API", "https://crates.io").rstrip("/")
GITHUB_OWNER = "IsaacBreen"
GITHUB_REPOSITORY = "glrmask"
WORKFLOW_FILENAME = "release.yml"
ENVIRONMENT = "release"
USER_AGENT = "GLRMask-release-workflow/1.0 (https://github.com/IsaacBreen/glrmask)"


class ApiError(Exception):
    """A crates.io request that did not succeed."""


def publishable_crates() -> list[dict]:
    """Reuse the release script's crate list, so the two scripts never disagree."""
    path = ROOT / "scripts" / "publish-crates.py"
    spec = importlib.util.spec_from_file_location("publish_crates", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.publishable_crates()


def error_detail(body: bytes) -> str:
    """Extract crates.io's human-readable error detail from an API error body."""
    try:
        payload = json.loads(body)
    except ValueError:
        return body.decode("utf-8", "replace").strip() or "no response body"
    errors = payload.get("errors") if isinstance(payload, dict) else None
    if isinstance(errors, list) and errors:
        return "; ".join(error.get("detail", str(error)) if isinstance(error, dict)
                         else str(error) for error in errors)
    return str(payload)


def request(method: str, url: str, token: str | None, payload: dict | None = None) -> dict:
    """Call the crates.io API, retrying only idempotent requests.

    A failed `POST` is never retried: the server may already have created the config,
    and the next run's listing will skip it.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = token
    if data is not None:
        headers["Content-Type"] = "application/json"

    attempts = 5 if method == "GET" else 1
    for attempt in range(attempts):
        try:
            http_request = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(http_request, timeout=60) as response:
                body = response.read()
                if not body:
                    return {}
                try:
                    return json.loads(body)
                except ValueError as exc:
                    raise ApiError(f"{method} {url} returned invalid JSON: "
                                   f"{body[:200]!r}") from exc
        except urllib.error.HTTPError as exc:
            body = exc.read()
            if method == "GET" and (exc.code == 429 or exc.code >= 500):
                error: Exception = exc
            else:
                raise ApiError(f"HTTP {exc.code}: {error_detail(body)}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if method != "GET":
                raise ApiError(f"{method} {url} failed: {exc}") from exc
            error = exc
        time.sleep(2 ** attempt)
    raise ApiError(f"{method} {url} failed: {error}")


def list_configs(crate: str, token: str | None) -> list[dict]:
    """List every GitHub trusted publishing configuration of a crate."""
    query = urllib.parse.urlencode({"crate": crate, "per_page": 100})
    endpoint = f"{API}/api/v1/trusted_publishing/github_configs"
    url = f"{endpoint}?{query}"
    configs: list[dict] = []
    while url:
        response = request("GET", url, token)
        configs.extend(response.get("github_configs", []))
        next_page = (response.get("meta") or {}).get("next_page")
        url = urllib.parse.urljoin(endpoint, next_page) if next_page else ""
    return configs


def is_glrmask_config(config: dict) -> bool:
    """Match GitHub's case-insensitive owner/repository names, exact workflow/environment."""
    return (
        str(config.get("repository_owner", "")).lower() == GITHUB_OWNER.lower()
        and str(config.get("repository_name", "")).lower() == GITHUB_REPOSITORY.lower()
        and config.get("workflow_filename") == WORKFLOW_FILENAME
        and config.get("environment") == ENVIRONMENT
    )


def describe_config() -> str:
    return (f"{GITHUB_OWNER}/{GITHUB_REPOSITORY} {WORKFLOW_FILENAME} "
            f"(environment {ENVIRONMENT})")


def create_config(crate: str, token: str) -> dict:
    """Create the GLRMask trusted publishing configuration for one crate."""
    payload = {"github_config": {
        "crate": crate,
        "repository_owner": GITHUB_OWNER,
        "repository_name": GITHUB_REPOSITORY,
        "workflow_filename": WORKFLOW_FILENAME,
        "environment": ENVIRONMENT,
    }}
    response = request("POST", f"{API}/api/v1/trusted_publishing/github_configs",
                       token, payload)
    return response.get("github_config", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="only print what would be created (default)")
    mode.add_argument("--apply", action="store_true",
                      help="create the missing trusted publishing configurations")
    args = parser.parse_args()

    token = os.environ.get("CRATES_IO_TOKEN") or None
    if args.apply and not token:
        raise SystemExit("CRATES_IO_TOKEN is required with --apply: create an API token with "
                         "the `trusted-publishing` endpoint scope and a crate scope "
                         "covering the GLRMask crates (e.g. `glrmask*`)")
    if not args.apply:
        print("Dry run: pass --apply to create missing configurations.", flush=True)
        if not token:
            print("CRATES_IO_TOKEN is not set, so existing configurations cannot be listed: "
                  "crates.io only shows trusted publishers to crate owners. Nothing below "
                  "is checked against crates.io.", flush=True)

    crates = publishable_crates()
    print(f"GitHub trusted publisher: {describe_config()}", flush=True)

    created = skipped = failed = 0
    for crate in crates:
        name = crate["name"]
        try:
            existing = list_configs(name, token) if token else None
            config = next((c for c in existing or [] if is_glrmask_config(c)), None)
            if config is not None:
                print(f"{name}: already configured (id {config.get('id')})", flush=True)
                skipped += 1
                continue
            if not args.apply:
                note = "" if existing is not None else " (unverified: no token)"
                print(f"{name}: would create {describe_config()}{note}", flush=True)
                continue
            config = create_config(name, token)
            print(f"{name}: created (id {config.get('id')})", flush=True)
            created += 1
        except ApiError as exc:
            failed += 1
            print(f"{name}: FAILED: {exc}", file=sys.stderr, flush=True)

    if args.apply:
        print(f"Done: {created} created, {skipped} already configured, {failed} failed.",
              flush=True)
    elif token:
        print(f"Done: {skipped} already configured, {failed} failed, "
              f"{len(crates) - skipped - failed} would be created.", flush=True)
    else:
        print(f"Done: {len(crates)} crates would be created (existing configurations "
              f"not checked).", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
