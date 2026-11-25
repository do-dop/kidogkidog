"""Apply local query normalization while using real staging search and media.

Run: python3.12 scripts/local_search_proxy.py
Point VITE_API_PROXY_TARGET at http://127.0.0.1:8098.
No stored events, search results, or database records are replaced.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.search.query_normalization import normalize_search_query
from pipeline.search.search_policy import contradicts_action
from pipeline.search.result_grouping import group_results_by_video
from pipeline.search.scene_answer import build_scene_answer


UPSTREAM = os.getenv("SEARCH_PROXY_UPSTREAM", "https://kidog-api-staging-ptiwpgodca-du.a.run.app").rstrip("/")
PORT = int(os.getenv("SEARCH_PROXY_PORT", "8098"))
_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade"}


def apply_search_policy(result, query, limit):
    """Filter explicit contradictions and group a larger real upstream pool."""
    if (result.get("query_type") == "sequence" or result.get("aggregate_result") is not None
            or result.get("clarification")):
        return result
    had_results = bool(result.get("results"))
    scenes = []
    for card in result.get("results") or []:
        for scene in card.get("related_scenes") or [card]:
            if not contradicts_action(query, scene):
                scenes.append(scene)
    grouped = group_results_by_video(scenes, limit)
    result["results"] = grouped
    result["evidence_items"] = grouped
    if "behavior_events" in result:
        result["behavior_events"] = [event for event in result["behavior_events"] or []
                                     if not contradicts_action(query, event)]
    if grouped:
        result["answer"] = build_scene_answer(grouped)
    elif had_results:
        result["answer"] = build_scene_answer([])
    return result


class SearchProxy(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _forward(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
        original_query = None
        search_query = None
        result_limit = 3
        if self.command == "POST" and self.path.split("?")[0] == "/query" and body:
            try:
                request = json.loads(body)
                if isinstance(request.get("query"), str):
                    original_query = request["query"]
                    search_query = normalize_search_query(original_query)
                    request["query"] = search_query
                    result_limit = max(1, min(int(request.get("top_k") or 3), 10))
                    if not request.get("video_id"):
                        request["top_k"] = 10
                    body = json.dumps(request, ensure_ascii=False).encode()
            except (ValueError, AttributeError):
                pass  # Let the upstream validate malformed input.
        headers = {key: value for key, value in self.headers.items()
                   if key.lower() not in _HOP_HEADERS | {"host", "content-length", "accept-encoding"}}
        request = urllib.request.Request(UPSTREAM + self.path, data=body,
                                         headers=headers, method=self.command)
        try:
            try:
                response = urllib.request.urlopen(request, timeout=120)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                payload = None
                if original_query is not None and response.status == 200:
                    result = json.load(response)
                    result = apply_search_policy(result, search_query, result_limit)
                    result["query"] = original_query
                    result["top_k"] = result_limit
                    payload = json.dumps(result, ensure_ascii=False).encode()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in _HOP_HEADERS and not (payload is not None and key.lower() == "content-length"):
                        self.send_header(key, value)
                if payload is not None:
                    self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if self.command == "HEAD":
                    return
                if payload is not None:
                    self.wfile.write(payload)
                else:
                    while chunk := response.read(64 * 1024):
                        self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (urllib.error.URLError, TimeoutError):
            self.send_error(502, "Upstream API unavailable")

    do_GET = _forward
    do_HEAD = _forward
    do_POST = _forward
    do_DELETE = _forward
    do_OPTIONS = _forward


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), SearchProxy)
    print(f"Local search normalization proxy: http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
