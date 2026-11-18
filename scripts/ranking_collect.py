"""스테이징 API에서 전역 검색 순위 실험용 데이터를 모은다(읽기 전용, 질문당 호출 2회).

세트 A: evaluation/data/labeling.final.csv의 semantic 질문, 세트 B: retrieval-r1 질문(0037·0949는 재색인 뒤라 제외하지 않음).
각 질문에 대해 (1) 전역 top_k=10 카드의 영상·점수, (2) 정답 영상으로 한정한 검색의 최고 점수를 저장한다.
영상별 이벤트 설명은 /suggestions에서 받는다.
"""
import csv, json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://kidog-api-staging-700543886823.asia-northeast3.run.app"
OUT = ROOT / "experiments/ranking-2026-11-19/data.json"


def post(body):
    req = urllib.request.Request(BASE + "/query", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


def cards(res):
    return [{"video_id": r["video_id"], "score": float(r.get("score") or 0)} for r in res.get("results", [])]


def collect(item):
    g = post({"query": item["query"], "top_k": 10})
    s = post({"query": item["query"], "top_k": 5, "video_id": item["gold_video"]})
    return {**item, "global": cards(g), "scoped": cards(s)}


def load_questions():
    items, seen = [], set()
    for r in csv.DictReader(open(ROOT / "evaluation/data/labeling.final.csv", encoding="utf-8-sig")):
        if r["task"] == "semantic" and r["query"] not in seen:
            seen.add(r["query"]); items.append({"set": "A", "query": r["query"], "gold_video": r["gold_video_id"]})
    for q in json.load(open(ROOT / "experiments/retrieval-r1-2026-11-15/questions.json")):
        if q["query"] not in seen:
            seen.add(q["query"]); items.append({"set": "B", "query": q["query"], "gold_video": q["video_id"]})
    return items


def main():
    items = load_questions()
    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(collect, items))
    videos = sorted({r["video_id"] for row in rows for r in row["global"]} | {i["gold_video"] for i in items})
    events = {}
    for v in videos:
        d = json.load(urllib.request.urlopen(f"{BASE}/suggestions?video_id={v}", timeout=60))
        events[v] = {"frames": d.get("indexed_frame_count"), "events": d.get("behavior_events", [])}
    OUT.write_text(json.dumps({"rows": rows, "videos": events}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(rows), "질문,", len(videos), "영상 →", OUT)


main()
