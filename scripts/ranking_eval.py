"""전역 검색 순위를 CLIP 점수만 쓸 때와 보정했을 때 비교한다(오프라인, API 호출 없음).

입력: experiments/ranking-2026-11-19/data.json (scripts/ranking_collect.py).
후보 영상 = 전역 top-10 카드 + 정답 영상(전역에 없으면 영상 한정 검색 점수를 사용).
보정: score' = score - beta*ln(frames) + alpha*text   (frames: 영상의 색인 프레임 수, text: 질문-이벤트 설명 겹침 0~1)
세트 A에서 고른 설정을 세트 B(처음 보는 질문)에서 확인한다. 색인 프레임이 0인 영상이 정답인 질문은 제외한다.
"""
import importlib.util, json, math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ets", ROOT / "pipeline/search/event_text_search.py")
ets = importlib.util.module_from_spec(spec); spec.loader.exec_module(ets)

data = json.load(open(ROOT / "experiments/ranking-2026-11-19/data.json", encoding="utf-8"))
videos = data["videos"]
ENDINGS = ets._QUERY_ENDINGS

vtext = {v: ets._grams(" ".join(ets._description(e) for e in x["events"])) for v, x in videos.items()}
df = Counter(g for grams in vtext.values() for g in grams)
N = len(vtext)


def text_score(query, video):
    q = ets._grams(ENDINGS.sub("", query.strip()))
    if not q:
        return 0.0
    w = lambda g: math.log((N + 1) / (df[g] + 1)) + 1
    return sum(w(g) for g in q & vtext.get(video, set())) / sum(w(g) for g in q)


def candidates(row):
    cands = {c["video_id"]: c["score"] for c in row["global"]}
    if row["gold_video"] not in cands and row["scoped"]:
        cands[row["gold_video"]] = row["scoped"][0]["score"]
    return cands


def evaluate(rows, alpha, beta):
    hit1 = hit3 = rr = n = 0
    for row in rows:
        gold = row["gold_video"]
        if not videos.get(gold, {}).get("frames"):
            continue
        cands = candidates(row)
        adj = {v: s - beta * math.log(max(1, videos[v]["frames"])) + alpha * text_score(row["query"], v)
               for v, s in cands.items() if v in videos}
        if gold not in adj:
            n += 1; continue
        rank = 1 + sum(1 for v, s in adj.items() if s > adj[gold])
        n += 1; hit1 += rank == 1; hit3 += rank <= 3; rr += 1 / rank
    return n, hit1, hit3, rr / n if n else 0


sets = {s: [r for r in data["rows"] if r["set"] == s] for s in "AB"}
print("기준(CLIP만):", {s: evaluate(r, 0, 0) for s, r in sets.items()})
print("alpha beta | A: n  H@1 H@3 MRR | B: n  H@1 H@3 MRR")
best = []
for alpha in (0, 0.02, 0.05, 0.1, 0.2):
    for beta in (0, 0.002, 0.005, 0.01, 0.02):
        a, b = evaluate(sets["A"], alpha, beta), evaluate(sets["B"], alpha, beta)
        best.append((a[3], alpha, beta, a, b))
        print(f"{alpha:5} {beta:5} | A: {a[0]} {a[1]:3} {a[2]:3} {a[3]:.3f} | B: {b[0]} {b[1]:3} {b[2]:3} {b[3]:.3f}")
