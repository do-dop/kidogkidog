"""보지 않은 클립 6개에 현재 서비스의 프레임 기반 행동 추출(추출 → gpt-4o-mini)을 그대로 돌린다. DB·GCS 없음."""
import contextlib, io, json, os, sys, time
from pathlib import Path
REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2]); WORK = Path(sys.argv[3])
sys.path.insert(0, str(REPO))
from dotenv import load_dotenv
load_dotenv("/Users/doyammu/Desktop/kidogkidog/.env")
for k in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
    print(k, "설정됨" if os.getenv(k) else "없음")
from pipeline.video.frame_extractor import extract_frames
from pipeline.events.behavior_event_extractor import extract_behavior_events
CLIPS = ["smartbench_0037","smartbench_0142","smartbench_0518","smartbench_0608","smartbench_0642","smartbench_0949"]
OUT.mkdir(parents=True, exist_ok=True)
for clip in CLIPS:
    src = Path.home() / "kidogkidog-clips" / f"{clip}.mp4"
    fdir = WORK / clip
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        frames = extract_frames(str(src), output_dir=str(fdir), frame_prefix=clip)
    frames = [{**f, "video_id": clip, "object_labels": []} for f in frames]
    with contextlib.redirect_stdout(io.StringIO()):
        events = extract_behavior_events(indexed_frames=frames, video_id=clip, limit=20)
    rec = {"clip": clip, "frames": len(frames), "seconds": round(time.time() - t0, 1),
           "events": [{k: e.get(k) for k in ("action","start_time","end_time","subject","subject_type","confidence")} for e in events]}
    json.dump(rec, open(OUT / f"{clip}.json", "w"), ensure_ascii=False, indent=1)
    print(clip, "프레임", len(frames), "이벤트", len(events), "초", rec["seconds"], flush=True)
