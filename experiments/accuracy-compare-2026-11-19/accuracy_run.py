"""보지 않은 클립 6개에서 프레임 방식과 영상 직접 방식을 각 2회 실행해 원자료를 만든다. 판정은 별도 단계에서 가려서 한다."""
import contextlib, io, json, os, sys, time
from pathlib import Path
REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2]); WORK = Path(sys.argv[3])
sys.path.insert(0, str(REPO))
from dotenv import load_dotenv
load_dotenv("/Users/doyammu/Desktop/kidogkidog/.env")
from pipeline.video.frame_extractor import extract_frames
from pipeline.events.behavior_event_extractor import extract_behavior_events
from pipeline.events import video_event_extractor as vee
CLIPS = ["smartbench_0037","smartbench_0142","smartbench_0518","smartbench_0608","smartbench_0642","smartbench_0949"]
OUT.mkdir(parents=True, exist_ok=True)
def slim(events):
    return [{k: e.get(k) for k in ("start_time","end_time","action","summary","evidence")} for e in events]
for clip in CLIPS:
    src = Path.home() / "kidogkidog-clips" / f"{clip}.mp4"
    with contextlib.redirect_stdout(io.StringIO()):
        frames = extract_frames(str(src), output_dir=str(WORK / clip), frame_prefix=clip)
    frames = [{**f, "video_id": clip, "object_labels": []} for f in frames]
    for rep in (1, 2):
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            fe = extract_behavior_events(indexed_frames=frames, video_id=clip, limit=20)
        ft = time.time() - t; t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            ve = vee.extract_video_events(str(src), frames, clip, 20)
        vt = time.time() - t
        json.dump({"frames": slim(fe), "frames_seconds": round(ft,1), "video": None if ve is None else slim(ve), "video_seconds": round(vt,1)},
                  open(OUT / f"{clip}_r{rep}.json", "w"), ensure_ascii=False, indent=1)
        print(clip, rep, "프레임", len(fe), "영상", None if ve is None else len(ve), "초", round(ft), round(vt), flush=True)
