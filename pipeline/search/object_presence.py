"""Find playable frames for simple object-visibility questions."""

import re

from pipeline.storage.vector_store import get_indexed_frames


_SUBJECTS = {"고양이": "cat", "강아지": "dog", "사람": "person"}
_FRAME_CHUNK = re.compile(r"_(\d{3})_frame_[0-9.]+\.jpg$")
_ACTION_WORDS = re.compile(r"먹|마시|올라|들어|나가|뛰|달리|다가|가까|이동|움직|떨어|숙이|바라|앞발|몇\s*(?:번|회)|횟수")
_STATIC_QUESTION = re.compile(r"있어|있니|있나요|있는|보여|보이|나와|나오|장면")


def _presence_subject(query):
    compact = re.sub(r"[\s?!.,]+", "", query or "")
    for name, label in _SUBJECTS.items():
        pattern = (
            rf"^{name}(?:가|이)?(?:"
            r"(?:보이는|나오는|등장하는|있는)(?:장면(?:이)?)?"
            r"(?:보여줘|보여주세요|찾아줘|찾아주세요|있어|있나요)?"
            r"|있어|있니|있나요|나와|나오니|보여|보이나요)"
            r"$"
        )
        if re.fullmatch(pattern, compact):
            return name, label
    return None


def _video_second(frame):
    timestamp = float(frame.get("timestamp") or 0)
    match = _FRAME_CHUNK.search(str(frame.get("s3_key") or ""))
    return (int(match.group(1)) * 60 if match else 0) + timestamp


def find_presence_scenes(query, video_id, top_k, recording_date=None, time_range=None):
    """Use exact indexed labels only when the question asks mere visibility."""
    subject = _presence_subject(query)
    if not subject or not video_id:
        return None

    name, label = subject
    try:
        frames = get_indexed_frames(video_id=video_id, recording_date=recording_date, time_range=time_range)
    except Exception as exc:
        print(f"객체 감지 프레임 조회 실패: {exc}", flush=True)
        return None

    matches = [
        frame for frame in frames
        if label in {part.strip().lower() for part in str(frame.get("object_labels") or "").split(",")}
        and frame.get("s3_key")
        and frame.get("timestamp") is not None
    ]
    matches.sort(key=_video_second)
    scenes = []
    for frame in matches:
        position = _video_second(frame)
        if any(abs(position - _video_second(chosen)) < 4 for chosen in scenes):
            continue
        scenes.append(frame)
        if len(scenes) >= max(1, top_k):
            break

    if not scenes:
        return None

    results = []
    for frame in scenes:
        timestamp = float(frame["timestamp"])
        results.append({
            "frame_id": frame.get("frame_id"),
            "frame_path": frame.get("frame_path"),
            "s3_key": frame["s3_key"],
            "timestamp": timestamp,
            "representative_timestamp": timestamp,
            "video_id": video_id,
            "object_labels": frame.get("object_labels"),
            "playback_start": max(0.0, timestamp - 3.0),
            "playback_end": timestamp + 5.0,
            "match_level": "object_label",
            "score": 0.0,
        })
    return {
        "query": query,
        "video_id": video_id,
        "answer": f"프레임에서 {name}로 감지된 장면 후보를 찾았어요. 아래 영상에서 확인해 주세요.",
        "results": results,
        "evidence_items": results,
        "behavior_events": [],
        "match_level": "object_label",
        "used_llm": False,
    }


def find_qualified_presence_candidates(query, video_id, top_k, recording_date=None, time_range=None):
    """Show the detected pet without claiming an unverified place or attribute."""
    if not video_id or _presence_subject(query) or not _STATIC_QUESTION.search(query) or _ACTION_WORDS.search(query):
        return None
    subjects = [name for name in _SUBJECTS if name in query]
    if len(subjects) != 1:
        return None
    name = subjects[0]
    result = find_presence_scenes(
        f"{name}가 보이는 장면 보여줘", video_id, top_k,
        recording_date=recording_date, time_range=time_range,
    )
    if result is None:
        return None
    result["query"] = query
    result["answer"] = (
        f"{name}가 감지된 영상 후보를 찾았어요. "
        "질문에 포함된 위치나 물건·상태까지는 이 감지 결과만으로 확인할 수 없으니 "
        "아래 영상을 재생해 확인해 주세요."
    )
    return result
