import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Screen = "live" | "recordings" | "search" | "chunkPlayback" | "searchPlayback" | "profile";
type Theme = "dark" | "light";
type TimeFilter = "all" | "오전" | "오후" | "저녁" | "야간";
type SuggestionStatus = "idle" | "loading" | "ready" | "fallback";

type Tag = {
  icon: string;
  label: string;
};

type Recording = {
  id: string;
  time: string;
  duration: string;
  period: Exclude<TimeFilter, "all">;
  thumb: number;
  motion: "낮음" | "보통" | "높음";
  tags: Tag[];
  note?: string;
  recordingDate?: string;
  recordedAt?: string;
  startSeconds?: number;
  playbackStartSeconds?: number;
  videoId?: string;
  videoUrl?: string;
  thumbnailUrl?: string;
};

type SearchResult = {
  id: string;
  time: string;
  duration: string;
  score: number;
  note: string;
  thumb: number;
  objects: Tag[];
  startSeconds?: number;
  seekStartSeconds?: number;
  seekEndSeconds?: number;
  videoId?: string;
  videoUrl?: string;
  thumbnailUrl?: string;
  s3Key?: string;
  framePath?: string;
  eventStart?: number;
  eventEnd?: number;
  representativeTimestamp?: number;
};

type ApiQueryResult = {
  answer?: string;
  results?: Array<{
    video_id?: string;
    timestamp?: number;
    score?: number;
    object_labels?: string;
    frame_id?: string;
    frame_path?: string;
    s3_key?: string;
    source_event_id?: number | string;
    event_start?: number;
    event_end?: number;
    representative_timestamp?: number;
    playback_start?: number;
    playback_end?: number;
  }>;
  evidence_items?: ApiQueryResult["results"];
  behavior_events?: BehaviorEvent[];
  used_llm?: boolean;
};

type ApiChunk = {
  id: string;
  s3_key: string;
  video_id: string;
  filename: string;
  chunk_index: number;
  start_seconds: number;
  duration_seconds: number;
  recorded_at?: string | null;
  recording_date?: string | null;
  time_label?: string | null;
  period: Exclude<TimeFilter, "all">;
  size: number;
  last_modified?: string | null;
  url: string;
  media_path?: string;
  thumbnail_url?: string | null;
  thumbnail_media_path?: string | null;
};

type ApiChunksResult = {
  count: number;
  chunks: ApiChunk[];
};

type BehaviorFrame = {
  s3_key?: string;
  timestamp?: number;
};

type BehaviorEvent = {
  id: number | string;
  video_id?: string;
  start_time?: number;
  end_time?: number;
  subject?: string;
  action?: string;
  target_object?: string;
  summary?: string;
  duration?: number;
  repeat_count?: number;
  confidence?: number;
  interestingness?: number;
  evidence?: unknown[];
  source_frames?: BehaviorFrame[];
  demo?: boolean;
};

type ApiBehaviorEventsResult = {
  count: number;
  events: BehaviorEvent[];
};

type UserQuery = {
  query: string;
  count?: number;
  created_at?: string;
  last_searched_at?: string;
};

type ApiSuggestionsResult = {
  questions?: string[];
  question_sources?: SuggestedQuestion[];
  behavior_events?: BehaviorEvent[];
  indexed_frame_count?: number;
  top_queries?: UserQuery[];
  recent_queries?: UserQuery[];
};

type SuggestedQuestion = {
  question: string;
  source_event_id?: number | string | null;
  event_start?: number | null;
  event_end?: number | null;
};

const defaultPetName = "코코";
const defaultUserName = "도";
const defaultPetAvatar = "🐕";
const petAvatarOptions = ["🐕", "🐈", "🐦", "🐰", "🐹", "🐢", "🐠", "🦜"];
const defaultBehaviorQuestions = new Set([
  "가장 오래 이어진 행동은 무엇인가요?",
  "같은 행동이 반복된 구간이 있나요?",
  "특정 물체 근처에 오래 머문 장면이 있나요?",
  "움직임이 많았던 장면은 언제였나요?",
  "확인해볼 만한 행동이 있었나요?",
]);

const gradients = [
  "linear-gradient(135deg,#3a2e24,#241c16)",
  "linear-gradient(140deg,#2f3a31,#1b241d)",
  "linear-gradient(135deg,#3a3522,#241f14)",
  "linear-gradient(160deg,#342a3c,#1d1826)",
  "linear-gradient(135deg,#3b2824,#241614)",
  "linear-gradient(150deg,#2a343c,#161f24)",
];

const recordings: Recording[] = [
  { id: "demo-0705", time: "07:05", duration: "5:00", period: "오전", thumb: 3, motion: "낮음", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "🛏️", label: "bed" }] },
  { id: "demo-0830", time: "08:30", duration: "5:00", period: "오전", thumb: 0, motion: "보통", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "🥣", label: "bowl" }] },
  { id: "demo-0910", time: "09:10", duration: "5:00", period: "오전", thumb: 1, motion: "높음", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "💧", label: "water" }] },
  { id: "demo-1230", time: "12:30", duration: "5:00", period: "오후", thumb: 2, motion: "보통", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "🥣", label: "bowl" }] },
  { id: "demo-1345", time: "13:45", duration: "5:00", period: "오후", thumb: 0, motion: "높음", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "🧍", label: "person" }, { icon: "🛋️", label: "couch" }] },
  { id: "demo-1420", time: "14:20", duration: "5:00", period: "오후", thumb: 4, motion: "높음", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "🥣", label: "bowl" }] },
  { id: "demo-1740", time: "17:40", duration: "5:00", period: "저녁", thumb: 1, motion: "보통", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "🚪", label: "door" }] },
  { id: "demo-1945", time: "19:45", duration: "5:00", period: "저녁", thumb: 4, motion: "높음", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "🧍", label: "person" }, { icon: "🥣", label: "bowl" }] },
  { id: "demo-2130", time: "21:30", duration: "5:00", period: "야간", thumb: 5, motion: "낮음", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "🛋️", label: "couch" }] },
  { id: "demo-2310", time: "23:10", duration: "5:00", period: "야간", thumb: 3, motion: "낮음", recordingDate: "2026-06-16", tags: [{ icon: "🐶", label: "dog" }, { icon: "🛏️", label: "bed" }] },
];

const liveEvents = [
  { icon: "pets", text: "소파 근처에 있어요", time: "방금", tags: ["dog", "couch"] },
  { icon: "restaurant", text: "밥그릇 앞에서 식사 중이에요", time: "2분 전", tags: ["dog", "bowl"] },
  { icon: "directions_walk", text: "거실을 돌아다니고 있어요", time: "8분 전", tags: ["dog", "floor"] },
  { icon: "bedtime", text: "방석 위에서 쉬고 있어요", time: "15분 전", tags: ["dog", "bed"] },
];

function getOrCreateUserId() {
  const key = "kidogkidog_user_id";
  const stored = window.localStorage.getItem(key);
  if (stored) return stored;

  const nextId = `react-${crypto.randomUUID()}`;
  window.localStorage.setItem(key, nextId);
  return nextId;
}

function Icon({ children, filled = false }: { children: string; filled?: boolean }) {
  return (
    <span className="material-symbols-rounded icon" style={{ fontVariationSettings: filled ? "'FILL' 1" : "'FILL' 0" }}>
      {children}
    </span>
  );
}

function tagList(tags: Tag[]) {
  return tags.map((tag) => (
    <span className="tag" key={`${tag.icon}-${tag.label}`}>
      <span>{tag.icon}</span>
      {tag.label}
    </span>
  ));
}

function timeToPercent(time: string) {
  const [hour, minute] = time.split(":").map(Number);
  return ((hour + minute / 60) / 24) * 100;
}

function formatSeconds(seconds: number) {
  const safeSeconds = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minute = Math.floor(safeSeconds / 60);
  const second = Math.floor(safeSeconds % 60);
  return `${String(minute).padStart(2, "0")}:${String(second).padStart(2, "0")}`;
}

function formatDuration(seconds: number) {
  const safeSeconds = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minute = Math.floor(safeSeconds / 60);
  const second = Math.floor(safeSeconds % 60);
  return `${minute}:${String(second).padStart(2, "0")}`;
}

function durationToSeconds(duration?: string) {
  if (!duration) return undefined;

  const [minute, second = "0"] = duration.split(":");
  const totalSeconds = Number(minute) * 60 + Number(second);
  return Number.isFinite(totalSeconds) ? totalSeconds : undefined;
}

function addSecondsToClockTime(time: string, seconds: number) {
  const [hour, minute] = time.split(":").map(Number);
  if (!Number.isFinite(hour) || !Number.isFinite(minute)) return formatSeconds(seconds);

  const totalSeconds = (((hour * 60 + minute) * 60) + Math.max(0, Math.floor(seconds))) % (24 * 60 * 60);
  const nextHour = Math.floor(totalSeconds / 3600);
  const nextMinute = Math.floor((totalSeconds % 3600) / 60);
  const nextSecond = totalSeconds % 60;

  const base = `${String(nextHour).padStart(2, "0")}:${String(nextMinute).padStart(2, "0")}`;
  return nextSecond > 0 ? `${base}:${String(nextSecond).padStart(2, "0")}` : base;
}

function formatHour(hour: number) {
  return `${String(hour).padStart(2, "0")}:00`;
}

function formatHourRange(startHour: number, endHour: number) {
  if (startHour === 0 && endHour === 24) return "전체 시간";
  return `${formatHour(startHour)} - ${formatHour(endHour)}`;
}

function labelsToTags(labels?: string): Tag[] {
  if (!labels) return [{ icon: "🐶", label: "scene" }];
  return labels
    .split(",")
    .map((label) => label.trim())
    .filter(Boolean)
    .slice(0, 3)
    .map((label) => ({ icon: label.toLowerCase().includes("dog") ? "🐶" : "•", label }));
}

function findRecordingForFrame(result: NonNullable<ApiQueryResult["results"]>[number], items: Recording[]) {
  const s3Key = result.s3_key || "";
  const frameId = result.frame_id || "";

  const directMatch = items.find((recording) => {
    const stem = chunkStem(recording);
    return Boolean(stem && (s3Key.includes(stem) || frameId.includes(stem)));
  });

  if (directMatch) return directMatch;

  const seconds = Number(result.timestamp ?? 0);
  return items.find((recording) => {
    if (!result.video_id || recording.videoId !== result.video_id) return false;
    const start = recording.startSeconds ?? 0;
    const end = start + 60;
    return seconds >= start && seconds < end;
  });
}

function mapApiResults(payload: ApiQueryResult, items: Recording[] = []): SearchResult[] {
  const results = payload.evidence_items ?? payload.results ?? [];
  return results.map((result, index) => {
    const seconds = Number(result.representative_timestamp ?? result.timestamp ?? 0);
    const playbackStart = Number(result.playback_start ?? seconds);
    const playbackEnd = result.playback_end !== undefined ? Number(result.playback_end) : undefined;
    const recording = findRecordingForFrame(result, items);
    const frameThumbnailUrl = result.s3_key ? mediaUrl(`/media/s3?key=${encodeURIComponent(result.s3_key)}`) : undefined;
    const eventStart = result.event_start;
    const eventEnd = result.event_end;
    const noteParts = [];

    if (playbackEnd !== undefined) {
      noteParts.push(`재생 구간: ${formatSeconds(playbackStart)}~${formatSeconds(playbackEnd)}`);
    }

    return {
      id: result.frame_id ?? `api-result-${index}`,
      time: formatSeconds(seconds),
      duration: "0:10",
      score: result.score ?? 0,
      note: noteParts.join(" · ") || (result.object_labels ? `감지 객체: ${result.object_labels}` : "검색어와 유사한 장면"),
      thumb: index % gradients.length,
      objects: labelsToTags(result.object_labels),
      startSeconds: seconds,
      seekStartSeconds: playbackStart,
      seekEndSeconds: playbackEnd,
      videoId: result.video_id,
      videoUrl: recording?.videoUrl,
      thumbnailUrl: frameThumbnailUrl || recording?.thumbnailUrl,
      s3Key: result.s3_key,
      framePath: result.frame_path,
      eventStart,
      eventEnd,
      representativeTimestamp: seconds,
    };
  });
}

function pickRelevantResults(items: SearchResult[]) {
  const sortedItems = [...items].sort((a, b) => b.score - a.score).slice(0, 3);
  const topScore = sortedItems[0]?.score;

  if (topScore === undefined) return [];

  const closeScoreCutoff = Math.max(topScore - 0.03, topScore * 0.9);
  const relevantItems = sortedItems.filter((item) => item.score >= closeScoreCutoff);

  return relevantItems.length > 0 ? relevantItems : sortedItems.slice(0, 1);
}

function _suggestionSourceMap(items: SuggestedQuestion[], visibleQuestions: string[]) {
  const visibleSet = new Set(visibleQuestions);
  const sourceMap: Record<string, SuggestedQuestion> = {};

  for (const item of items) {
    if (!item.question || !visibleSet.has(item.question)) continue;
    sourceMap[item.question] = item;
  }

  return sourceMap;
}

function mediaUrl(mediaPath?: string | null, fallbackUrl?: string | null) {
  if (mediaPath) return mediaPath.startsWith("/api/") ? mediaPath : `/api${mediaPath}`;
  return fallbackUrl || undefined;
}

function dateKey(year: number, month: number, day: number) {
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function dateParts(dateISO?: string) {
  if (!dateISO) return null;
  const [year, month, day] = dateISO.split("-").map(Number);
  if (!year || !month || !day) return null;
  return { year, month, day };
}

function getWeekDays(year: number, month: number, day: number) {
  const selectedDate = new Date(year, month - 1, day);
  const mondayOffset = selectedDate.getDay() === 0 ? -6 : 1 - selectedDate.getDay();
  const formatter = new Intl.DateTimeFormat("ko-KR", { weekday: "short" });

  return Array.from({ length: 7 }, (_, index) => {
    const date = new Date(selectedDate);
    date.setDate(selectedDate.getDate() + mondayOffset + index);

    return {
      year: date.getFullYear(),
      month: date.getMonth() + 1,
      day: date.getDate(),
      weekday: formatter.format(date).replace("요일", ""),
    };
  });
}

function mapApiChunks(payload: ApiChunksResult): Recording[] {
  return (payload.chunks ?? [])
    .filter((chunk) => {
      const identity = `${chunk.video_id} ${chunk.filename} ${chunk.s3_key}`.toLowerCase();
      return !identity.includes("cat5min");
    })
    .map((chunk, index) => ({
    id: chunk.id,
    time: chunk.time_label || formatSeconds(chunk.start_seconds),
    duration: formatDuration(chunk.duration_seconds),
    period: chunk.period,
    thumb: index % gradients.length,
    motion: "보통",
    tags: [
      { icon: "🎬", label: chunk.video_id },
      { icon: "☁️", label: "S3 chunk" },
    ],
    note: `${chunk.video_id} · ${chunk.filename}`,
    recordingDate: chunk.recording_date || undefined,
    recordedAt: chunk.recorded_at || undefined,
    startSeconds: chunk.start_seconds,
    videoId: chunk.video_id,
    videoUrl: mediaUrl(chunk.media_path, chunk.url),
    thumbnailUrl: mediaUrl(chunk.thumbnail_media_path, chunk.thumbnail_url),
  }));
}

function chunkStem(recording: Recording) {
  return recording.id.split("/").at(-1)?.replace(/\.mp4$/i, "") || recording.id;
}

function behaviorTitle(event: BehaviorEvent) {
  const action = event.action || "행동";
  const target = event.target_object ? ` · ${event.target_object}` : "";
  return `${action}${target}`;
}

function behaviorTimeLabel(event: BehaviorEvent, recording?: Recording) {
  const eventStart = event.start_time ?? 0;

  if (!recording) return formatSeconds(eventStart);

  const offset = Math.max(0, eventStart - (recording.startSeconds ?? 0));
  const eventEnd = event.end_time ?? eventStart + (event.duration ?? 0);
  const clipDuration = durationToSeconds(recording.duration);
  const endOffset = Math.max(offset, eventEnd - (recording.startSeconds ?? 0));
  const displayEndOffset = clipDuration === undefined ? endOffset : Math.min(endOffset, clipDuration);
  const startLabel = addSecondsToClockTime(recording.time, offset);
  const endLabel = addSecondsToClockTime(recording.time, displayEndOffset);

  return startLabel === endLabel ? startLabel : `${startLabel}-${endLabel}`;
}

function behaviorVideoTimeLabel(event: BehaviorEvent) {
  const startTime = event.start_time;
  const endTime = event.end_time;

  if (startTime !== undefined && endTime !== undefined) {
    return `${Number(startTime).toFixed(1)}초 ~ ${Number(endTime).toFixed(1)}초`;
  }

  if (startTime !== undefined) {
    return `${Number(startTime).toFixed(1)}초`;
  }

  return "시간 정보 없음";
}

function eventMatchesRecording(event: BehaviorEvent, recording: Recording) {
  const recordingStem = chunkStem(recording);
  const frameHit = event.source_frames?.some((frame) => frame.s3_key?.includes(recordingStem));

  if (frameHit) return true;

  if (event.video_id && recording.videoId && event.video_id !== recording.videoId) return false;

  const recordingStart = recording.startSeconds ?? 0;
  const recordingEnd = recordingStart + 60;
  const eventStart = event.start_time ?? 0;
  const eventEnd = event.end_time ?? eventStart + (event.duration ?? 10);

  return eventStart < recordingEnd && eventEnd >= recordingStart;
}

function eventsForRecording(recording: Recording, events: BehaviorEvent[]) {
  return events.filter((event) => eventMatchesRecording(event, recording));
}

function eventsForVideo(videoId: string | undefined, events: BehaviorEvent[]) {
  if (!videoId) return [];
  return events.filter((event) => event.video_id === videoId);
}

function scoreBehavior(event: BehaviorEvent) {
  return Math.round(((event.interestingness ?? event.confidence ?? 0.75) * 100));
}

export default function App() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [screen, setScreen] = useState<Screen>("live");
  const [userId] = useState(() => getOrCreateUserId());
  const [userName, setUserName] = useState(() => window.localStorage.getItem("kidogkidog_user_name") || defaultUserName);
  const [petName, setPetName] = useState(() => window.localStorage.getItem("kidogkidog_pet_name") || defaultPetName);
  const [notificationBehavior, setNotificationBehavior] = useState(() => window.localStorage.getItem("kidogkidog_notification_behavior") || "");
  const [profileUserNameDraft, setProfileUserNameDraft] = useState(userName);
  const [profilePetNameDraft, setProfilePetNameDraft] = useState(petName);
  const [profileNotificationDraft, setProfileNotificationDraft] = useState(notificationBehavior);
  const [profileNotice, setProfileNotice] = useState("");
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [hasUnreadNotification, setHasUnreadNotification] = useState(true);
  const [petAvatar, setPetAvatar] = useState(() => window.localStorage.getItem("kidogkidog_pet_avatar") || defaultPetAvatar);
  const [avatarPickerOpen, setAvatarPickerOpen] = useState(false);
  const [editingPetName, setEditingPetName] = useState(false);
  const [petNameDraft, setPetNameDraft] = useState(petName);
  const [selectedYear, setSelectedYear] = useState(2026);
  const [selectedMonth, setSelectedMonth] = useState(6);
  const [selectedDay, setSelectedDay] = useState(16);
  const [timeFilter, setTimeFilter] = useState<TimeFilter>("all");
  const [searchStartHour, setSearchStartHour] = useState(0);
  const [searchEndHour, setSearchEndHour] = useState(24);
  const [selectedSearchVideoId, setSelectedSearchVideoId] = useState("");
  const previousSearchVideoId = useRef("");
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionSources, setSuggestionSources] = useState<Record<string, SuggestedQuestion>>({});
  const [activeSuggestionSource, setActiveSuggestionSource] = useState<SuggestedQuestion | null>(null);
  const [suggestionStatus, setSuggestionStatus] = useState<SuggestionStatus>("idle");
  const [suggestionNotice, setSuggestionNotice] = useState("");
  const [indexedFrameCount, setIndexedFrameCount] = useState<number | null>(null);
  const [topQueries, setTopQueries] = useState<UserQuery[]>([]);
  const [recordingsLoading, setRecordingsLoading] = useState(false);
  const [recordingsNotice, setRecordingsNotice] = useState("");
  const [reindexing, setReindexing] = useState(false);
  const [behaviorRefreshToken, setBehaviorRefreshToken] = useState(0);
  const [suggestionRefreshToken, setSuggestionRefreshToken] = useState(0);
  const [s3Recordings, setS3Recordings] = useState<Recording[]>([]);
  const [behaviorEvents, setBehaviorEvents] = useState<BehaviorEvent[]>([]);
  const [behaviorNotice, setBehaviorNotice] = useState("");
  const [apiNotice, setApiNotice] = useState("");
  const [activeResult, setActiveResult] = useState<SearchResult | null>(null);
  const [activeRecording, setActiveRecording] = useState<Recording>(recordings[0]);

  useEffect(() => {
    let ignore = false;

    async function loadChunks() {
      setRecordingsLoading(true);
      setRecordingsNotice("");

      try {
        const response = await fetch("/api/recordings/chunks");

        if (!response.ok) throw new Error(`API ${response.status}`);

        const payload = (await response.json()) as ApiChunksResult;
        const nextRecordings = mapApiChunks(payload);

        if (!ignore) {
          setS3Recordings(nextRecordings);
          const latestDate = nextRecordings
            .map((recording) => recording.recordingDate)
            .filter(Boolean)
            .sort()
            .at(-1);

          if (latestDate) {
            const parts = dateParts(latestDate);
            if (parts) {
              setSelectedYear(parts.year);
              setSelectedMonth(parts.month);
              setSelectedDay(parts.day);
            }
          }

          setRecordingsNotice(
            nextRecordings.length > 0
              ? `S3 청크 ${nextRecordings.length}개를 불러왔습니다.`
              : "S3 chunks/ 경로에 표시할 청크 영상이 없습니다.",
          );
        }
      } catch {
        if (!ignore) {
          setS3Recordings([]);
          setRecordingsNotice("S3 청크 목록 API에 연결하지 못해 데모 녹화 목록을 표시합니다.");
        }
      } finally {
        if (!ignore) setRecordingsLoading(false);
      }
    }

    loadChunks();

    return () => {
      ignore = true;
    };
  }, []);

  const recordingItems = s3Recordings.length > 0 ? s3Recordings : recordings;
  const searchVideoOptions = useMemo(
    () => Array.from(new Set(recordingItems.map((recording) => recording.videoId).filter(Boolean))) as string[],
    [recordingItems],
  );
  const selectedDate = dateKey(selectedYear, selectedMonth, selectedDay);

  useEffect(() => {
    if (selectedSearchVideoId || s3Recordings.length === 0) return;

    const latestRecording = [...s3Recordings]
      .filter((recording) => recording.videoId)
      .sort((a, b) => {
        const aTime = a.recordedAt || `${a.recordingDate || ""}T${a.time}`;
        const bTime = b.recordedAt || `${b.recordingDate || ""}T${b.time}`;
        return aTime.localeCompare(bTime);
      })
      .at(-1);

    if (latestRecording?.videoId) {
      setSelectedSearchVideoId(latestRecording.videoId);
    }
  }, [s3Recordings, selectedSearchVideoId]);

  useEffect(() => {
    let ignore = false;
    let retryTimer: number | undefined;

    async function loadSuggestions() {
      if (s3Recordings.length > 0 && searchVideoOptions.length > 0 && !selectedSearchVideoId) {
        setSuggestions([]);
        setSuggestionSources({});
        setTopQueries([]);
        setSuggestionStatus("loading");
        setSuggestionNotice("영상 청크를 연결한 뒤 추천 질문을 생성합니다.");
        return;
      }

      setSuggestionStatus("loading");
      setSuggestionNotice("");
      setIndexedFrameCount(null);

      const params = new URLSearchParams({
        user_id: userId,
        limit: "3",
      });

      if (selectedSearchVideoId) {
        params.set("video_id", selectedSearchVideoId);
      }

      try {
        const response = await fetch(`/api/suggestions?${params.toString()}`);

        if (!response.ok) throw new Error(`API ${response.status}`);

        const payload = (await response.json()) as ApiSuggestionsResult;
        const nextSuggestions = payload.questions ?? [];
        const generatedSuggestions = nextSuggestions.filter((question) => !defaultBehaviorQuestions.has(question));

        if (!ignore) {
          setIndexedFrameCount(payload.indexed_frame_count ?? null);
          setTopQueries(payload.top_queries ?? []);

          if (payload.behavior_events?.length) {
            setBehaviorEvents((current) => {
              const merged = [...payload.behavior_events!, ...current];
              const seen = new Set<string>();
              return merged.filter((event) => {
                const key = String(event.id);
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
              });
            });
          }

          if (generatedSuggestions.length) {
            setSuggestions(generatedSuggestions);
            setSuggestionSources(_suggestionSourceMap(payload.question_sources ?? [], generatedSuggestions));
            setSuggestionStatus("ready");
            setSuggestionNotice("분석된 행동 이벤트에서 추천 질문을 생성했습니다.");
          } else {
            setSuggestions([]);
            setSuggestionSources({});
            setSuggestionStatus("loading");
            setSuggestionNotice("");
            retryTimer = window.setTimeout(loadSuggestions, 5000);
          }
        }
      } catch {
        if (!ignore) {
          setSuggestions([]);
          setSuggestionSources({});
          setSuggestionStatus("fallback");
          setTopQueries([]);
          setIndexedFrameCount(null);
          setSuggestionNotice("추천 질문 API에 연결하지 못했습니다.");
        }
      }
    }

    loadSuggestions();

    return () => {
      ignore = true;
      if (retryTimer !== undefined) {
        window.clearTimeout(retryTimer);
      }
    };
  }, [s3Recordings.length, searchVideoOptions.length, selectedSearchVideoId, userId, suggestionRefreshToken]);

  useEffect(() => {
    if (previousSearchVideoId.current === selectedSearchVideoId) return;

    if (previousSearchVideoId.current) {
      setQuery("");
      setSubmitted(false);
      setAnswer("");
      setResults([]);
      setActiveResult(null);
      setActiveSuggestionSource(null);
      setApiNotice("");
    }

    previousSearchVideoId.current = selectedSearchVideoId;
  }, [selectedSearchVideoId]);

  useEffect(() => {
    let ignore = false;

    async function loadBehaviorEvents() {
      try {
        const params = new URLSearchParams({ limit: "5" });
        if (selectedSearchVideoId) {
          params.set("video_id", selectedSearchVideoId);
        }

        const response = await fetch(`/api/recordings/events?${params.toString()}`);

        if (!response.ok) throw new Error(`API ${response.status}`);

        const payload = (await response.json()) as ApiBehaviorEventsResult;
        const nextEvents = payload.events ?? [];

        if (!ignore) {
          setBehaviorEvents(nextEvents.length > 0 ? nextEvents : []);
          setBehaviorNotice(nextEvents.length > 0 ? "DB 메타데이터에서 주요 행동을 불러왔습니다." : "DB에 표시할 주요 행동 이벤트가 아직 없습니다.");
        }
      } catch {
        if (!ignore) {
          setBehaviorEvents([]);
          setBehaviorNotice("행동 이벤트 API에 연결하지 못했습니다.");
        }
      }
    }

    loadBehaviorEvents();

    return () => {
      ignore = true;
    };
  }, [recordingItems, selectedSearchVideoId, behaviorRefreshToken]);

  const filteredRecordings = useMemo(
    () => recordingItems.filter((recording) => {
      const matchesDate = !recording.recordingDate || recording.recordingDate === selectedDate;
      const matchesTime = timeFilter === "all" || recording.period === timeFilter;
      return matchesDate && matchesTime;
    }),
    [recordingItems, selectedDate, timeFilter],
  );

  const highlightedBehaviors = useMemo(
    () => behaviorEvents
      .map((event) => ({
        event,
        recording: recordingItems.find((recording) => eventMatchesRecording(event, recording)),
      }))
      .filter((item): item is { event: BehaviorEvent; recording: Recording } => Boolean(item.recording))
      .sort((a, b) => scoreBehavior(b.event) - scoreBehavior(a.event))
      .slice(0, 5),
    [behaviorEvents, recordingItems],
  );

  const activeRecordingEvents = useMemo(
    () => eventsForRecording(activeRecording, behaviorEvents),
    [activeRecording, behaviorEvents],
  );
  const activeVideoEvents = useMemo(
    () => eventsForVideo(activeRecording.videoId, behaviorEvents),
    [activeRecording.videoId, behaviorEvents],
  );

  const visibleResults = results;
  const currentVideoRecordings = recordingItems.filter(
    (recording) => recording.videoId && activeRecording.videoId
      ? recording.videoId === activeRecording.videoId
      : true,
  );

  const topTitle = {
    live: ["실시간 라이브", `${petName}의 거실을 실시간으로 지켜보고 있어요`],
    recordings: ["녹화 영상", "S3에 업로드된 1분 단위 청크를 둘러보세요"],
    search: ["AI 검색", "자연어로 물어보면 관련 장면을 찾아드려요"],
    chunkPlayback: ["녹화 영상 재생", "S3 청크를 원본 흐름대로 확인합니다"],
    searchPlayback: ["검색 결과 재생", "AI가 찾은 장면과 근거를 검토합니다"],
    profile: ["마이페이지", "사용자와 알림 설정을 관리합니다"],
  }[screen];

  function go(nextScreen: Screen) {
    if (nextScreen === "profile") {
      setProfileUserNameDraft(userName);
      setProfilePetNameDraft(petName);
      setProfileNotificationDraft(notificationBehavior);
      setProfileNotice("");
    }

    setScreen(nextScreen);
  }

  function openResult(result: SearchResult) {
    setActiveResult(result);
    setScreen("searchPlayback");
  }

  function openRecording(recording: Recording, event?: BehaviorEvent) {
    const playbackStartSeconds = event
      ? Math.max(0, (event.start_time ?? 0) - (recording.startSeconds ?? 0))
      : undefined;

    setActiveRecording({
      ...recording,
      playbackStartSeconds,
    });
    setScreen("chunkPlayback");
  }

  async function runSearch(searchText: string) {
    const nextQuery = searchText.trim();
    if (!nextQuery) return;

    setQuery(nextQuery);
    setSubmitted(true);
    setLoading(true);
    setApiNotice("");
    setAnswer("");
    setResults([]);
    setActiveResult(null);
    setActiveSuggestionSource(null);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: nextQuery,
          video_id: selectedSearchVideoId || undefined,
          top_k: 3,
          user_id: userId,
          source_event_id: activeSuggestionSource?.source_event_id ?? undefined,
          event_start: activeSuggestionSource?.event_start ?? undefined,
          event_end: activeSuggestionSource?.event_end ?? undefined,
          time_range: {
            start_hour: searchStartHour,
            end_hour: searchEndHour,
          },
        }),
      });

      if (!response.ok) throw new Error(`API ${response.status}`);

      const payload = (await response.json()) as ApiQueryResult;
      const nextResults = pickRelevantResults(mapApiResults(payload, recordingItems));
      setAnswer(payload.answer || `${nextQuery}와 관련된 장면을 찾았어요.`);
      setResults(nextResults);
      setActiveResult(nextResults[0] ?? null);

      if (payload.behavior_events?.length) {
        setBehaviorEvents((current) => {
          const merged = [...payload.behavior_events!, ...current];
          const seen = new Set<string>();
          return merged.filter((event) => {
            const key = String(event.id);
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          });
        });
      }
    } catch {
      setAnswer(`"${nextQuery}" 검색 중 문제가 발생했습니다. 백엔드와 인덱스 상태를 확인해주세요.`);
      setResults([]);
      setActiveResult(null);
      setApiNotice("FastAPI 서버 또는 검색 인덱스에 연결하지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }

  async function submitSearch(event?: FormEvent) {
    event?.preventDefault();
    await runSearch(query);
  }

  function applySuggestion(text: string) {
    setQuery(text);
    setActiveSuggestionSource(suggestionSources[text] ?? null);
    setSubmitted(false);
    setAnswer("");
    setResults([]);
    setActiveResult(null);
    setApiNotice("");
  }

  async function reindexLocalFrames() {
    setReindexing(true);
    setRecordingsNotice("로컬 프레임 인덱스를 갱신하는 중입니다.");
    setSuggestions([]);
    setSuggestionSources({});
    setSuggestionStatus("loading");
    setSuggestionNotice("");

    const params = new URLSearchParams();
    if (selectedSearchVideoId) {
      params.set("video_id", selectedSearchVideoId);
    }

    try {
      const response = await fetch(`/api/frames/reindex${params.toString() ? `?${params.toString()}` : ""}`, {
        method: "POST",
      });

      if (!response.ok) throw new Error(`API ${response.status}`);

      setRecordingsNotice("로컬 프레임 인덱스 갱신이 완료되었습니다.");
      setBehaviorRefreshToken((value) => value + 1);
      setSuggestionRefreshToken((value) => value + 1);
    } catch {
      setRecordingsNotice("로컬 프레임 인덱스 갱신에 실패했습니다. 프레임 폴더와 백엔드를 확인해주세요.");
    } finally {
      setReindexing(false);
    }
  }

  async function removeTopQuery(queryToRemove: string) {
    const previousQueries = topQueries;
    setTopQueries((items) => items.filter((item) => item.query !== queryToRemove));

    try {
      const response = await fetch(`/api/users/${encodeURIComponent(userId)}/frequent-queries`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryToRemove }),
      });

      if (!response.ok) throw new Error(`API ${response.status}`);
    } catch {
      setTopQueries(previousQueries);
      setSuggestionNotice("자주 찾는 검색어를 삭제하지 못했습니다.");
    }
  }

  function startPetNameEdit() {
    setAvatarPickerOpen(false);
    setPetNameDraft(petName);
    setEditingPetName(true);
  }

  function savePetName() {
    const nextName = petNameDraft.trim() || defaultPetName;
    setPetName(nextName);
    window.localStorage.setItem("kidogkidog_pet_name", nextName);
    setEditingPetName(false);
  }

  function saveProfile() {
    const nextUserName = profileUserNameDraft.trim() || defaultUserName;
    const nextPetName = profilePetNameDraft.trim() || defaultPetName;

    setUserName(nextUserName);
    setPetName(nextPetName);
    setPetNameDraft(nextPetName);
    setNotificationBehavior(profileNotificationDraft);
    setHasUnreadNotification(true);

    window.localStorage.setItem("kidogkidog_user_name", nextUserName);
    window.localStorage.setItem("kidogkidog_pet_name", nextPetName);
    window.localStorage.setItem("kidogkidog_notification_behavior", profileNotificationDraft);

    setProfileUserNameDraft(nextUserName);
    setProfilePetNameDraft(nextPetName);
    setProfileNotice("저장되었습니다.");
  }

  function cancelPetNameEdit() {
    setPetNameDraft(petName);
    setEditingPetName(false);
  }

  function selectPetAvatar(nextAvatar: string) {
    setPetAvatar(nextAvatar);
    window.localStorage.setItem("kidogkidog_pet_avatar", nextAvatar);
    setAvatarPickerOpen(false);
  }

  function stepResult(delta: number) {
    if (!activeResult || visibleResults.length === 0) return;

    const index = Math.max(0, visibleResults.findIndex((item) => item.id === activeResult.id));
    const nextIndex = Math.min(visibleResults.length - 1, Math.max(0, index + delta));
    setActiveResult(visibleResults[nextIndex]);
  }

  function stepRecording(delta: number) {
    const source = currentVideoRecordings.length > 0 ? currentVideoRecordings : recordingItems;
    const index = Math.max(0, source.findIndex((item) => item.id === activeRecording.id));
    const nextIndex = Math.min(source.length - 1, Math.max(0, index + delta));
    setActiveRecording(source[nextIndex]);
  }

  return (
    <div className="app-shell" data-theme={theme}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Icon filled>pets</Icon>
          </div>
          <div>
            <strong>kidog<span>kidog</span></strong>
            <p>AI 펫캠 검색</p>
          </div>
        </div>

        <p className="nav-label">메뉴</p>
        {[
          { key: "live" as const, icon: "sensors", label: "라이브" },
          { key: "recordings" as const, icon: "grid_view", label: "녹화 영상" },
          { key: "search" as const, icon: "search", label: "AI 검색" },
        ].map((item) => {
          const activeScreen = screen === "chunkPlayback" ? "recordings" : screen === "searchPlayback" ? "search" : screen;
          const active = activeScreen === item.key;
          return (
            <button className={active ? "nav-item active" : "nav-item"} key={item.key} onClick={() => go(item.key)}>
              <Icon filled={active}>{item.icon}</Icon>
              {item.label}
            </button>
          );
        })}

        <div className={editingPetName ? "pet-status editing" : "pet-status"} onClick={() => {
          if (!editingPetName) startPetNameEdit();
        }}>
          <div className="pet-avatar-wrap" onClick={(event) => event.stopPropagation()}>
            <button
              className={avatarPickerOpen ? "pet-avatar active" : "pet-avatar"}
              type="button"
              aria-label="펫 아이콘 변경"
              onClick={() => setAvatarPickerOpen((open) => !open)}
            >
              {petAvatar}
            </button>
            {avatarPickerOpen && (
              <div className="pet-avatar-picker">
                {petAvatarOptions.map((avatar) => (
                  <button
                    className={avatar === petAvatar ? "active" : ""}
                    type="button"
                    key={avatar}
                    onClick={() => selectPetAvatar(avatar)}
                    aria-label={`${avatar} 선택`}
                  >
                    {avatar}
                  </button>
                ))}
              </div>
            )}
          </div>
          {editingPetName ? (
            <div className="pet-edit" onClick={(event) => event.stopPropagation()}>
              <input
                value={petNameDraft}
                autoFocus
                maxLength={12}
                onChange={(event) => setPetNameDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") savePetName();
                  if (event.key === "Escape") cancelPetNameEdit();
                }}
              />
              <div className="pet-edit-actions">
                <button type="button" onClick={savePetName} aria-label="이름 저장"><Icon>check</Icon></button>
                <button type="button" onClick={cancelPetNameEdit} aria-label="이름 수정 취소"><Icon>close</Icon></button>
              </div>
            </div>
          ) : (
            <>
              <div className="pet-info">
                <div className="pet-name-row">
                  <strong>{petName}</strong>
                  <button className="pet-edit-button" type="button" aria-label="펫 이름 수정">
                    <Icon>edit</Icon>
                  </button>
                </div>
                <p>거실 카메라 · 온라인 <span className="status-dot" /></p>
              </div>
            </>
          )}
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h1>{topTitle[0]}</h1>
            <p>{topTitle[1]}</p>
          </div>
          <div className="top-actions">
            <button className="icon-button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="테마 전환">
              <Icon>{theme === "dark" ? "light_mode" : "dark_mode"}</Icon>
            </button>
            <div className="notification-wrap">
              <button className="icon-button" onClick={() => {
                setNotificationOpen((open) => !open);
                setHasUnreadNotification(false);
              }} aria-label="알림">
                <Icon>notifications</Icon>
                {hasUnreadNotification && <span className="notification-dot" />}
              </button>
              {notificationOpen && (
                <div className="notification-popover">
                  <div className="notification-popover-head">
                    <strong>알림</strong>
                    <button type="button" onClick={() => setNotificationOpen(false)} aria-label="알림 닫기">
                      <Icon>close</Icon>
                    </button>
                  </div>
                  <p>
                    {notificationBehavior.trim()
                      ? `알림 조건: ${notificationBehavior}`
                      : "마이페이지에서 알림받고 싶은 내용을 입력하세요!"}
                  </p>
                </div>
              )}
            </div>
            <button className="user-avatar" type="button" onClick={() => go("profile")} aria-label="마이페이지">
              {Array.from(userName.trim() || defaultUserName)[0]}
            </button>
          </div>
        </header>

        <div className="content">
          {screen === "live" && (
            <section className="live-grid">
              <div className="stack">
                <VideoPanel label="LIVE" camera="거실 카메라" time="14:32:08" />
                <div className="metric-row">
                  <Metric icon="pets" label="현재 상태" value="소파 근처에서 휴식 중" />
                  <Metric icon="schedule" label="오늘 활동 시간" value="3.2시간 · 클립 12개" />
                </div>
              </div>
              <aside className="side-panel">
                <div className="panel-title">
                  <Icon filled>auto_awesome</Icon>
                  <strong>AI 실시간 감지</strong>
                  <span className="analyzing"><span />분석 중</span>
                </div>
                <div className="moment-card">
                  <span>지금 이 순간</span>
                  <strong>{petName}가 소파 근처에 있어요</strong>
                  <div className="tag-row">
                    <span className="tag">🐶 dog</span>
                    <span className="tag">🛋️ couch</span>
                    <span className="confidence">97%</span>
                  </div>
                </div>
                <p className="section-kicker">최근 감지 기록</p>
                <div className="event-list">
                  {liveEvents.map((event) => (
                    <div className="event-item" key={event.text}>
                      <div className="event-icon"><Icon>{event.icon}</Icon></div>
                      <div>
                        <strong>{event.text}</strong>
                        <p>{event.tags.join(" · ")} <span>{event.time}</span></p>
                      </div>
                    </div>
                  ))}
                </div>
                <button className="primary-button" onClick={() => go("search")}>
                  <Icon>search</Icon>
                  지난 영상에서 장면 검색하기
                </button>
              </aside>
            </section>
          )}

          {screen === "recordings" && (
            <section className="recording-page stack">
              <DateFilter selectedYear={selectedYear} setSelectedYear={setSelectedYear} selectedMonth={selectedMonth} setSelectedMonth={setSelectedMonth} selectedDay={selectedDay} setSelectedDay={setSelectedDay} />
              <TimeChips value={timeFilter} onChange={setTimeFilter} />
              <div className="recording-toolbar">
                <p className={s3Recordings.length > 0 ? "notice success" : "notice"}>{recordingsLoading ? "S3 청크 목록을 불러오는 중입니다." : recordingsNotice}</p>
                <button className="secondary-button" onClick={reindexLocalFrames} disabled={reindexing}>
                  <Icon>{reindexing ? "progress_activity" : "sync"}</Icon>
                  {reindexing ? "갱신 중" : "로컬 프레임 인덱스 갱신"}
                </button>
              </div>
              <BehaviorHighlights
                items={highlightedBehaviors}
                notice={behaviorNotice}
                petName={petName}
                onOpen={(recording) => openRecording(recording)}
              />
              <div className="clip-grid">
                {filteredRecordings.map((clip) => (
                  <ClipCard key={clip.id} clip={clip} eventCount={eventsForRecording(clip, behaviorEvents).length} onOpen={() => openRecording(clip)} />
                ))}
              </div>
            </section>
          )}

          {screen === "search" && (
            <section className="search-page">
              <aside className="search-filter-panel">
                <div className="filter-panel-title">
                  <Icon>tune</Icon>
                  <strong>검색 조건</strong>
                </div>
                <div className="filter-group">
                  <span>날짜</span>
                  <DateFilter selectedYear={selectedYear} setSelectedYear={setSelectedYear} selectedMonth={selectedMonth} setSelectedMonth={setSelectedMonth} selectedDay={selectedDay} setSelectedDay={setSelectedDay} compact />
                </div>
                <div className="filter-group">
                  <span>검색할 영상</span>
                  <select className="video-select" value={selectedSearchVideoId} onChange={(event) => setSelectedSearchVideoId(event.target.value)}>
                    <option value="">전체 영상</option>
                    {searchVideoOptions.map((videoId) => (
                      <option value={videoId} key={videoId}>{videoId}</option>
                    ))}
                  </select>
                </div>
                <div className="filter-group">
                  <span>시간대</span>
                  <TimeRangeBar
                    startHour={searchStartHour}
                    endHour={searchEndHour}
                    onChange={(start, end) => {
                      setSearchStartHour(start);
                      setSearchEndHour(end);
                    }}
                  />
                </div>
                <div className="filter-summary">
                  <span>선택 범위</span>
                  <strong>{selectedSearchVideoId || "전체 영상"} · {selectedYear}년 {selectedMonth}월 {selectedDay}일 · {formatHourRange(searchStartHour, searchEndHour)}</strong>
                </div>
              </aside>

              <div className="search-main">
                <div className="search-hero">
                  <div className="search-heading">
                    <Icon filled>auto_awesome</Icon>
                    <strong>AI 영상 검색</strong>
                  </div>
                  <span>{petName}의 하루 중 무엇이 궁금하세요?</span>
                  <form className="search-bar" onSubmit={submitSearch}>
                    <Icon>search</Icon>
                    <input
                      value={query}
                      onChange={(event) => {
                        setQuery(event.target.value);
                        setActiveSuggestionSource(null);
                      }}
                      placeholder="예: 강아지가 밥 먹는 장면 찾아줘"
                    />
                    <button className={loading ? "loading" : ""} type="submit" disabled={loading}>{loading ? "검색 중" : "검색"}</button>
                  </form>
                  <div className="suggestions">
                    <span><Icon>auto_awesome</Icon>추천</span>
                    {selectedSearchVideoId && indexedFrameCount === 0 ? (
                      <div className="suggestion-loading suggestion-message">
                        <Icon>info</Icon>
                        <strong>로컬에서 이 영상을 먼저 인덱싱해주세요</strong>
                      </div>
                    ) : suggestionStatus === "loading" ? (
                      <div className="suggestion-loading">
                        <span className="loading-mark"><Icon filled>auto_awesome</Icon></span>
                        <strong>추천질문 생성중입니다</strong>
                      </div>
                    ) : (
                      suggestions.map((item) => (
                        <button key={item} onClick={() => applySuggestion(item)}>{item}</button>
                      ))
                    )}
                  </div>
                  {suggestionNotice && <p className="suggestion-note">{suggestionNotice}</p>}
                  {topQueries.length > 0 && (
                    <div className="query-summary">
                      <span className="query-summary-label">자주 찾는 검색어</span>
                      {topQueries.map((item) => (
                        <span className="query-chip" key={item.query}>
                          <button className="query-chip-main" type="button" onClick={() => applySuggestion(item.query)}>
                            {item.query}
                          </button>
                          <button
                            className="query-chip-remove"
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              void removeTopQuery(item.query);
                            }}
                            aria-label={`${item.query} 삭제`}
                          >
                            <Icon>close</Icon>
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {submitted ? (
                  <div className="search-results stack">
                    {apiNotice && <p className="notice">{apiNotice}</p>}
                    <div className="answer-card">
                      <div><Icon filled>auto_awesome</Icon></div>
                      <div>
                        <span>AI 답변</span>
                        {loading ? (
                          <p className="answer-loading">
                            <span className="loading-mark"><Icon filled>auto_awesome</Icon></span>
                            답변 생성중입니다
                          </p>
                        ) : (
                          <p>{answer}</p>
                        )}
                      </div>
                    </div>
                    {!loading && (
                      <>
                        <div className="result-header">
                          <strong>관련 장면 {visibleResults.length}개</strong>
                          <span>유사도 순</span>
                        </div>
                        <div className="result-grid">
                          {visibleResults.map((result) => (
                            <ResultCard key={result.id} result={result} onOpen={() => openResult(result)} />
                          ))}
                        </div>
                      </>
                    )}
                  </div>
                ) : (
                  <div className="empty-state">
                    <Icon filled>auto_awesome</Icon>
                    <strong>무엇이든 자연어로 물어보세요</strong>
                    <p>"{petName}가 밥 먹는 장면"처럼 입력하면 AI가 하루 영상에서 관련 구간을 찾아드려요.</p>
                  </div>
                )}
              </div>
            </section>
          )}

          {screen === "profile" && (
            <section className="profile-page">
              <div className="profile-card">
                <div className="profile-head">
                  <div className="profile-avatar">{Array.from(userName.trim() || defaultUserName)[0]}</div>
                  <div>
                    <span>마이페이지</span>
                    <strong>{profileUserNameDraft.trim() || userName.trim() || defaultUserName}</strong>
                  </div>
                </div>

                <label className="profile-field">
                  <span>사용자 이름</span>
                  <input
                    value={profileUserNameDraft}
                    maxLength={12}
                    onChange={(event) => {
                      setProfileUserNameDraft(event.target.value);
                      setProfileNotice("");
                    }}
                    placeholder="사용자 이름"
                  />
                </label>

                <label className="profile-field">
                  <span>반려동물 이름</span>
                  <input
                    value={profilePetNameDraft}
                    maxLength={12}
                    onChange={(event) => {
                      setProfilePetNameDraft(event.target.value);
                      setProfileNotice("");
                    }}
                    placeholder="반려동물 이름"
                  />
                </label>

                <label className="profile-field">
                  <span>알림 받고 싶은 행동</span>
                  <textarea
                    value={profileNotificationDraft}
                    onChange={(event) => {
                      setProfileNotificationDraft(event.target.value);
                      setProfileNotice("");
                    }}
                    placeholder="예: 밥 먹는 장면, 물 마시는 장면, 오래 움직이지 않는 상황"
                    rows={4}
                  />
                </label>
                <div className="profile-actions">
                  {profileNotice && <span>{profileNotice}</span>}
                  <button className="primary-button" type="button" onClick={saveProfile}>
                    <Icon>save</Icon>
                    저장
                  </button>
                </div>
              </div>
            </section>
          )}

          {screen === "chunkPlayback" && (
            <section className="playback-page stack">
              <div className="playback-head">
                <button className="secondary-button" onClick={() => go("recordings")}>
                  <Icon>arrow_back</Icon>
                  녹화 목록으로
                </button>
                <strong>녹화 청크 재생</strong>
              </div>
              <div className="playback-grid">
                <div className="stack">
                  <VideoPanel
                    label={activeRecording.playbackStartSeconds !== undefined ? "행동 구간 자동 점프" : "S3 청크 재생"}
                    camera={activeRecording.note || `${activeRecording.videoId ?? "영상"} · ${activeRecording.time}`}
                    time={activeRecording.playbackStartSeconds !== undefined ? addSecondsToClockTime(activeRecording.time, activeRecording.playbackStartSeconds) : activeRecording.time}
                    videoUrl={activeRecording.videoUrl}
                    startAtSeconds={activeRecording.playbackStartSeconds}
                    wide
                  />
                  <div className="chunk-info-grid">
                    <Metric icon="movie" label="원본 영상" value={activeRecording.videoId || "알 수 없음"} />
                    <Metric icon="schedule" label="청크 구간" value={`${activeRecording.time} · ${activeRecording.duration}`} />
                  </div>
                  <div className="button-row">
                    <button className="secondary-button" onClick={() => stepRecording(-1)}><Icon>skip_previous</Icon>이전 청크</button>
                    <button className="primary-button" onClick={() => stepRecording(1)}>다음 청크<Icon>skip_next</Icon></button>
                  </div>
                </div>
                <aside className="side-panel">
                  <div className="chunk-detail-card">
                    <span>청크 정보</span>
                    <strong>{activeRecording.note || activeRecording.id}</strong>
                    <p>S3에 저장된 1분 단위 원본 청크입니다. 이 화면은 녹화 파일 탐색과 연속 재생에 집중합니다.</p>
                  </div>
                  <BehaviorEventPanel videoEvents={activeVideoEvents} chunkEvents={activeRecordingEvents} recording={activeRecording} />
                  <button className="primary-button" onClick={() => {
                    setQuery(`${activeRecording.videoId || "이 영상"} ${activeRecording.time} 근처 장면`);
                    go("search");
                  }}>
                    <Icon>search</Icon>
                    이 구간에서 AI 검색
                  </button>
                  <p className="section-kicker">같은 원본 영상의 청크 {currentVideoRecordings.length || recordingItems.length}개</p>
                  <div className="side-results">
                    {(currentVideoRecordings.length > 0 ? currentVideoRecordings : recordingItems).map((recording) => (
                      <button className={recording.id === activeRecording.id ? "side-result active" : "side-result"} key={recording.id} onClick={() => setActiveRecording(recording)}>
                        <div style={recording.thumbnailUrl ? { backgroundImage: `url(${recording.thumbnailUrl})` } : { background: gradients[recording.thumb] }} />
                        <span>{recording.time}</span>
                        <strong>{recording.note || recording.id}</strong>
                      </button>
                    ))}
                  </div>
                </aside>
              </div>
            </section>
          )}

          {screen === "searchPlayback" && activeResult && (
            <section className="playback-page stack">
              <div className="playback-head">
                <button className="secondary-button" onClick={() => go("search")}>
                  <Icon>arrow_back</Icon>
                  검색으로
                </button>
                <strong>검색 결과 재생</strong>
              </div>
              <div className="playback-grid">
                <div className="stack">
                  <VideoPanel label="검색 구간 자동 점프" camera={activeResult.note} time={activeResult.time} videoUrl={activeResult.videoUrl} startAtSeconds={activeResult.seekStartSeconds} wide />
                  <Timeline activeResult={activeResult} results={visibleResults} />
                  <div className="button-row">
                    <button className="secondary-button" onClick={() => stepResult(-1)}><Icon>skip_previous</Icon>이전 결과</button>
                    <button className="primary-button" onClick={() => stepResult(1)}>다음 결과<Icon>skip_next</Icon></button>
                  </div>
                </div>
                <aside className="side-panel">
                  <div className="answer-card compact">
                    <div><Icon filled>auto_awesome</Icon></div>
                    <div>
                      <span>AI 답변</span>
                      <p>{answer || `${activeResult.time} 구간에서 ${activeResult.note}이 확인됐어요.`}</p>
                    </div>
                  </div>
                  <p className="section-kicker">검색된 장면 {visibleResults.length}개</p>
                  <div className="side-results">
                    {visibleResults.map((result) => (
                      <button className={result.id === activeResult.id ? "side-result active" : "side-result"} key={result.id} onClick={() => setActiveResult(result)}>
                        <div style={result.thumbnailUrl ? { backgroundImage: `url(${result.thumbnailUrl})` } : { background: gradients[result.thumb] }} />
                        <span>{result.time}</span>
                        <strong>{result.note}</strong>
                      </button>
                    ))}
                  </div>
                </aside>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

function VideoPanel({
  label,
  camera,
  time,
  videoUrl,
  startAtSeconds = 0,
  wide = false,
}: {
  label: string;
  camera: string;
  time: string;
  videoUrl?: string;
  startAtSeconds?: number;
  wide?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const safeStartAtSeconds = Math.max(0, Number.isFinite(startAtSeconds) ? startAtSeconds : 0);
  const videoKey = `${videoUrl || "empty"}-${safeStartAtSeconds}`;

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;

    const seek = () => {
      if (safeStartAtSeconds <= 0) return;

      try {
        const canSeek = !Number.isFinite(video.duration) || safeStartAtSeconds < video.duration;
        if (!canSeek) return;
        video.currentTime = safeStartAtSeconds;
        void video.play().catch(() => undefined);
      } catch {
        // Keep the player visible even if the browser cannot seek this media URL.
      }
    };

    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      seek();
      return;
    }

    video.addEventListener("loadedmetadata", seek, { once: true });
    return () => video.removeEventListener("loadedmetadata", seek);
  }, [safeStartAtSeconds, videoUrl]);

  return (
    <div className={`${wide ? "video-panel wide" : "video-panel"} ${videoUrl ? "with-media" : ""}`}>
      {videoUrl ? (
        <video key={videoKey} ref={videoRef} className="video-player" src={videoUrl} controls autoPlay muted playsInline />
      ) : (
        <div className="video-gradient" />
      )}
      <div className="video-badge"><span />{label}</div>
      <div className="video-time">{time}</div>
      <div className="video-camera"><Icon>videocam</Icon>{camera}</div>
      <div className="video-controls">
        <button><Icon>volume_up</Icon></button>
        <button><Icon>fullscreen</Icon></button>
      </div>
    </div>
  );
}

function Metric({ icon, label, value }: { icon: string; label: string; value: string }) {
  return (
    <div className="metric">
      <div><Icon>{icon}</Icon></div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DateFilter({
  selectedYear,
  setSelectedYear,
  selectedMonth,
  setSelectedMonth,
  selectedDay,
  setSelectedDay,
  compact = false,
}: {
  selectedYear: number;
  setSelectedYear: (year: number) => void;
  selectedMonth: number;
  setSelectedMonth: (month: number) => void;
  selectedDay: number;
  setSelectedDay: (day: number) => void;
  compact?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const daysInMonth = new Date(selectedYear, selectedMonth, 0).getDate();
  const firstDay = new Date(selectedYear, selectedMonth - 1, 1).getDay();
  const leadingBlanks = firstDay === 0 ? 6 : firstDay - 1;
  const calendarCells = [
    ...Array.from({ length: leadingBlanks }, (_, index) => ({ key: `blank-${index}`, day: null })),
    ...Array.from({ length: daysInMonth }, (_, index) => ({ key: `day-${index + 1}`, day: index + 1 })),
  ];
  const weekDays = getWeekDays(selectedYear, selectedMonth, selectedDay);

  function selectDay(day: number) {
    setSelectedDay(day);
    setOpen(false);
  }

  function selectDate(year: number, month: number, day: number) {
    setSelectedYear(year);
    setSelectedMonth(month);
    setSelectedDay(day);
    setOpen(false);
  }

  function moveMonth(delta: number) {
    let nextYear = selectedYear;
    let nextMonth = selectedMonth + delta;

    if (nextMonth < 1) {
      nextMonth = 12;
      nextYear -= 1;
    }

    if (nextMonth > 12) {
      nextMonth = 1;
      nextYear += 1;
    }

    setSelectedYear(nextYear);
    setSelectedMonth(nextMonth);
    const nextMonthDays = new Date(nextYear, nextMonth, 0).getDate();
    if (selectedDay > nextMonthDays) setSelectedDay(nextMonthDays);
  }

  return (
    <div className={compact ? "date-filter compact" : "date-filter"}>
      <button className={open ? "date-label active" : "date-label"} type="button" onClick={() => setOpen((value) => !value)}>
        <Icon>calendar_today</Icon>
        {selectedYear}년 {selectedMonth}월 {selectedDay}일
        <Icon>{open ? "expand_less" : "expand_more"}</Icon>
      </button>
      {!compact && (
        <div className="day-row">
          {weekDays.map((date) => {
            const active = date.year === selectedYear && date.month === selectedMonth && date.day === selectedDay;
            return (
              <button className={active ? "day active" : "day"} key={`${date.year}-${date.month}-${date.day}`} type="button" onClick={() => selectDate(date.year, date.month, date.day)}>
                <span>{date.weekday}</span>
                <strong>{date.day}</strong>
              </button>
            );
          })}
        </div>
      )}
      {open && (
        <div className="calendar-popover">
          <div className="calendar-head">
            <button type="button" onClick={() => moveMonth(-1)} disabled={selectedMonth === 1} aria-label="이전 달">
              <Icon>chevron_left</Icon>
            </button>
            <strong>{selectedYear}년 {selectedMonth}월</strong>
            <button type="button" onClick={() => moveMonth(1)} disabled={selectedMonth === 12 && selectedYear >= 2026} aria-label="다음 달">
              <Icon>chevron_right</Icon>
            </button>
          </div>
          <div className="calendar-weekdays">
            {["월", "화", "수", "목", "금", "토", "일"].map((weekday) => (
              <span key={weekday}>{weekday}</span>
            ))}
          </div>
          <div className="calendar-grid">
            {calendarCells.map((cell) => (
              cell.day ? (
                <button className={cell.day === selectedDay ? "calendar-day active" : "calendar-day"} key={cell.key} type="button" onClick={() => selectDay(cell.day)}>
                  {cell.day}
                </button>
              ) : (
                <span className="calendar-blank" key={cell.key} />
              )
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function TimeChips({ value, onChange, compact = false }: { value: TimeFilter; onChange: (value: TimeFilter) => void; compact?: boolean }) {
  const chips: Array<{ value: TimeFilter; label: string }> = [
    { value: "all", label: "전체" },
    { value: "오전", label: "오전" },
    { value: "오후", label: "오후" },
    { value: "저녁", label: "저녁" },
    { value: "야간", label: "야간" },
  ];

  return (
    <div className={compact ? "time-chips compact" : "time-chips"}>
      {chips.map((chip) => (
        <button className={chip.value === value ? "chip active" : "chip"} key={chip.value} onClick={() => onChange(chip.value)}>
          {chip.label}
        </button>
      ))}
    </div>
  );
}

function TimeRangeBar({
  startHour,
  endHour,
  onChange,
}: {
  startHour: number;
  endHour: number;
  onChange: (startHour: number, endHour: number) => void;
}) {
  const startPct = (startHour / 24) * 100;
  const endPct = (endHour / 24) * 100;

  function updateStart(value: number) {
    onChange(Math.min(value, endHour - 1), endHour);
  }

  function updateEnd(value: number) {
    onChange(startHour, Math.max(value, startHour + 1));
  }

  return (
    <div className="time-range">
      <div className="time-range-labels">
        <strong>{formatHour(startHour)}</strong>
        <span>{formatHourRange(startHour, endHour)}</span>
        <strong>{formatHour(endHour)}</strong>
      </div>
      <div className="range-slider">
        <div className="range-track" />
        <div className="range-fill" style={{ left: `${startPct}%`, width: `${endPct - startPct}%` }} />
        <input
          aria-label="검색 시작 시간"
          min="0"
          max="23"
          step="1"
          type="range"
          value={startHour}
          onChange={(event) => updateStart(Number(event.target.value))}
        />
        <input
          aria-label="검색 종료 시간"
          min="1"
          max="24"
          step="1"
          type="range"
          value={endHour}
          onChange={(event) => updateEnd(Number(event.target.value))}
        />
      </div>
      <div className="time-range-scale">
        <span>00</span>
        <span>06</span>
        <span>12</span>
        <span>18</span>
        <span>24</span>
      </div>
    </div>
  );
}

function BehaviorHighlights({
  items,
  notice,
  petName,
  onOpen,
}: {
  items: Array<{ event: BehaviorEvent; recording: Recording }>;
  notice: string;
  petName: string;
  onOpen: (recording: Recording, event?: BehaviorEvent) => void;
}) {
  if (items.length === 0) {
    return null;
  }

  return (
    <section className="behavior-strip">
      <div className="behavior-strip-head">
        <div>
          <span>오늘 발견한 주요행동</span>
          <strong>눈에 띄는 구간 {items.length}개</strong>
        </div>
        <p>{notice}</p>
      </div>
      <div className="behavior-row">
        {items.map(({ event, recording }) => (
          <button className="behavior-card" key={`${event.id}-${recording.id}`} onClick={() => onOpen(recording, event)}>
            <div className="behavior-icon"><Icon filled>auto_awesome</Icon></div>
            <div>
              <span>{behaviorTimeLabel(event, recording)} · {recording.period}</span>
              <strong>{behaviorTitle(event)}</strong>
              <p>{event.summary || `${petName}의 행동 변화가 감지된 구간입니다.`}</p>
            </div>
            <span className="behavior-score">{scoreBehavior(event)}%</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function behaviorDetailItems(event: BehaviorEvent) {
  const items = [];

  if (event.action) items.push(`행동: ${event.action}`);
  if (event.target_object) items.push(`대상: ${event.target_object}`);
  if (event.duration !== undefined) items.push(`지속 시간: ${Number(event.duration).toFixed(1)}초`);
  if (event.repeat_count) items.push(`반복 후보: ${event.repeat_count}회`);
  if (event.confidence !== undefined) items.push(`신뢰도: ${Number(event.confidence).toFixed(2)}`);
  if (event.interestingness !== undefined) items.push(`흥미도: ${Number(event.interestingness).toFixed(2)}`);

  return items;
}

function behaviorEvidenceItems(event: BehaviorEvent) {
  const evidence = event.evidence ?? [];
  if (!Array.isArray(evidence)) return [];
  return evidence.map((item) => String(item)).filter(Boolean).slice(0, 3);
}

function BehaviorEventPanel({
  videoEvents,
  chunkEvents,
  recording,
}: {
  videoEvents: BehaviorEvent[];
  chunkEvents: BehaviorEvent[];
  recording: Recording;
}) {
  const summaryEvents = videoEvents.length > 0 ? videoEvents : chunkEvents;

  return (
    <div className="chunk-events">
      <div className="chunk-events-head">
        <span>이 영상에서 발견한 행동 요약</span>
        <strong>{summaryEvents.length}개</strong>
      </div>
      {summaryEvents.length > 0 ? (
        <div className="chunk-event-list">
          {summaryEvents.slice(0, 5).map((event) => {
            const details = behaviorDetailItems(event);
            const evidenceItems = behaviorEvidenceItems(event);

            return (
            <div className="chunk-event-row" key={`${event.id}-${recording.id}`}>
              <div>
                <span className="chunk-event-time">{behaviorVideoTimeLabel(event)}</span>
                <strong>{behaviorTitle(event)}</strong>
                <p>{event.summary || "행동 변화가 감지됐어요."}</p>
                {details.length > 0 && <small>{details.join(" · ")}</small>}
                {evidenceItems.length > 0 && (
                  <ul>
                    {evidenceItems.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                )}
              </div>
              <span>{scoreBehavior(event)}%</span>
            </div>
            );
          })}
        </div>
      ) : (
        <p className="chunk-event-empty">이 영상에는 아직 주요 행동 메타데이터가 없습니다.</p>
      )}
      {chunkEvents.length > 0 && videoEvents.length !== chunkEvents.length && (
        <p className="chunk-event-empty">현재 청크와 직접 겹치는 행동 후보는 {chunkEvents.length}개입니다.</p>
      )}
    </div>
  );
}

function ClipCard({ clip, eventCount = 0, onOpen }: { clip: Recording; eventCount?: number; onOpen: () => void }) {
  const motionClass = clip.motion === "높음" ? "high" : clip.motion === "보통" ? "medium" : "low";

  return (
    <button className="clip-card" onClick={onOpen}>
      <div className={clip.thumbnailUrl ? "thumb has-image" : "thumb"} style={clip.thumbnailUrl ? { backgroundImage: `url(${clip.thumbnailUrl})` } : { background: gradients[clip.thumb] }}>
        <span className="time-pill">{clip.time}</span>
        <span className="duration-pill">{clip.duration}</span>
        <span className={`motion-pill ${motionClass}`}><i />{clip.motion}</span>
        {eventCount > 0 && <span className="event-badge"><Icon filled>auto_awesome</Icon>{eventCount}</span>}
        <div className="play-circle"><Icon filled>play_arrow</Icon></div>
      </div>
      <div className="tag-row">{tagList(clip.tags)}</div>
    </button>
  );
}

function ResultCard({ result, onOpen }: { result: SearchResult; onOpen: () => void }) {
  const score = result.score.toFixed(3);

  return (
    <button className="result-card" onClick={onOpen}>
      <div className={result.thumbnailUrl ? "thumb has-image" : "thumb"} style={result.thumbnailUrl ? { backgroundImage: `url(${result.thumbnailUrl})` } : { background: gradients[result.thumb] }}>
        <span className="time-pill">{result.time}</span>
        <span className="score-pill">유사도 {score}</span>
        <div className="play-circle"><Icon filled>play_arrow</Icon></div>
      </div>
      <div className="result-body">
        <strong>{result.note}</strong>
        <div className="tag-row">{tagList(result.objects)}</div>
      </div>
    </button>
  );
}

function Timeline({ activeResult, results }: { activeResult: SearchResult; results: SearchResult[] }) {
  const timelinePercent = (result: SearchResult) => {
    if (typeof result.startSeconds === "number") {
      return (result.startSeconds / 86400) * 100;
    }

    return timeToPercent(result.time);
  };

  return (
    <div className="timeline-card">
      <p><Icon>timeline</Icon>오늘 타임라인 · 검색 결과 구간 하이라이트</p>
      <div className="timeline">
        {results.map((result) => (
          <span className="timeline-marker" key={result.id} style={{ left: `${timelinePercent(result)}%` }} />
        ))}
        <span className="playhead" style={{ left: `${timelinePercent(activeResult)}%` }} />
      </div>
      <div className="timeline-labels">
        <span>00</span>
        <span>06</span>
        <span>12</span>
        <span>18</span>
        <span>24</span>
      </div>
    </div>
  );
}
