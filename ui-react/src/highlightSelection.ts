/** Pick representative moments without removing events from the full timeline. */
type HighlightEvent = {
  video_id?: string;
  action?: string;
  summary?: string;
  target_object?: string;
  interestingness?: number;
  confidence?: number;
};

type HighlightItem = { event: HighlightEvent; recording: { videoId?: string } };

// These labels only group highlights; they never establish a search match.
const actions: [string, RegExp][] = [
  ["nursing", /젖.{0,8}먹|수유/],
  ["drink", /마시|마심|마신|마셔|음수|물[을\s]*먹/],
  ["food", /(?:사료|음식|먹이|밥|간식).{0,16}(?:먹|섭취)|식사/],
  ["bath", /목욕|씻고|씻는|씻음/],
  ["shake", /몸.{0,5}(?:털|터는)/],
  ["sleep", /잠을 자|잠자|낮잠|자는 중/],
];

function text(event: HighlightEvent) {
  return `${event.action ?? ""} ${event.summary ?? ""}`;
}

function phase(event: HighlightEvent) {
  const value = text(event);
  if (/않|못|없|하려|려고|추정|것 같/.test(value)) return "uncertain";
  if (/멈추|멈춘|멈췄|멈춤|중단/.test(value)) return "stop";
  return "observed";
}

function actionKey(event: HighlightEvent) {
  const value = text(event);
  return actions.find(([, pattern]) => pattern.test(value))?.[0];
}

function grams(value: string) {
  const normalized = value
    .replace(/반려동물|강아지|고양이|것으로 보임|것으로 보인다|보여요/g, "")
    .replace(/[^가-힣a-z0-9]/gi, "");
  return new Set(Array.from({ length: Math.max(0, normalized.length - 1) }, (_, i) => normalized.slice(i, i + 2)));
}

function sameAction(a: HighlightEvent, b: HighlightEvent) {
  if (phase(a) !== phase(b)) return false;
  const leftKey = actionKey(a);
  const rightKey = actionKey(b);
  if (leftKey || rightKey) return Boolean(leftKey && leftKey === rightKey);
  // For unclassified actions, keep distinct objects and require close wording.
  if (a.target_object && b.target_object && a.target_object !== b.target_object) return false;
  const left = grams(a.action || a.summary || "");
  const right = grams(b.action || b.summary || "");
  if (!left.size || !right.size) return false;
  const common = [...left].filter(value => right.has(value)).length;
  return common / (left.size + right.size - common) >= 0.65;
}

export function selectHighlights<T extends HighlightItem>(items: T[], limit = 5): T[] {
  const score = (item: T) => item.event.interestingness ?? item.event.confidence ?? 0.75;
  const selected: T[] = [];
  for (const item of [...items].sort((a, b) => score(b) - score(a))) {
    if (selected.length >= limit) break;
    const video = item.event.video_id || item.recording.videoId;
    if (video && selected.some(previous =>
      (previous.event.video_id || previous.recording.videoId) === video && sameAction(previous.event, item.event))) continue;
    selected.push(item);
  }
  return selected;
}
