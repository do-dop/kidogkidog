/** 화면에 보여 줄 때만 쓰는 행동 이벤트 묶기.
 *
 * 12초 분석 창이 겹치고 같은 상태가 이어지면 거의 같은 이벤트가 여러 건 저장된다. 저장 데이터를 합치면
 * 구간 정보가 사라지므로 그대로 두고, 시간상 가깝고 설명이 비슷한 이벤트를 한 줄로 묶어 보여 준다.
 */

/** 앞 장면의 끝과 다음 장면의 시작이 이 간격(초) 이하일 때만 묶는다. */
export const MAX_GROUP_GAP_SECONDS = 4;
/** 행동 문장의 글자 bigram Jaccard 유사도 하한. */
export const MIN_ACTION_SIMILARITY = 0.5;

export type GroupableEvent = {
  video_id?: string;
  start_time?: number;
  end_time?: number;
  action?: string;
  target_object?: string;
  interestingness?: number;
  confidence?: number;
};

/** 이벤트가 어느 청크의 것인지, 그리고 알 수 있다면 그 청크가 영상에서 시작하는 초. */
export type EventPosition = { chunkKey?: string; chunkStartSeconds?: number };

export type EventGroup<T> = {
  /** 목록에 대표로 보여 줄 이벤트(점수가 가장 높고, 같으면 가장 긴 것). */
  representative: T;
  /** 시간순으로 정렬한 원본 이벤트. 하나뿐이면 묶이지 않은 것이다. */
  members: T[];
  /** 영상 전체 기준 시작·끝 초. 청크 위치를 모르면 없다. */
  videoStart?: number;
  videoEnd?: number;
};

function bigrams(text: string): Set<string> {
  const cleaned = text.replace(/\s+/g, "");
  const result = new Set<string>();
  for (let i = 0; i < cleaned.length - 1; i += 1) result.add(cleaned.slice(i, i + 2));
  return result;
}

export function actionSimilarity(a: string, b: string): number {
  const left = bigrams(a);
  const right = bigrams(b);
  if (left.size === 0 || right.size === 0) return 0;
  let shared = 0;
  left.forEach((item) => { if (right.has(item)) shared += 1; });
  return shared / (left.size + right.size - shared);
}

const score = (event: GroupableEvent) => event.interestingness ?? event.confidence ?? 0;
const length = (event: GroupableEvent) => (event.end_time ?? event.start_time ?? 0) - (event.start_time ?? 0);

export function groupBehaviorEvents<T extends GroupableEvent>(
  events: T[],
  locate: (event: T) => EventPosition,
): EventGroup<T>[] {
  const rows = events.map((event, index) => {
    const position = locate(event);
    const offset = position.chunkStartSeconds;
    const start = event.start_time;
    const end = event.end_time ?? start;
    // 시간축: 청크 시작 위치를 알면 영상 전체, 모르면 청크 안. 서로 다른 시간축끼리는 묶지 않는다.
    const axis = offset !== undefined ? "video" : position.chunkKey ? `chunk:${position.chunkKey}` : undefined;
    return {
      event,
      index,
      axis: start === undefined ? undefined : axis,
      from: (offset ?? 0) + (start ?? 0),
      to: (offset ?? 0) + (end ?? 0),
    };
  });

  type Row = (typeof rows)[number];
  const groups: { rows: Row[]; to: number }[] = [];

  [...rows].sort((a, b) => a.from - b.from || a.to - b.to).forEach((row) => {
    const target = row.axis === undefined ? undefined : [...groups].reverse().find((group) => {
      const first = group.rows[0];
      return first.axis === row.axis
        && first.event.video_id === row.event.video_id
        && (first.event.target_object || "") === (row.event.target_object || "")
        && row.from - group.to <= MAX_GROUP_GAP_SECONDS
        // 첫 이벤트와 비교한다. 바로 앞 것과만 비교하면 설명이 조금씩 달라지며 다른 행동까지 이어 붙는다.
        && actionSimilarity(first.event.action || "", row.event.action || "") >= MIN_ACTION_SIMILARITY;
    });
    if (target) {
      target.rows.push(row);
      target.to = Math.max(target.to, row.to);
    } else {
      groups.push({ rows: [row], to: row.to });
    }
  });

  // 받은 순서(점수순 등)를 유지한다: 묶음에서 가장 앞에 있던 이벤트의 순서를 따른다.
  return groups
    .sort((a, b) => Math.min(...a.rows.map((r) => r.index)) - Math.min(...b.rows.map((r) => r.index)))
    .map((group) => {
      const representative = group.rows.reduce((best, row) => (
        score(row.event) > score(best.event)
          || (score(row.event) === score(best.event) && length(row.event) > length(best.event)) ? row : best
      )).event;
      const onVideoAxis = group.rows[0].axis === "video";
      return {
        representative,
        members: group.rows.map((row) => row.event),
        videoStart: onVideoAxis ? group.rows[0].from : undefined,
        videoEnd: onVideoAxis ? group.to : undefined,
      };
    });
}
