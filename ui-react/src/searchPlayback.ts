/** Resolve a source chunk before converting its timestamps. Never infer the
 * time coordinate system from the numeric size of a timestamp. */
export function resolveSearchRecording<T extends {id: string; videoId?: string; startSeconds?: number}>(
  result: {video_id?: string; s3_key?: string; frame_id?: string; timestamp?: number}, items: T[],
): {recording?: T; chunkRelative: boolean; chunkStartSeconds: number} {
  const sources = [result.s3_key, result.frame_id].filter(Boolean) as string[];
  const sourceStems = sources.map(source => source.split('/').at(-1)!.replace(/_frame_.*$/, '').replace(/\.mp4$/i, ''));
  const eligible = items.filter(item => !result.video_id || item.videoId === result.video_id);
  const direct = eligible.find(item => sourceStems.includes(item.id.split('/').at(-1)!.replace(/\.mp4$/i, '')));
  const indexedStem = sourceStems.find(stem => /_\d{3,}$/.test(stem));
  const inferredStart = indexedStem ? Number(indexedStem.match(/_(\d+)$/)![1]) * 60 : 0;
  if (direct) return {recording: direct, chunkRelative: true, chunkStartSeconds: direct.startSeconds ?? inferredStart};
  // A known chunk identity that is absent must not redirect to another chunk.
  if (indexedStem) {
    return {chunkRelative: true, chunkStartSeconds: inferredStart};
  }
  const seconds = Number(result.timestamp ?? 0);
  return {
    recording: eligible.find(item => result.video_id && seconds >= (item.startSeconds ?? 0) && seconds < (item.startSeconds ?? 0) + 60),
    chunkRelative: false,
    chunkStartSeconds: 0,
  };
}

/** Display a scene in the full source video without changing media seek offsets. */
export function displayOffsets(moment: number, start: number, end: number | undefined, chunkStart: number, chunkRelative: boolean) {
  const offset = chunkRelative ? chunkStart : 0;
  return {moment: moment + offset, start: start + offset, end: end === undefined ? undefined : end + offset};
}

export function playbackOffsets(start: number, end: number | undefined, chunkStart: number, chunkRelative: boolean) {
  const offset = chunkRelative ? 0 : chunkStart;
  return {start: Math.max(0, start - offset), end: end === undefined ? undefined : end - offset};
}
