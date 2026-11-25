// 실행: node --test tests/eventGroups.test.mjs  (Node 23.6 이상은 .ts를 바로 불러온다)
import test from "node:test";
import assert from "node:assert/strict";
import { actionSimilarity, groupBehaviorEvents } from "../src/eventGroups.ts";

const ev = (id, chunk, start, end, action, extra = {}) => ({ id, chunk, start_time: start, end_time: end, action, video_id: "v", ...extra });
const known = (event) => ({ chunkKey: `c${event.chunk}`, chunkStartSeconds: event.chunk * 60 });
const unknown = (event) => ({ chunkKey: `c${event.chunk}` });
const look = "반려동물이 테이블 위에서 앞쪽을 바라봄";
const lookMove = "반려동물이 테이블 위에서 앞쪽을 바라보며 움직임";
const ids = (groups) => groups.map((g) => g.members.map((m) => m.id));

test("similarity", () => {
  assert.equal(actionSimilarity(look, look), 1);
  assert.equal(actionSimilarity("abc", "xyz"), 0);
  assert.equal(actionSimilarity("", look), 0);
});

test("overlapping and consecutive similar events become one group with a video time range", () => {
  const groups = groupBehaviorEvents([ev(1, 19, 0, 12, look), ev(2, 19, 10, 18, lookMove), ev(3, 19, 20, 22, look)], known);
  assert.deepEqual(ids(groups), [[1, 2, 3]]);
  assert.equal(groups[0].videoStart, 1140);
  assert.equal(groups[0].videoEnd, 1162);
});

test("groups continue across a chunk boundary only when chunk positions are known", () => {
  const events = [ev(1, 18, 50, 58, look), ev(2, 19, 0, 12, look)];
  assert.deepEqual(ids(groupBehaviorEvents(events, known)), [[1, 2]]);
  assert.deepEqual(ids(groupBehaviorEvents(events, unknown)), [[1], [2]]);
});

test("different action, long gap, different video or object are not grouped", () => {
  assert.equal(groupBehaviorEvents([ev(1, 0, 0, 4, look), ev(2, 0, 4, 8, "반려동물이 화면 오른쪽으로 뛰어감")], known).length, 2);
  assert.equal(groupBehaviorEvents([ev(1, 0, 0, 4, look), ev(2, 0, 9, 12, look)], known).length, 2);
  assert.equal(groupBehaviorEvents([ev(1, 0, 0, 4, look), ev(2, 0, 4, 8, look, { video_id: "w" })], known).length, 2);
  assert.equal(groupBehaviorEvents([ev(1, 0, 0, 4, look), ev(2, 0, 4, 8, look, { target_object: "소파" })], known).length, 2);
});

test("an unrelated event in between does not break a group", () => {
  const groups = groupBehaviorEvents([ev(1, 0, 0, 10, look), ev(2, 0, 4, 6, "소파 위의 담요가 바닥으로 떨어짐"), ev(3, 0, 11, 14, look)], known);
  assert.deepEqual(ids(groups), [[1, 3], [2]]);
});

test("events without position or time stay separate", () => {
  const events = [ev(1, 0, 0, 4, look), ev(2, 0, 0, 4, look)];
  assert.equal(groupBehaviorEvents(events, () => ({})).length, 2);
  assert.equal(groupBehaviorEvents([{ id: 1, action: look }, { id: 2, action: look }], known).length, 2);
});

test("representative has the highest score and incoming order is preserved", () => {
  const events = [
    ev(9, 5, 0, 4, "반려동물이 화면 오른쪽으로 뛰어감", { interestingness: 0.9 }),
    ev(2, 0, 4, 8, look, { interestingness: 0.8 }),
    ev(1, 0, 0, 6, look, { interestingness: 0.3 }),
  ];
  const groups = groupBehaviorEvents(events, known);
  assert.deepEqual(ids(groups), [[9], [1, 2]]);
  assert.equal(groups[1].representative.id, 2);
});

test("input is not mutated", () => {
  const events = [ev(1, 0, 0, 4, look), ev(2, 0, 2, 6, look)];
  const copy = structuredClone(events);
  groupBehaviorEvents(events, known);
  assert.deepEqual(events, copy);
});
