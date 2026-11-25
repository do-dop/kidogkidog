import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import ts from 'typescript';
const source = ts.transpileModule(readFileSync(new URL('../src/highlightSelection.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const { selectHighlights } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const item = (id, action, score = 0.8, video = 'video', target = '') => ({
  event: { id, video_id: video, action, interestingness: score, target_object: target },
  recording: { videoId: video },
});

test('same meal across distant chunks contributes only its highest scoring scene', () => {
  const rows = [item(1, '반려동물이 사료를 먹는 것으로 보임', 0.8),
    item(2, '반려동물이 음식을 섭취함', 0.95), item(3, '반려동물이 식사 중', 0.9)];
  assert.deepEqual(selectHighlights(rows).map(row => row.event.id), [2]);
});

test('food, drinking, nursing and bathing remain separate highlights', () => {
  const rows = [item(1, '사료를 먹음'), item(2, '물을 마심'), item(3, '젖을 먹음'), item(4, '목욕하는 중')];
  assert.equal(selectHighlights(rows).length, 4);
});

test('the same action from different videos remains available', () => {
  assert.equal(selectHighlights([item(1, '사료를 먹음', 0.8, 'a'), item(2, '사료를 먹음', 0.7, 'b')]).length, 2);
});

test('unknown actions use conservative wording and object comparison', () => {
  const rows = [item(1, '장난감을 바라봄', 0.9, 'v', '공'), item(2, '장난감을 바라봄', 0.8, 'v', '공'),
    item(3, '장난감을 바라봄', 0.7, 'v', '인형'), item(4, '장난감을 물고 달림', 0.6, 'v', '공')];
  assert.deepEqual(selectHighlights(rows).map(row => row.event.id), [1, 3, 4]);
});

test('negated actions and stopping are not merged with performing an action', () => {
  assert.equal(selectHighlights([item(1, '밥을 먹음'), item(2, '밥을 먹지 않음'), item(3, '밥 먹기를 멈춤')]).length, 3);
});

test('does not fill empty slots with duplicates, mutate input, or merge unknown videos', () => {
  const rows = [item(1, '사료를 먹음', 0.7), item(2, '사료를 먹음', 0.9)];
  const before = structuredClone(rows);
  assert.equal(selectHighlights(rows, 5).length, 1);
  assert.deepEqual(rows, before);
  assert.equal(selectHighlights(rows, 0).length, 0);
  assert.equal(selectHighlights([item(3, '사료를 먹음', 0.7, ''), item(4, '사료를 먹음', 0.8, '')]).length, 2);
});
