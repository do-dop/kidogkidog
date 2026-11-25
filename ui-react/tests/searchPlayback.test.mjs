import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import ts from 'typescript';
const source = ts.transpileModule(readFileSync(new URL('../src/searchPlayback.ts', import.meta.url), 'utf8'), {compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
const {resolveSearchRecording, playbackOffsets, displayOffsets} = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const items = [{id:'chunks/wrong/petcam_20261105_100000_001.mp4',videoId:'wrong',startSeconds:60}, {id:'chunks/right/petcam_20261105_100000_001.mp4',videoId:'right',startSeconds:60}];
test('source chunk must belong to the result video', () => {
 const match = resolveSearchRecording({video_id:'right',s3_key:'frames/right/petcam_20261105_100000_001_frame_4.00.jpg',timestamp:4},items);
 assert.equal(match.recording.videoId,'right'); assert.equal(match.chunkRelative,true);
 assert.deepEqual(playbackOffsets(2,6,60,match.chunkRelative),{start:2,end:6});
});
test('global time is converted even when shorter than media duration',()=>{
 const match=resolveSearchRecording({video_id:'right',timestamp:60},items);
 assert.equal(match.chunkRelative,false);
 assert.deepEqual(playbackOffsets(60,64,60,match.chunkRelative),{start:0,end:4});
});
test('unknown source and video never select an unrelated recording',()=>{
 assert.equal(resolveSearchRecording({video_id:'missing',timestamp:4},items).recording,undefined);
});

test('missing identified chunk must not fall back to another chunk at the same local second',()=>{
 const match=resolveSearchRecording({video_id:'right',s3_key:'frames/right/petcam_20261105_100000_002_frame_60.00.jpg',timestamp:60},items);
 assert.equal(match.recording,undefined);
});

test('third-minute scene displays full-video times while seeking within its chunk',()=>{
 const recording={id:'chunks/video/video_003.mp4',videoId:'video',startSeconds:180};
 const match=resolveSearchRecording({video_id:'video',s3_key:'frames/video/video_003_frame_2.8.jpg',timestamp:2.8},[recording]);
 assert.deepEqual(displayOffsets(2.8,1,4,match.chunkStartSeconds,match.chunkRelative),{moment:182.8,start:181,end:184});
 assert.deepEqual(playbackOffsets(1,4,180,match.chunkRelative),{start:1,end:4});
});

test('global timestamps never receive the chunk offset twice',()=>{
 const match=resolveSearchRecording({video_id:'right',timestamp:62},items);
 assert.deepEqual(displayOffsets(62,61,64,match.chunkStartSeconds,match.chunkRelative),{moment:62,start:61,end:64});
 assert.deepEqual(playbackOffsets(61,64,60,match.chunkRelative),{start:1,end:4});
});

test('missing standard named chunk retains its full-video display offset',()=>{
 const match=resolveSearchRecording({video_id:'video',s3_key:'frames/video/video_003_frame_2.jpg',timestamp:2},[]);
 assert.equal(match.recording,undefined);
 assert.deepEqual(displayOffsets(2,1,4,match.chunkStartSeconds,match.chunkRelative),{moment:182,start:181,end:184});
});

