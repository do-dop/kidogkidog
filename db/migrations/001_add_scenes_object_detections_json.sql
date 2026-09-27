-- Run once against an existing MySQL database before deploying code that writes
-- scene detections. This file is a migration artifact only; it is not executed
-- by application startup or by the evaluation task.
ALTER TABLE scenes
    ADD COLUMN object_detections_json JSON NULL AFTER object_labels;
