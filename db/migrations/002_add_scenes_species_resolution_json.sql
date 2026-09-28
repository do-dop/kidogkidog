-- Apply once to an existing MySQL database before enabling the CLIP species
-- resolver. This migration artifact is not executed by application startup.
ALTER TABLE scenes
    ADD COLUMN species_resolution_json JSON NULL AFTER object_labels;
