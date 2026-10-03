"""Video indexing, event extraction, and search pipeline.

Keep the original ``pipeline.<module>`` imports working after moving their
implementations into subpackages. Aliases are loaded only when requested, so
importing ``pipeline`` does not initialize models or external services.
"""

import importlib
import importlib.util
import sys


_LEGACY_MODULES = {
    "behavior_event_extractor": "events.behavior_event_extractor",
    "clip_embedder": "models.clip_embedder",
    "frame_extractor": "video.frame_extractor",
    "gcs_uploader": "storage.gcs_uploader",
    "metrics": "observability.metrics",
    "motion_detector": "video.motion_detector",
    "query_analyzer": "search.query_analyzer",
    "query_suggester": "search.query_suggester",
    "question_generator": "search.question_generator",
    "rag_chain": "search.rag_chain",
    "scene_event_extractor": "events.scene_event_extractor",
    "vector_store": "storage.vector_store",
    "yolo_detector": "models.yolo_detector",
}


class _LegacyModuleFinder:
    def find_spec(self, fullname, path=None, target=None):
        prefix = __name__ + "."
        if not fullname.startswith(prefix) or fullname[len(prefix) :] not in _LEGACY_MODULES:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        legacy_name = spec.name[len(__name__) + 1 :]
        return importlib.import_module("." + _LEGACY_MODULES[legacy_name], __name__)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _LegacyModuleFinder())
