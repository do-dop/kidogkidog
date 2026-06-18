from pydantic import BaseModel


class ChunkRequest(BaseModel):
    chunk_path: str | None = None
    video_path: str | None = None


class TimeRangeRequest(BaseModel):
    start_hour: int
    end_hour: int


class QueryRequest(BaseModel):
    query: str
    video_id: str | None = None
    top_k: int = 5
    user_id: str | None = None
    recording_date: str | None = None
    time_range: TimeRangeRequest | None = None


class SuggestionRequest(BaseModel):
    user_id: str | None = None
    video_id: str | None = None
    limit: int = 5


class FrequentQueryDeleteRequest(BaseModel):
    query: str
