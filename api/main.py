from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import frames, media, recordings, search, suggestions, upload
from db.schema import init_db


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


app.include_router(media.router)
app.include_router(recordings.router)
app.include_router(frames.router)
app.include_router(suggestions.router)
app.include_router(upload.router)
app.include_router(search.router)
