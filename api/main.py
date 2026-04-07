from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
def upload():
    # 영상 청크 수신 (나중에 구현)
    return {"status": "received"}

@app.post("/query")
def query():
    # 자연어 쿼리 수신 (나중에 구현)
    return {"status": "query received"}