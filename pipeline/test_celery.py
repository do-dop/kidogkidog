from celery import Celery

app = Celery('tasks', broker='amqp://guest:guest@localhost:5672/')

@app.task
def process_chunk(chunk_path):
    print(f"Processing: {chunk_path}")
    return f"Done: {chunk_path}"