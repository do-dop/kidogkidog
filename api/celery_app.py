import os

from celery import Celery


celery_app = Celery(
    "tasks",
    broker=os.getenv("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672/"),
)
