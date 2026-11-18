"""Only explicitly supplied camera timestamps count as recording times."""

from datetime import datetime


def trusted_capture_time(metadata):
    metadata = metadata or {}
    if metadata.get("source_type") != "petcam":
        return None

    value = metadata.get("captured_at")
    if not value:
        return None

    try:
        captured_at = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None

    # A timezone-free string cannot distinguish the camera's local time from UTC.
    return captured_at if captured_at.tzinfo is not None else None
