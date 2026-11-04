import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from api.services import media_service as media


def jpeg_bytes(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


class ResizedImageTests(unittest.TestCase):
    def setUp(self):
        media._RESIZED_IMAGE_CACHE.clear()

    def test_large_frame_is_shrunk_keeping_aspect_ratio(self):
        original = jpeg_bytes(1920, 1080)
        with patch.object(media, "download_bytes", return_value=original):
            response = media.build_resized_image_response("frames/v/a.jpg", 480)

        image = Image.open(io.BytesIO(response.body))
        self.assertEqual(image.size, (480, 270))
        self.assertLess(len(response.body), len(original))
        self.assertEqual(response.media_type, "image/jpeg")
        self.assertIn("max-age", response.headers["Cache-Control"])

    def test_small_image_is_not_enlarged(self):
        with patch.object(media, "download_bytes", return_value=jpeg_bytes(200, 100)):
            response = media.build_resized_image_response("frames/v/a.jpg", 480)

        self.assertEqual(Image.open(io.BytesIO(response.body)).size, (200, 100))

    def test_width_is_clamped(self):
        with patch.object(media, "download_bytes", return_value=jpeg_bytes(3000, 1500)):
            huge = media.build_resized_image_response("frames/v/a.jpg", 100000)
            tiny = media.build_resized_image_response("frames/v/b.jpg", 1)

        self.assertEqual(Image.open(io.BytesIO(huge.body)).width, media.MAX_IMAGE_WIDTH)
        self.assertEqual(Image.open(io.BytesIO(tiny.body)).width, media.MIN_IMAGE_WIDTH)

    def test_second_request_is_served_from_cache(self):
        with patch.object(media, "download_bytes", return_value=jpeg_bytes(1920, 1080)) as download:
            media.build_resized_image_response("frames/v/a.jpg", 480)
            media.build_resized_image_response("frames/v/a.jpg", 480)

        download.assert_called_once()

    def test_unreadable_image_is_404(self):
        with patch.object(media, "download_bytes", return_value=b"not an image"):
            with self.assertRaises(HTTPException) as caught:
                media.build_resized_image_response("frames/v/a.jpg", 480)

        self.assertEqual(caught.exception.status_code, 404)

    def test_width_is_ignored_for_video(self):
        class FakeRequest:
            headers = {}

        with patch.object(media, "download_bytes", return_value=b"video-bytes"), \
                patch.object(media, "build_resized_image_response") as resized:
            response = media.build_s3_media_response("chunks/v/a.mp4", FakeRequest(), width=480)

        resized.assert_not_called()
        self.assertEqual(response.body, b"video-bytes")


if __name__ == "__main__":
    unittest.main()
