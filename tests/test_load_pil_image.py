import base64
import io

import pytest
from renderers.mm_image import load_pil_image


def _tiny_png_bytes():
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_load_pil_image_accepts_pil_path_file_and_data_uri(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "tiny.png"
    path.write_bytes(_tiny_png_bytes())
    data_uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

    for part in (
        {"type": "image", "image": Image.open(path)},
        {"type": "image", "image": str(path)},
        {"type": "image", "image": path.as_uri()},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ):
        pil = load_pil_image(part)
        assert pil.size == (8, 8)
        assert pil.mode == "RGB"


def test_load_pil_image_rejects_http_and_raw_bytes():
    pytest.importorskip("PIL.Image")

    with pytest.raises(ValueError, match="does not fetch remote images"):
        load_pil_image({"type": "image", "image": "https://example.com/a.png"})

    with pytest.raises(TypeError, match="Unsupported image source"):
        load_pil_image({"type": "image", "image": _tiny_png_bytes()})
