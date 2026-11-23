"""真实 OCR 引擎用例（ocr_real 标记，默认跳过）

需要已安装 easyocr / paddleocr 并能加载模型。运行：
    set GUI_AGENT_REAL_OCR=1
    pytest -m ocr_real -v
用 PIL 在白底图上画出已知文字和位置，检查引擎能否识别出文字、bbox 是否覆盖真实位置。
"""
import os

import pytest
from PIL import Image, ImageDraw, ImageFont

pytestmark = pytest.mark.ocr_real

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]
WORDS = [("设置", (100, 80)), ("搜索", (400, 80)), ("文件", (100, 300)), ("Settings", (400, 300))]


def _font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    pytest.skip("找不到中文字体")


@pytest.fixture(scope="module", params=[(800, 600, 36), (1920, 1080, 28), (2560, 1440, 22)],
                ids=["800x600", "1080p", "1440p-small-font"])
def synthetic(request, tmp_path_factory):
    w, h, size = request.param
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font = _font(size)
    boxes = {}
    for text, (x, y) in WORDS:
        d.text((x, y), text, fill="black", font=font)
        boxes[text] = d.textbbox((x, y), text, font=font)
    p = tmp_path_factory.mktemp("ocr") / f"syn_{w}x{h}.png"
    img.save(p)
    return str(p), boxes


def _overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, (b[2] - b[0]) * (b[3] - b[1]))


@pytest.mark.multi_resolution
@pytest.mark.parametrize("engine", ["easyocr", "paddleocr"])
def test_real_engine_finds_all_words(synthetic, engine):
    from perception.ocr_detector import OCRDetector
    path, boxes = synthetic
    results = OCRDetector(engine=engine).detect(path)
    found = " ".join(r["text"] for r in results)
    missing = [t for t in boxes if t not in found]
    assert not missing, f"{engine} 未识别出 {missing}，实际: {found}"


@pytest.mark.multi_resolution
@pytest.mark.parametrize("engine", ["easyocr", "paddleocr"])
def test_real_engine_bbox_covers_text(synthetic, engine):
    from perception.ocr_detector import OCRDetector
    path, boxes = synthetic
    results = OCRDetector(engine=engine).detect(path)
    for text, truth in boxes.items():
        hits = [r for r in results if text in r["text"]]
        assert hits and max(_overlap(r["bbox"], truth) for r in hits) > 0.6, text
