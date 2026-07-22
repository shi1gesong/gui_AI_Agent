"""
屏幕文字识别与UI元素定位模块
支持 EasyOCR 与 PaddleOCR 两种引擎，可对比精度与速度后选定主用方案
输出统一格式：[{"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
"""
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class OCRDetector:
    def __init__(self, engine: Literal["easyocr", "paddleocr"] = "easyocr", lang: str = "ch"):
        self.engine_name = engine
        if engine == "easyocr":
            import easyocr
            lang_list = ["ch_sim", "en"] if lang == "ch" else ["en"]
            self.engine = easyocr.Reader(lang_list)
        elif engine == "paddleocr":
            from paddleocr import PaddleOCR
            self.engine = PaddleOCR(
                lang="ch" if lang == "ch" else "en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
        else:
            raise ValueError(f"不支持的引擎: {engine}")

    def detect(self, image_path: str) -> list[dict]:
        """对图片跑OCR，返回统一格式的识别结果列表"""
        results = []
        if self.engine_name == "easyocr":
            raw = self.engine.readtext(image_path)
            for bbox_points, text, conf in raw:
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                results.append({
                    "text": text,
                    "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "confidence": float(conf),
                })
        elif self.engine_name == "paddleocr":
            raw = self.engine.predict(image_path)
            for page_result in raw:
                # PaddleOCR 3.x predict() 返回的结果对象支持类字典访问
                texts = page_result.get("rec_texts", [])
                scores = page_result.get("rec_scores", [])
                boxes = page_result.get("rec_boxes", [])
                for text, conf, box in zip(texts, scores, boxes):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    results.append({
                        "text": text,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(conf),
                    })
        return results

    def draw_boxes(self, image_path: str, results: list[dict], save_path: str = None,
                    conf_threshold: float = 0.3) -> Image.Image:
        """在原图上绘制识别出的文字边界框，用于可视化检查识别效果"""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        kept = 0
        for item in results:
            if item["confidence"] < conf_threshold:
                continue
            x1, y1, x2, y2 = item["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            kept += 1

        if save_path:
            img.save(save_path)
            print(f"标注图已保存: {save_path}，共绘制 {kept} 个边界框（置信度阈值 {conf_threshold}）")

        return img


def benchmark_engines(image_path: str):
    """对比 EasyOCR 与 PaddleOCR 在同一张图上的速度与识别数量"""
    report = {}
    for engine_name in ["easyocr", "paddleocr"]:
        try:
            t0 = time.time()
            detector = OCRDetector(engine=engine_name)
            init_time = time.time() - t0

            t0 = time.time()
            results = detector.detect(image_path)
            infer_time = time.time() - t0

            avg_conf = sum(r["confidence"] for r in results) / len(results) if results else 0
            report[engine_name] = {
                "init_time_sec": round(init_time, 2),
                "infer_time_sec": round(infer_time, 2),
                "detected_count": len(results),
                "avg_confidence": round(avg_conf, 3),
            }
        except Exception as e:
            report[engine_name] = {"error": str(e)}

    print("\n===== OCR引擎对比结果 =====")
    for engine_name, stats in report.items():
        print(f"{engine_name}: {stats}")
    return report


def find_latest_screenshot(screenshots_dir: str = "./screenshots") -> str | None:
    """自动查找screenshots目录下最新生成的截图文件（按修改时间排序）"""
    folder = Path(screenshots_dir)
    if not folder.exists():
        return None
    images = list(folder.glob("*.png"))
    if not images:
        return None
    latest = max(images, key=lambda f: f.stat().st_mtime)
    return str(latest)


if __name__ == "__main__":
    test_image = find_latest_screenshot()
    if test_image is None:
        print("未在 ./screenshots 目录下找到任何截图，请先运行 screenshot.py 生成截图")
    else:
        print(f"使用最新截图: {test_image}")
        # 单元测试1：EasyOCR 识别
        detector = OCRDetector(engine="easyocr")
        results = detector.detect(test_image)
        assert len(results) > 0, "OCR未识别到任何文字，请检查图片内容"
        print(f"✅ EasyOCR识别测试通过，共识别 {len(results)} 处文字")

        # 单元测试2：绘制边界框
        detector.draw_boxes(test_image, results, save_path="./screenshots/annotated_easyocr.png")
        print("✅ 边界框绘制测试通过")

        # 单元测试3：两引擎速度/精度对比（可选，PaddleOCR未装可跳过）
        try:
            benchmark_engines(test_image)
        except Exception as e:
            print(f"引擎对比跳过（PaddleOCR可能未装好）: {e}")
