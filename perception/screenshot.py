"""
屏幕截图模块
支持：全屏截图、指定区域截图、多显示器截图
依赖：mss（高性能截图，优先使用）、pyautogui（备用方案）
"""
import time
from pathlib import Path

import mss
import mss.tools
from PIL import Image


class ScreenCapture:
    def __init__(self, save_dir: str = "./screenshots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sct = mss.mss()

    def list_monitors(self):
        """列出所有显示器信息，monitors[0]是所有屏幕拼接的虚拟大屏，monitors[1..]是每个物理屏幕"""
        return self.sct.monitors

    def capture_full_screen(self, monitor_index: int = 1, save: bool = True) -> Image.Image:
        """
        截取指定显示器的全屏
        monitor_index: 1 = 主屏幕，2 = 第二块屏幕，以此类推。0 = 所有屏幕拼接
        """
        monitor = self.sct.monitors[monitor_index]
        screenshot = self.sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        if save:
            filename = self.save_dir / f"screenshot_{int(time.time())}.png"
            img.save(filename)
            print(f"截图已保存: {filename}, 尺寸: {img.size}")

        return img

    def capture_region(self, left: int, top: int, width: int, height: int, save: bool = True) -> Image.Image:
        """截取屏幕指定矩形区域，坐标为屏幕像素坐标"""
        region = {"left": left, "top": top, "width": width, "height": height}
        screenshot = self.sct.grab(region)
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        if save:
            filename = self.save_dir / f"region_{int(time.time())}.png"
            img.save(filename)
            print(f"区域截图已保存: {filename}, 尺寸: {img.size}")

        return img


if __name__ == "__main__":
    # 单元测试1：列出显示器信息
    capturer = ScreenCapture()
    monitors = capturer.list_monitors()
    print(f"检测到 {len(monitors) - 1} 个物理显示器")
    for i, m in enumerate(monitors):
        print(f"  monitor[{i}]: {m}")

    # 单元测试2：全屏截图
    img_full = capturer.capture_full_screen(monitor_index=1)
    assert img_full.size[0] > 0 and img_full.size[1] > 0, "全屏截图尺寸异常"
    print("✅ 全屏截图测试通过")

    # 单元测试3：区域截图（截取左上角 800x600 区域）
    img_region = capturer.capture_region(left=0, top=0, width=800, height=600)
    assert img_region.size == (800, 600), f"区域截图尺寸不符，实际: {img_region.size}"
    print("✅ 区域截图测试通过")
