"""第2周 截图模块 —— Xvfb 虚拟屏幕上的真实 mss 截图（非 mock）

每个用例单独起一个指定分辨率的 Xvfb，用真实 mss 截图，验证尺寸、颜色通道、区域截图。
Windows 上无 Xvfb，自动跳过；Windows 真机截图见 tests/real/test_real_device.py。
"""
import pytest

from conftest import xvfb_available
from gui.xvfb_utils import fill_screen, tk_available, virtual_display

pytestmark = [pytest.mark.xvfb]
RESOLUTIONS = ["800x600", "1280x720", "1366x768", "1920x1080", "2560x1440", "3840x2160"]


def _capture(tmp_path):
    from perception.screenshot import ScreenCapture
    return ScreenCapture(save_dir=str(tmp_path / "shots"))


@pytest.mark.multi_resolution
@pytest.mark.parametrize("res", RESOLUTIONS)
def test_real_full_screen_size(tmp_path, res):
    w, h = map(int, res.split("x"))
    with virtual_display(res):
        img = _capture(tmp_path).capture_full_screen(save=True)
    assert img.size == (w, h)
    assert len(list((tmp_path / "shots").glob("screenshot_*.png"))) == 1


@pytest.mark.multi_resolution
@pytest.mark.parametrize("res", RESOLUTIONS)
def test_real_monitor_info_matches_resolution(tmp_path, res):
    w, h = map(int, res.split("x"))
    with virtual_display(res):
        mons = _capture(tmp_path).list_monitors()
    assert (mons[1]["width"], mons[1]["height"]) == (w, h)


@pytest.mark.multi_resolution
@pytest.mark.parametrize("res", ["800x600", "1920x1080", "3840x2160"])
def test_real_region_800x600_or_quarter(tmp_path, res):
    w, h = map(int, res.split("x"))
    rw, rh = min(800, w // 2), min(600, h // 2)
    with virtual_display(res):
        img = _capture(tmp_path).capture_region(w - rw, h - rh, rw, rh, save=False)
    assert img.size == (rw, rh)


@pytest.mark.functional
@pytest.mark.skipif(not tk_available(), reason="需要 tkinter 绘制纯色窗口")
@pytest.mark.parametrize("color,rgb", [("#ff0000", (255, 0, 0)), ("#00ff00", (0, 255, 0)), ("#0000ff", (0, 0, 255))],
                         ids=["red", "green", "blue"])
def test_real_color_channels_correct(tmp_path, color, rgb):
    """BGRX -> RGB 转换在真实截图上正确：纯红屏幕截出来必须是 (255,0,0) 而不是 (0,0,255)"""
    with virtual_display("640x480") as disp:
        proc = fill_screen(color, 640, 480, disp)
        try:
            img = _capture(tmp_path).capture_full_screen(save=False)
        finally:
            proc.terminate()
    assert img.getpixel((320, 240)) == rgb


@pytest.mark.functional
@pytest.mark.skipif(not tk_available(), reason="需要 tkinter 绘制纯色窗口")
def test_real_region_matches_full_screen_crop(tmp_path):
    with virtual_display("640x480") as disp:
        proc = fill_screen("#336699", 640, 480, disp)
        try:
            cap = _capture(tmp_path)
            full = cap.capture_full_screen(save=False)
            region = cap.capture_region(100, 50, 200, 100, save=False)
        finally:
            proc.terminate()
    assert list(region.getdata()) == list(full.crop((100, 50, 300, 150)).getdata())


@pytest.mark.boundary
def test_real_region_last_pixel(tmp_path):
    with virtual_display("1280x720"):
        img = _capture(tmp_path).capture_region(1279, 719, 1, 1, save=False)
    assert img.size == (1, 1)


@pytest.mark.exception
def test_real_region_outside_screen_raises(tmp_path):
    """区域超出屏幕时 X11 下 mss 抛 ScreenShotError；ScreenCapture 没有预先校验/裁剪（记录现状）"""
    import mss.exception
    with virtual_display("1280x720"):
        with pytest.raises(mss.exception.ScreenShotError):
            _capture(tmp_path).capture_region(1200, 700, 200, 100, save=False)


@pytest.mark.exception
def test_real_region_zero_size_raises(tmp_path):
    import mss.exception
    with virtual_display("1280x720"):
        with pytest.raises(mss.exception.ScreenShotError):
            _capture(tmp_path).capture_region(0, 0, 0, 10, save=False)
