"""第2周 感知模块：perception/screenshot.py（mss 截图）—— mock 用例"""
import pytest
from PIL import Image

from fakes import FakeMSS

import perception.screenshot as shot_mod
from perception.screenshot import ScreenCapture

RESOLUTIONS = [
    (800, 600), (1280, 720), (1366, 768), (1600, 900),
    (1920, 1080), (2560, 1440), (3840, 2160), (1080, 1920),
]


@pytest.fixture
def make_capture(monkeypatch, tmp_path):
    def _make(screens=((1920, 1080),), bgrx=(10, 20, 30, 255), save_dir=None):
        fake = FakeMSS(screens=screens, bgrx=bgrx)
        monkeypatch.setattr(shot_mod.mss, "mss", lambda *a, **k: fake)
        cap = ScreenCapture(save_dir=str(save_dir or tmp_path / "shots"))
        return cap, fake
    return _make


@pytest.mark.functional
class TestScreenshotFunctional:
    def test_list_monitors_single_screen(self, make_capture):
        cap, _ = make_capture()
        mons = cap.list_monitors()
        assert len(mons) == 2  # [0]=虚拟总屏 + 1 块物理屏
        assert mons[1]["width"] == 1920 and mons[1]["height"] == 1080

    def test_full_screen_returns_pil_image(self, make_capture):
        cap, _ = make_capture()
        img = cap.capture_full_screen(save=False)
        assert isinstance(img, Image.Image)

    def test_full_screen_size_matches_monitor(self, make_capture):
        cap, _ = make_capture()
        assert cap.capture_full_screen(save=False).size == (1920, 1080)

    def test_full_screen_is_rgb_mode(self, make_capture):
        cap, _ = make_capture()
        assert cap.capture_full_screen(save=False).mode == "RGB"

    def test_bgrx_converted_to_rgb(self, make_capture):
        cap, _ = make_capture(bgrx=(10, 20, 30, 255))
        img = cap.capture_full_screen(save=False)
        assert img.getpixel((0, 0)) == (30, 20, 10)

    def test_full_screen_grabs_selected_monitor_dict(self, make_capture):
        cap, fake = make_capture()
        cap.capture_full_screen(monitor_index=1, save=False)
        assert fake.grabs == [fake.monitors[1]]

    def test_default_monitor_index_is_primary(self, make_capture):
        cap, fake = make_capture(screens=((1920, 1080), (1280, 1024)))
        cap.capture_full_screen(save=False)
        assert fake.grabs[-1] == fake.monitors[1]

    def test_full_screen_save_writes_png(self, make_capture, tmp_path):
        cap, _ = make_capture(save_dir=tmp_path / "out")
        cap.capture_full_screen(save=True)
        files = list((tmp_path / "out").glob("screenshot_*.png"))
        assert len(files) == 1
        assert Image.open(files[0]).size == (1920, 1080)

    def test_full_screen_no_save_writes_nothing(self, make_capture, tmp_path):
        cap, _ = make_capture(save_dir=tmp_path / "out")
        cap.capture_full_screen(save=False)
        assert list((tmp_path / "out").iterdir()) == []

    def test_region_passes_exact_region(self, make_capture):
        cap, fake = make_capture()
        cap.capture_region(10, 20, 300, 200, save=False)
        assert fake.grabs[-1] == {"left": 10, "top": 20, "width": 300, "height": 200}

    def test_region_output_size(self, make_capture):
        cap, _ = make_capture()
        assert cap.capture_region(0, 0, 800, 600, save=False).size == (800, 600)

    def test_region_save_uses_region_prefix(self, make_capture, tmp_path):
        cap, _ = make_capture(save_dir=tmp_path / "out")
        cap.capture_region(0, 0, 64, 32, save=True)
        files = list((tmp_path / "out").glob("region_*.png"))
        assert len(files) == 1 and Image.open(files[0]).size == (64, 32)

    def test_save_dir_created_including_parents(self, make_capture, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        make_capture(save_dir=target)
        assert target.is_dir()

    def test_saved_png_pixels_roundtrip(self, make_capture, tmp_path):
        cap, _ = make_capture(bgrx=(0, 0, 255, 255), save_dir=tmp_path / "o")  # 纯红
        cap.capture_region(0, 0, 4, 4, save=True)
        f = next((tmp_path / "o").glob("region_*.png"))
        assert Image.open(f).convert("RGB").getpixel((2, 2)) == (255, 0, 0)


@pytest.mark.boundary
class TestScreenshotBoundary:
    def test_region_1x1(self, make_capture):
        cap, _ = make_capture()
        assert cap.capture_region(0, 0, 1, 1, save=False).size == (1, 1)

    def test_region_bottom_right_pixel(self, make_capture):
        cap, fake = make_capture()
        cap.capture_region(1919, 1079, 1, 1, save=False)
        assert fake.grabs[-1] == {"left": 1919, "top": 1079, "width": 1, "height": 1}

    def test_region_equal_to_full_screen(self, make_capture):
        cap, _ = make_capture()
        assert cap.capture_region(0, 0, 1920, 1080, save=False).size == (1920, 1080)

    def test_existing_save_dir_is_reused(self, make_capture, tmp_path):
        d = tmp_path / "exists"
        d.mkdir()
        (d / "keep.txt").write_text("x")
        make_capture(save_dir=d)
        assert (d / "keep.txt").exists()

    def test_monitor_index_zero_is_virtual_desktop(self, make_capture):
        cap, _ = make_capture(screens=((1920, 1080), (2560, 1440)))
        assert cap.capture_full_screen(monitor_index=0, save=False).size == (4480, 1440)

    @pytest.mark.xfail(strict=True, reason="已知问题：文件名只精确到秒，同一秒内两次截图会互相覆盖")
    def test_two_saves_in_same_second_keep_both_files(self, make_capture, tmp_path, monkeypatch):
        cap, _ = make_capture(save_dir=tmp_path / "o")
        monkeypatch.setattr(shot_mod.time, "time", lambda: 1700000000.5)
        cap.capture_full_screen(save=True)
        cap.capture_full_screen(save=True)
        assert len(list((tmp_path / "o").glob("*.png"))) == 2

    @pytest.mark.xfail(strict=True, reason="已知问题：monitor_index=-1 被 Python 负索引静默接受，截到了最后一块屏而不是报错")
    def test_negative_monitor_index_rejected(self, make_capture):
        cap, _ = make_capture(screens=((1920, 1080), (1280, 1024)))
        with pytest.raises((IndexError, ValueError)):
            cap.capture_full_screen(monitor_index=-1, save=False)


@pytest.mark.exception
class TestScreenshotException:
    def test_monitor_index_out_of_range(self, make_capture):
        cap, _ = make_capture()
        with pytest.raises(IndexError):
            cap.capture_full_screen(monitor_index=5, save=False)

    def test_region_zero_width_raises(self, make_capture):
        cap, _ = make_capture()
        with pytest.raises(Exception):
            cap.capture_region(0, 0, 0, 100, save=False)

    def test_region_negative_height_raises(self, make_capture):
        cap, _ = make_capture()
        with pytest.raises(Exception):
            cap.capture_region(0, 0, 100, -5, save=False)

    def test_grab_backend_error_propagates(self, make_capture):
        cap, fake = make_capture()

        def boom(region):
            raise RuntimeError("XGetImage failed")
        fake.grab = boom
        with pytest.raises(RuntimeError, match="XGetImage"):
            cap.capture_full_screen(save=False)

    def test_save_dir_is_existing_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shot_mod.mss, "mss", lambda *a, **k: FakeMSS())
        f = tmp_path / "not_a_dir"
        f.write_text("x")
        with pytest.raises(FileExistsError):
            ScreenCapture(save_dir=str(f))

    def test_mss_init_failure_propagates(self, monkeypatch, tmp_path):
        def fail(*a, **k):
            raise OSError("no display")
        monkeypatch.setattr(shot_mod.mss, "mss", fail)
        with pytest.raises(OSError):
            ScreenCapture(save_dir=str(tmp_path))


@pytest.mark.multi_resolution
class TestScreenshotMultiResolution:
    @pytest.mark.parametrize("w,h", RESOLUTIONS, ids=[f"{w}x{h}" for w, h in RESOLUTIONS])
    def test_full_screen_size_per_resolution(self, make_capture, w, h):
        cap, _ = make_capture(screens=((w, h),))
        assert cap.capture_full_screen(save=False).size == (w, h)

    @pytest.mark.parametrize("w,h", RESOLUTIONS, ids=[f"{w}x{h}" for w, h in RESOLUTIONS])
    def test_center_quarter_region_per_resolution(self, make_capture, w, h):
        cap, fake = make_capture(screens=((w, h),))
        img = cap.capture_region(w // 4, h // 4, w // 2, h // 2, save=False)
        assert img.size == (w // 2, h // 2)
        assert fake.grabs[-1]["left"] == w // 4

    @pytest.mark.parametrize("screens", [
        ((1920, 1080), (1920, 1080)),
        ((2560, 1440), (1920, 1080)),
        ((1920, 1080), (1080, 1920)),
        ((3840, 2160), (1920, 1080), (1280, 1024)),
    ], ids=["2x1080p", "1440p+1080p", "landscape+portrait", "3-screens"])
    def test_multi_monitor_layout(self, make_capture, screens):
        cap, _ = make_capture(screens=screens)
        mons = cap.list_monitors()
        assert len(mons) == len(screens) + 1
        for i, (w, h) in enumerate(screens, start=1):
            assert cap.capture_full_screen(monitor_index=i, save=False).size == (w, h)
        assert mons[0]["width"] == sum(w for w, _ in screens)
