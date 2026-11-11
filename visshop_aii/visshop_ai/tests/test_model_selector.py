from visshop.frame_skipper import AdaptiveFrameSkipper
from visshop.model_selector import DeviceInfo, select_model


def test_gpu_gtx1660_selects_yolo11n():
    c = select_model(DeviceInfo(True, 5.9, "GTX 1660"))
    assert (c.name, c.device, c.half, c.frame_skip) == ("yolo11n", "cuda:0", True, 3)


def test_cpu_falls_back_to_yolov8n_skip5():
    c = select_model(DeviceInfo(False))
    assert (c.name, c.device, c.frame_skip) == ("yolov8n", "cpu", 5)


def test_low_vram_gpu_uses_cpu_path():
    assert select_model(DeviceInfo(True, 4.0, "GTX 1650")).device == "cpu"


def test_force_cpu_and_env(monkeypatch):
    assert select_model(DeviceInfo(True, 12, "RTX"), force="cpu").device == "cpu"
    monkeypatch.setenv("VISSHOP_DEVICE", "cpu")
    assert select_model(DeviceInfo(True, 12, "RTX")).device == "cpu"


def test_skipper_adapts():
    s = AdaptiveFrameSkipper(base_n=3, source_fps=30, adjust_every=5)
    for _ in range(30):
        s.report(0.1)       # trop lent (budget 33 ms)
    assert s.n > 3
    for _ in range(300):
        s.report(0.001)     # beaucoup de marge
    assert s.n == s.min_n
    assert s.should_analyze(0) and not s.should_analyze(1)


def test_skipper_fixed():
    s = AdaptiveFrameSkipper(base_n=5, adaptive=False)
    for _ in range(100):
        s.report(1.0)
    assert s.n == 5
