#!/usr/bin/env python
"""Record real Intel RealSense data in the exact layout the SynthTree pipeline reads.

Install (pick one):
    Linux / Windows : pip install pyrealsense2
    macOS (arm64)   : pip install pyrealsense2-macosx      (community build; macOS usually needs `sudo`
                      for USB access:  sudo .venv/bin/python scripts/capture/rs_capture.py ...)
    CLI check       : brew install librealsense && rs-enumerate-devices -c

Modes (all streams use the resolution/fps of configs/sensor_d435i.yaml unless overridden;
depth is ALIGNED TO COLOUR and saved as 16-bit PNG in millimetres, exactly like the synthetic data):

  intrinsics   dump colour intrinsics, depth scale, device info -> rec/intrinsics.json
               --write-config configs/sensor_d435i.yaml  also updates the sensor config in place
  wall         flat-wall noise recording at one distance  -> rec/wall/<distance>/depth/NNNNN.png
               python rs_capture.py wall --distance 1.2 --frames 30
  leaves       real leaves against a far background      -> rec/leaves/{depth,rgb}/NNNNN.png
  clip         free recording (e.g. fan level videos)    -> rec/clips/<name>/{depth,rgb}/ + timestamps.txt
               python rs_capture.py clip --name fan_high --seconds 10
  orbit        real dataset sequence (hand-held orbit)   -> rec/real/tree_<id>/wind_<level>/{rgb,depth}/ + timestamps.txt

Every recording writes meta.json (device serial, firmware, preset, exposure, intrinsics, depth scale).
Untested on hardware in this repo (no sensor attached when written): run `intrinsics` first as a check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402


def _rs():
    try:
        import pyrealsense2 as rs
    except ImportError:
        sys.exit("pyrealsense2 not installed: `pip install pyrealsense2` (macOS: `pip install pyrealsense2-macosx`)")
    return rs


def _imwrite(path, img):
    import cv2
    if img.ndim == 3:
        img = img[..., ::-1]  # RGB -> BGR for OpenCV
    if not cv2.imwrite(path, np.ascontiguousarray(img), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(path)


def load_sensor_cfg(path):
    import yaml
    return yaml.safe_load(open(path))


class Camera:
    def __init__(self, width, height, fps, preset=None, laser_power=None, warmup=30):
        rs = _rs()
        self.rs = rs
        if len(rs.context().query_devices()) == 0:
            sys.exit("No RealSense device found. Plug the camera into a USB-3 port (blue / USB-C data cable), "
                     "then retry; on macOS run with sudo if it is connected but still not found.")
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipe.start(cfg)
        dev = self.profile.get_device()
        self.depth_sensor = dev.first_depth_sensor()
        self.depth_scale = float(self.depth_sensor.get_depth_scale())
        if preset is not None and self.depth_sensor.supports(rs.option.visual_preset):
            names = {"default": 1, "high_accuracy": 3, "high_density": 4, "medium_density": 5}
            self.depth_sensor.set_option(rs.option.visual_preset, names.get(preset, preset))
        if laser_power is not None and self.depth_sensor.supports(rs.option.laser_power):
            self.depth_sensor.set_option(rs.option.laser_power, float(laser_power))
        self.align = rs.align(rs.stream.color)
        self.info = {
            "name": dev.get_info(rs.camera_info.name),
            "serial": dev.get_info(rs.camera_info.serial_number),
            "firmware": dev.get_info(rs.camera_info.firmware_version),
            "depth_scale_m": self.depth_scale,
            "preset": preset, "laser_power": laser_power,
            "width": width, "height": height, "fps": fps,
        }
        for _ in range(warmup):  # let auto-exposure settle
            self.pipe.wait_for_frames()

    def intrinsics(self):
        cs = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile().get_intrinsics()
        ds = self.profile.get_stream(self.rs.stream.depth).as_video_stream_profile().get_intrinsics()
        pack = lambda i: {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy,  # noqa: E731
                          "model": str(i.model), "coeffs": list(i.coeffs)}
        return {"color": pack(cs), "depth_native": pack(ds)}

    def grab(self):
        fs = self.align.process(self.pipe.wait_for_frames())
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        if not c or not d:
            return None
        rgb = np.asanyarray(c.get_data()).copy()
        raw = np.asanyarray(d.get_data()).astype(np.float64)
        mm = np.clip(np.round(raw * self.depth_scale * 1000.0), 0, 65535).astype(np.uint16)
        return rgb, mm, fs.get_timestamp() / 1000.0

    def stop(self):
        self.pipe.stop()


def record(cam, out, frames=None, seconds=None, save_rgb=True, save_depth=True, preview=False):
    os.makedirs(out, exist_ok=True)
    if save_rgb:
        os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    if save_depth:
        os.makedirs(os.path.join(out, "depth"), exist_ok=True)
    ts, t0, i = [], time.time(), 0
    while (frames is None or i < frames) and (seconds is None or time.time() - t0 < seconds):
        g = cam.grab()
        if g is None:
            continue
        rgb, mm, t = g
        name = f"{i:05d}.png"
        if save_rgb:
            _imwrite(os.path.join(out, "rgb", name), rgb)
        if save_depth:
            _imwrite(os.path.join(out, "depth", name), mm)
        ts.append(t)
        if preview:
            import cv2
            v = cv2.applyColorMap(cv2.convertScaleAbs(mm, alpha=255 / 3000), cv2.COLORMAP_JET)
            cv2.imshow("depth", v)
            if cv2.waitKey(1) == 27:
                break
        i += 1
    with open(os.path.join(out, "timestamps.txt"), "w") as f:
        f.write("# frame device_time_s\n" + "".join(f"{k:05d} {t:.6f}\n" for k, t in enumerate(ts)))
    valid = (mm > 0).mean() if ts else 0
    meta = {"device": cam.info, "intrinsics": cam.intrinsics(), "n_frames": len(ts),
            "fps_measured": (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else None,
            "last_frame_valid_depth_frac": float(valid), "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"saved {len(ts)} frames -> {out}  (valid depth in last frame: {valid:.0%})")
    return meta


def write_sensor_config(path, intr, depth_scale, fps):
    """Update fx/fy/cx/cy/resolution in the sensor YAML, keeping comments elsewhere intact."""
    import re
    s = open(path).read()
    c = intr["color"]
    for k, v in (("width", c["width"]), ("height", c["height"]), ("fx", c["fx"]), ("fy", c["fy"]), ("cx", c["cx"]), ("cy", c["cy"])):
        s = re.sub(rf"(\n  {k}: )[0-9.]+", rf"\g<1>{v}", s, count=1)
    s = re.sub(r"(\nfps: )[0-9.]+", rf"\g<1>{fps}", s, count=1)
    dn = intr["depth_native"]
    if dn["fx"] > 0:  # native depth sampling relative to the colour grid after alignment
        eff = round(min(1.0, (dn["fx"] / dn["width"]) / (c["fx"] / c["width"])), 3)
        s = re.sub(r"(\n  effective_scale: )[0-9.]+", rf"\g<1>{eff}", s, count=1)
    if "# intrinsics from the device" not in s:
        s = s.replace("name: realsense_d435i", "name: realsense_d435i  # intrinsics from the device (rs_capture.py)")
    open(path, "w").write(s)
    print("updated", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["intrinsics", "wall", "leaves", "clip", "orbit"])
    ap.add_argument("--sensor", default=os.path.join(REPO, "configs", "sensor_d435i.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO, "rec"))
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--preset", default="high_accuracy", help="visual preset; keep identical for all recordings")
    ap.add_argument("--laser-power", type=float)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--distance", type=float, help="wall: nominal distance in metres")
    ap.add_argument("--name", help="clip name")
    ap.add_argument("--tree", type=int)
    ap.add_argument("--wind", help="orbit: wind level name (calm/low/medium/high)")
    ap.add_argument("--write-config", nargs="?", const="__default__", help="intrinsics: update the sensor YAML")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--countdown", type=float, default=3.0)
    a = ap.parse_args()
    sc = load_sensor_cfg(a.sensor)
    W, H = a.width or sc["rgb"]["width"], a.height or sc["rgb"]["height"]
    fps = a.fps or sc.get("fps", 30)
    cam = Camera(W, H, fps, a.preset, a.laser_power)
    try:
        if a.mode == "intrinsics":
            intr = cam.intrinsics()
            os.makedirs(a.out, exist_ok=True)
            info = {**intr["color"], "depth_scale_m": cam.depth_scale, "device": cam.info, "depth_native": intr["depth_native"]}
            json.dump(info, open(os.path.join(a.out, "intrinsics.json"), "w"), indent=2)
            print(json.dumps(info, indent=2))
            if a.write_config:
                write_sensor_config(a.sensor if a.write_config == "__default__" else a.write_config, intr, cam.depth_scale, fps)
            return
        if a.countdown:
            print(f"recording in {a.countdown:.0f} s ...")
            time.sleep(a.countdown)
        if a.mode == "wall":
            assert a.distance, "--distance is required"
            record(cam, os.path.join(a.out, "wall", f"{a.distance:g}"), frames=a.frames or 30, save_rgb=False, preview=a.preview)
        elif a.mode == "leaves":
            record(cam, os.path.join(a.out, "leaves"), frames=a.frames or 30, preview=a.preview)
        elif a.mode == "clip":
            assert a.name, "--name is required"
            record(cam, os.path.join(a.out, "clips", a.name), frames=a.frames, seconds=a.seconds or 10, preview=a.preview)
        elif a.mode == "orbit":
            assert a.tree is not None and a.wind, "--tree and --wind are required"
            record(cam, os.path.join(a.out, "real", f"tree_{a.tree:02d}", f"wind_{a.wind}"),
                   frames=a.frames, seconds=a.seconds or sc.get("orbit_seconds", 20), preview=a.preview)
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
