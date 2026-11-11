#!/usr/bin/env python3
"""VisShop AI - point d'entree.

Exemples :
  python main.py --demo theft_pocket --save-video out.mp4      # sans camera ni modele
  python main.py --source 0 --display                          # webcam USB
  python main.py --source rtsp://user:pwd@192.168.1.10/stream --config configs/config.example.json
  python main.py --source video.mp4 --device cpu
"""
import argparse
import logging
import os
import sys

import cv2

from visshop.config import load_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VisShop AI - detection de vol en magasin")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", help="0 (USB), URL RTSP/IP ou chemin de fichier video")
    g.add_argument("--demo", metavar="SCENARIO",
                   help="scenario synthetique (ex: theft_pocket, normal_browse, ...) ou 'list'")
    p.add_argument("--config", help="fichier JSON de configuration (zones, seuils...)")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--device", choices=["cpu", "cuda"], help="force CPU ou GPU")
    p.add_argument("--display", action="store_true", help="affiche le flux (q pour quitter)")
    p.add_argument("--save-video", help="enregistre la video annotee (.mp4)")
    p.add_argument("--style", choices=["monitor", "classic"], default="monitor",
                   help="monitor = rendu 'AI SECURITY MONITOR' (defaut) ; classic = cadre rouge simple")
    p.add_argument("--no-summary", action="store_true", help="pas d'ecran final INCIDENT REPORT SUMMARY")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--timeline-csv", help="exporte les scores par personne et par instant (reglage des seuils)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _show(img) -> bool:
    cv2.imshow("VisShop AI", img)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.demo:
        from visshop.synthetic import SCENARIO_NAMES, build_scenario

        if args.demo == "list":
            print("\n".join(SCENARIO_NAMES))
            return 0
        from visshop.evaluation import build_demo_pipeline

        scenario = build_scenario(args.demo)
        cfg = load_config(args.config) if args.config else scenario.make_config()
        pipeline = build_demo_pipeline(scenario, args.output_dir, cfg)
        stream, fps = scenario.frames(), scenario.fps
        print(f"[demo] {scenario.name}: {scenario.description}")
    else:
        from visshop.factory import build_real_pipeline
        from visshop.video_source import VideoSource

        cfg = load_config(args.config)
        src = VideoSource(args.source)
        pipeline = build_real_pipeline(cfg, args.output_dir, args.device, src.fps,
                                       adaptive=None if not src.is_file else False)
        stream, fps = iter(src), src.fps

    if args.style == "classic":
        from visshop.overlay import Overlay
        pipeline.overlay = Overlay()
    writer, n = None, 0
    last_size = None
    csv_f = csv_w = None
    if args.timeline_csv:
        import csv
        from visshop.types import Behavior
        csv_f = open(args.timeline_csv, "w", newline="", encoding="utf-8")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["t", "track_id", "active", "behavior"]
                       + [f"raw_{b.value}" for b in Behavior] + [f"smooth_{b.value}" for b in Behavior])
    try:
        for frame, ts in stream:
            res = pipeline.process_frame(frame, ts)
            for ev in res.events:
                print(f"[ALERTE] t={ev.frame_ts:6.2f}s  ID{ev.track_id}  {ev.behavior_label}  "
                      f"score={ev.score:.2f}  -> {ev.screenshot}")
            if csv_w and res.analyzed:
                for tid, d in res.decisions.items():
                    csv_w.writerow([f"{ts:.2f}", tid, int(d.active), d.behavior.value if d.behavior else ""]
                                   + [f"{v:.3f}" for v in d.raw.values()] + [f"{v:.3f}" for v in d.smoothed.values()])
            last_size = (res.frame.shape[1], res.frame.shape[0])
            if args.save_video:
                if writer is None:
                    h, w = res.frame.shape[:2]
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(res.frame)
            if args.display and not _show(res.frame):
                break
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if writer and last_size and args.style == "monitor" and not args.no_summary:
            from visshop.overlay import summary_frame
            card = summary_frame(last_size, pipeline.incidents)
            for _ in range(int(3 * fps)):
                writer.write(card)
        if writer:
            writer.release()
        if csv_f:
            csv_f.close()
        if args.display:
            cv2.destroyAllWindows()
    print(f"{n} frames traitees, {pipeline.alerts.total_events} alerte(s). "
          f"Journal: {pipeline.alerts.log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
