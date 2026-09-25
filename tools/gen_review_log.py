#!/usr/bin/env python3
"""So sánh pre-label gốc với nhãn hiện tại trên CVAT.

Chỉ đọc và in kết quả ra stdout. KHÔNG ghi file nào.

Cách dùng:
    python3 tools/gen_review_log.py --job-id 19 --task-id 22 --round 1 \\
        --prelabels day8_round0_out/to_label/round1/prelabels
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from yolo_io import read_yolo, match_boxes  # noqa: E402

ACCEPT_IOU = 0.85
MATCH_IOU = 0.50


def run_docker(container: str, cmd: str) -> str:
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-c", cmd],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker exec {container} failed:\n{result.stderr}")
    return result.stdout.strip()


def query_django(code: str) -> str:
    escaped = code.replace('"', '\\"')
    raw = run_docker("cvat_server", f'python3 manage.py shell -c "{escaped}"')
    lines = raw.split("\n")
    for i, line in enumerate(lines):
        if "objects imported automatically" not in line:
            return "\n".join(lines[i:])
    return ""


def query_clickhouse(sql: str) -> str:
    return run_docker("cvat_clickhouse", f"clickhouse-client --query \"{sql}\"")


def get_frame_mapping(task_id: int) -> dict[int, str]:
    code = (
        f"from cvat.apps.engine.models import Task; "
        f"t = Task.objects.get(id={task_id}); "
        f"[print(f'{{f.frame}}\\t{{f.path}}') for f in t.data.images.all().order_by('frame')]"
    )
    out = query_django(code)
    mapping = {}
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) == 2:
            mapping[int(parts[0])] = parts[1]
    return mapping


def get_shapes(job_id: int) -> dict[int, list[dict]]:
    code = (
        f"from cvat.apps.engine.models import LabeledShape; "
        f"shapes = LabeledShape.objects.filter(job_id={job_id}).order_by('frame','id'); "
        f"[print(f'{{s.frame}}\\t{{list(s.points)[0]:.4f}}\\t{{list(s.points)[1]:.4f}}\\t{{list(s.points)[2]:.4f}}\\t{{list(s.points)[3]:.4f}}') for s in shapes]"
    )
    out = query_django(code)
    by_frame: dict[int, list[dict]] = {}
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        frame = int(parts[0])
        x1, y1, x2, y2 = (float(v) for v in parts[1:])
        by_frame.setdefault(frame, []).append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return by_frame


def pixel_to_yolo(box: dict, w: int, h: int) -> dict:
    cx = (box["x1"] + box["x2"]) / 2 / w
    cy = (box["y1"] + box["y2"]) / 2 / h
    bw = (box["x2"] - box["x1"]) / w
    bh = (box["y2"] - box["y1"]) / h
    return {"cls": 0, "cx": cx, "cy": cy, "w": bw, "h": bh}


def get_events_for_job(job_id: int) -> list[dict]:
    sql = (
        f"SELECT scope, timestamp, payload "
        f"FROM cvat.events "
        f"WHERE job_id = {job_id} "
        f"AND scope IN ("
        f"'create:shapes','delete:shapes','update:shapes',"
        f"'draw:object','delete:object','drag:object','resize:object',"
        f"'paste:object','copy:object','change:frame',"
        f"'action:undo','action:redo','import:dataset'"
        f") ORDER BY timestamp ASC FORMAT TabSeparatedWithNames"
    )
    out = query_clickhouse(sql)
    events = []
    for line in out.strip().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            events.append({"scope": parts[0], "timestamp": parts[1], "payload": parts[2]})
    return events


def count_user_actions_per_frame(events: list[dict], frame_mapping: dict[int, str]) -> dict[str, dict]:
    current_frame = 0
    per_frame: dict[str, dict] = {}
    for idx, name in frame_mapping.items():
        per_frame[name] = {"creates": 0, "deletes": 0, "drags": 0, "resizes": 0, "pastes": 0}

    import_ids: set[str] = set()
    for ev in events:
        if ev["scope"] == "import:dataset":
            try:
                p = json.loads(ev["payload"])
                rid = p.get("request", {}).get("id", "")
                if rid:
                    import_ids.add(rid)
            except (json.JSONDecodeError, AttributeError):
                pass

    for ev in events:
        scope = ev["scope"]
        if scope == "change:frame":
            try:
                p = json.loads(ev["payload"])
                current_frame = p.get("to", current_frame)
            except (json.JSONDecodeError, AttributeError):
                pass
            continue
        if scope == "import:dataset":
            continue
        if scope == "create:shapes":
            try:
                p = json.loads(ev["payload"])
                rid = p.get("request", {}).get("id", "")
                if rid in import_ids:
                    continue
            except (json.JSONDecodeError, AttributeError):
                pass

        fname = frame_mapping.get(current_frame)
        if not fname:
            continue
        stats = per_frame[fname]
        if scope in ("draw:object", "create:shapes"):
            stats["creates"] += 1
        elif scope in ("delete:object", "delete:shapes"):
            stats["deletes"] += 1
        elif scope == "drag:object":
            stats["drags"] += 1
        elif scope in ("resize:object", "update:shapes"):
            stats["resizes"] += 1
        elif scope == "paste:object":
            stats["pastes"] += 1

    return per_frame


def describe_position(cx: float, cy: float, w: float, h: float) -> str:
    if cx < 0.33:
        horiz = "trái"
    elif cx > 0.67:
        horiz = "phải"
    else:
        horiz = "giữa"

    if cy < 0.33:
        vert = "trên"
    elif cy > 0.67:
        vert = "dưới"
    else:
        vert = "giữa"

    area = w * h
    if area > 0.04:
        size = "xe lớn"
    elif area > 0.01:
        size = "xe vừa"
    elif area > 0.003:
        size = "xe nhỏ"
    else:
        size = "xe rất xa"

    clipped = []
    if cx - w / 2 < 0.02:
        clipped.append("mép trái")
    if cx + w / 2 > 0.98:
        clipped.append("mép phải")
    if cy - h / 2 < 0.02:
        clipped.append("mép trên")
    if cy + h / 2 > 0.98:
        clipped.append("mép dưới")

    if clipped:
        return f"{size} sát {'/'.join(clipped)}"
    if vert == horiz == "giữa":
        return f"{size} trung tâm"
    parts = [size]
    if vert != "giữa":
        parts.append(vert)
    if horiz != "giữa":
        parts.append(horiz)
    return " ".join(parts)


def diff_one_frame(pre_boxes: list[dict], cur_boxes: list[dict]) -> list[dict]:
    matches = match_boxes(pre_boxes, cur_boxes, MATCH_IOU)
    matched_pre = {i for i, _, _ in matches}
    matched_cur = {j for _, j, _ in matches}
    results = []
    for i, j, score in matches:
        action = "accepted" if score >= ACCEPT_IOU else "edited"
        results.append({"action": action, "box": cur_boxes[j], "iou": score,
                         "pre_box": pre_boxes[i]})
    for j, box in enumerate(cur_boxes):
        if j not in matched_cur:
            results.append({"action": "added", "box": box, "iou": None, "pre_box": None})
    for i, box in enumerate(pre_boxes):
        if i not in matched_pre:
            results.append({"action": "deleted", "box": box, "iou": None, "pre_box": box})
    return results


def main():
    parser = argparse.ArgumentParser(description="So sánh pre-label vs CVAT (stdout only)")
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--prelabels", type=str, required=True)
    parser.add_argument("--img-width", type=int, default=1280)
    parser.add_argument("--img-height", type=int, default=720)
    args = parser.parse_args()

    prelabel_dir = Path(args.prelabels)
    if not prelabel_dir.is_absolute():
        prelabel_dir = ROOT / prelabel_dir
    W, H = args.img_width, args.img_height

    print("▶ Đọc frame mapping từ CVAT…")
    frame_mapping = get_frame_mapping(args.task_id)
    print(f"  {len(frame_mapping)} frames")

    print("▶ Đọc shapes hiện tại từ CVAT DB…")
    shapes_by_frame = get_shapes(args.job_id)

    print("▶ Đọc ClickHouse events…")
    events = get_events_for_job(args.job_id)
    frame_stats = count_user_actions_per_frame(events, frame_mapping)

    print("▶ So sánh pre-label vs nhãn hiện tại…\n")
    print("=" * 90)

    totals = {"accepted": 0, "edited": 0, "deleted": 0, "added": 0}
    csv_rows = []

    for frame_idx in sorted(frame_mapping.keys()):
        fname = frame_mapping[frame_idx]
        stem = Path(fname).stem

        pre_path = prelabel_dir / f"{stem}.txt"
        pre_boxes = read_yolo(pre_path) if pre_path.exists() else []

        pixel_boxes = shapes_by_frame.get(frame_idx, [])
        cur_boxes = [pixel_to_yolo(b, W, H) for b in pixel_boxes]

        diffs = diff_one_frame(pre_boxes, cur_boxes)
        stats = frame_stats.get(fname, {})

        actions_here = {}
        for d in diffs:
            actions_here[d["action"]] = actions_here.get(d["action"], 0) + 1
            totals[d["action"]] += 1

        interesting = [d for d in diffs if d["action"] != "accepted"]

        if interesting or any(v > 0 for v in stats.values()):
            print(f"\n📷 {fname}  (pre={len(pre_boxes)} → now={len(cur_boxes)})  "
                  f"CVAT events: {stats}")
            print(f"   Diff: {actions_here}")

            for d in diffs:
                box = d["box"]
                desc = describe_position(box["cx"], box["cy"], box["w"], box["h"])
                iou_str = f"  IoU={d['iou']:.3f}" if d["iou"] is not None else ""
                marker = {"added": "🟢", "deleted": "🔴", "edited": "🟡", "accepted": "⚪"}
                print(f"   {marker.get(d['action'], '?')} {d['action']:10s}  {desc}{iou_str}")

                if d["action"] != "accepted":
                    reason = _auto_reason(d["action"], d.get("iou"), box)
                    csv_rows.append({
                        "round": args.round,
                        "frame_id": fname,
                        "object": desc,
                        "action": d["action"],
                        "rule_or_reason": reason,
                    })

    print("\n" + "=" * 90)
    print(f"\n📊 Tổng kết: {totals}")
    judged = totals["accepted"] + totals["edited"] + totals["deleted"]
    if judged:
        print(f"   Accept rate: {totals['accepted']/judged:.0%}")

    print(f"\n{'─'*90}")
    print("Suggested REVIEW_LOG.csv rows (copy what you need):")
    print("─" * 90)
    print("round,frame_id,object,action,rule_or_reason")
    for r in csv_rows:
        print(f"{r['round']},{r['frame_id']},{r['object']},{r['action']},{r['rule_or_reason']}")

    return 0


def _auto_reason(action: str, iou: float | None, box: dict) -> str:
    if action == "added":
        return "AI bỏ sót xe nên vẽ thêm box"
    elif action == "deleted":
        return "Không phải xe hoặc box thừa nên xóa"
    elif action == "edited":
        if iou is not None and iou < 0.7:
            return "Kéo box ôm sát thân xe vì AI lệch nhiều"
        return "Kéo nhẹ box cho sát thân xe hơn"
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
