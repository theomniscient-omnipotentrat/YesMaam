"""
evaluate_system.py

Evaluation script for:
1. Face detection rate
2. Recognition accuracy
3. False reject rate
4. False accept rate
5. Wrong identity rate
6. Campus alert accuracy
7. Processing speed

Run:
    python evaluate_system.py --ground-truth evaluation_ground_truth.csv --sample-every 0.5
"""

import argparse
import csv
import os
import time
from datetime import datetime, timedelta

import cv2

import database
import embeddings as emb_module
from detector import FaceDetector

try:
    import campus_monitoring
except Exception:
    campus_monitoring = None


def read_ground_truth(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def evaluate_video(row, detector, emb_db, sample_every):
    video_path = row["video_path"]
    zone = row["zone"]
    event_date = row["event_date"]
    start_time = row["start_time"]
    actual_id = row["actual_student_id"].strip()
    actual_name = row["actual_student_name"].strip()
    expected_zone = row["expected_zone"].strip()
    expected_status = row["expected_status"].strip().upper()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration = total_frames / fps if fps else 0

    sample_interval = max(1, int(fps * sample_every))

    start_dt = datetime.strptime(
        f"{event_date} {start_time}",
        "%Y-%m-%d %H:%M",
    )

    frame_index = 0
    sampled_frames = 0
    face_detected_frames = 0

    correct_recognitions = 0
    false_rejects = 0
    false_accepts = 0
    wrong_identity = 0
    unknown_correct = 0

    alert_checks = 0
    correct_alert_decisions = 0

    predictions = []

    processing_start = time.perf_counter()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % sample_interval != 0:
            frame_index += 1
            continue

        sampled_frames += 1

        simulated_seconds = frame_index / fps
        event_dt = start_dt + timedelta(seconds=simulated_seconds)
        event_time = event_dt.strftime("%H:%M:%S")

        faces = detector.detect(frame)

        detected = len(faces) > 0

        if detected:
            face_detected_frames += 1

        if not detected:
            predictions.append({
                "test_id": row["test_id"],
                "video_path": video_path,
                "frame_index": frame_index,
                "event_time": event_time,
                "actual_student_id": actual_id,
                "predicted_student_id": "NO_FACE",
                "predicted_student_name": "NO_FACE",
                "distance": "",
                "detected_zone": zone,
                "expected_zone": expected_zone,
                "expected_status": expected_status,
                "system_status": "NO_FACE",
                "correct_recognition": "No",
                "alert_correct": "N/A",
            })

            frame_index += 1
            continue

        # For controlled test videos, evaluate the largest/first face.
        face = faces[0]
        bbox = face["bbox"]

        embedding = emb_module.extract_embedding_from_frame(frame, bbox)

        if embedding is None:
            predicted_id = "NO_EMBEDDING"
            predicted_name = "NO_EMBEDDING"
            distance = ""
        else:
            predicted_id, predicted_name, distance = emb_db.recognize(embedding)

        # Recognition metrics
        correct_recognition = False

        if actual_id.upper() == "UNKNOWN":
            if predicted_id == "Unknown":
                unknown_correct += 1
                correct_recognition = True
            elif predicted_id not in ["Unknown", "NO_FACE", "NO_EMBEDDING"]:
                false_accepts += 1

        else:
            if predicted_id == actual_id:
                correct_recognitions += 1
                correct_recognition = True
            elif predicted_id == "Unknown":
                false_rejects += 1
            elif predicted_id in ["NO_FACE", "NO_EMBEDDING"]:
                false_rejects += 1
            else:
                wrong_identity += 1

        # Alert decision metrics
        system_status = "N/A"
        alert_correct = "N/A"

        if actual_id.upper() != "UNKNOWN" and predicted_id == actual_id:
            alert_checks += 1

            if zone.strip().lower() == expected_zone.strip().lower():
                system_status = "OK"
            else:
                system_status = "ALERT"

            if system_status == expected_status:
                correct_alert_decisions += 1
                alert_correct = "Yes"
            else:
                alert_correct = "No"

        elif actual_id.upper() == "UNKNOWN":
            alert_checks += 1

            if predicted_id == "Unknown":
                system_status = "UNKNOWN"
            else:
                system_status = "FALSE_ACCEPT"

            if system_status == expected_status:
                correct_alert_decisions += 1
                alert_correct = "Yes"
            else:
                alert_correct = "No"

        predictions.append({
            "test_id": row["test_id"],
            "video_path": video_path,
            "frame_index": frame_index,
            "event_time": event_time,
            "actual_student_id": actual_id,
            "predicted_student_id": predicted_id,
            "predicted_student_name": predicted_name,
            "distance": distance,
            "detected_zone": zone,
            "expected_zone": expected_zone,
            "expected_status": expected_status,
            "system_status": system_status,
            "correct_recognition": "Yes" if correct_recognition else "No",
            "alert_correct": alert_correct,
        })

        frame_index += 1

    cap.release()

    processing_time = time.perf_counter() - processing_start

    enrolled_attempts = 0
    unknown_attempts = 0

    if actual_id.upper() == "UNKNOWN":
        unknown_attempts = sampled_frames
    else:
        enrolled_attempts = sampled_frames

    return {
        "test_id": row["test_id"],
        "video_path": video_path,
        "zone": zone,
        "actual_student_id": actual_id,
        "expected_zone": expected_zone,
        "expected_status": expected_status,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "video_duration_seconds": round(video_duration, 2),
        "sampled_frames": sampled_frames,
        "face_detected_frames": face_detected_frames,
        "correct_recognitions": correct_recognitions,
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
        "wrong_identity": wrong_identity,
        "unknown_correct": unknown_correct,
        "alert_checks": alert_checks,
        "correct_alert_decisions": correct_alert_decisions,
        "processing_time_seconds": round(processing_time, 2),
        "sample_processing_rate": round(sampled_frames / processing_time, 2) if processing_time else 0,
        "real_time_factor": round(video_duration / processing_time, 2) if processing_time else 0,
        "predictions": predictions,
        "enrolled_attempts": enrolled_attempts,
        "unknown_attempts": unknown_attempts,
    }


def percent(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def write_csv(path, rows):
    if not rows:
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Collect all possible column names from every row
    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="evaluation_ground_truth.csv")
    parser.add_argument("--sample-every", type=float, default=0.5)
    parser.add_argument("--output-dir", default="evaluation_results")
    args = parser.parse_args()

    database.init_db()

    if campus_monitoring is not None:
        if hasattr(campus_monitoring, "init_monitoring_db"):
            campus_monitoring.init_monitoring_db()
        elif hasattr(campus_monitoring, "init_campus_tables"):
            campus_monitoring.init_campus_tables()

    rows = read_ground_truth(args.ground_truth)

    detector = FaceDetector()
    emb_db = emb_module.get_embedding_db()

    if emb_db.count == 0:
        raise RuntimeError("No embeddings loaded. Enroll students before evaluation.")

    all_video_summaries = []
    all_predictions = []

    totals = {
        "sampled_frames": 0,
        "face_detected_frames": 0,
        "correct_recognitions": 0,
        "false_rejects": 0,
        "false_accepts": 0,
        "wrong_identity": 0,
        "unknown_correct": 0,
        "alert_checks": 0,
        "correct_alert_decisions": 0,
        "enrolled_attempts": 0,
        "unknown_attempts": 0,
        "processing_time_seconds": 0.0,
        "video_duration_seconds": 0.0,
    }

    for row in rows:
        print(f"Evaluating {row['test_id']}: {row['video_path']}")

        result = evaluate_video(
            row=row,
            detector=detector,
            emb_db=emb_db,
            sample_every=args.sample_every,
        )

        predictions = result.pop("predictions")
        all_predictions.extend(predictions)

        all_video_summaries.append(result)

        for key in totals:
            totals[key] += safe_float(result.get(key, 0))

    metrics = [{
        "metric": "Face Detection Rate",
        "value_percent": percent(totals["face_detected_frames"], totals["sampled_frames"]),
        "formula": "face_detected_frames / sampled_frames * 100",
    }, {
        "metric": "Recognition Accuracy",
        "value_percent": percent(
            totals["correct_recognitions"] + totals["unknown_correct"],
            totals["enrolled_attempts"] + totals["unknown_attempts"],
        ),
        "formula": "(correct_recognitions + unknown_correct) / all_attempts * 100",
    }, {
        "metric": "False Reject Rate",
        "value_percent": percent(totals["false_rejects"], totals["enrolled_attempts"]),
        "formula": "false_rejects / enrolled_attempts * 100",
    }, {
        "metric": "False Accept Rate",
        "value_percent": percent(totals["false_accepts"], totals["unknown_attempts"]),
        "formula": "false_accepts / unknown_attempts * 100",
    }, {
        "metric": "Wrong Identity Rate",
        "value_percent": percent(totals["wrong_identity"], totals["enrolled_attempts"]),
        "formula": "wrong_identity / enrolled_attempts * 100",
    }, {
        "metric": "Alert Accuracy",
        "value_percent": percent(totals["correct_alert_decisions"], totals["alert_checks"]),
        "formula": "correct_alert_decisions / alert_checks * 100",
    }, {
        "metric": "Average Sample Processing Rate",
        "value_percent": "",
        "formula": "sampled_frames / processing_time_seconds",
        "raw_value": round(totals["sampled_frames"] / totals["processing_time_seconds"], 2)
        if totals["processing_time_seconds"] else 0,
    }, {
        "metric": "Average Real-Time Factor",
        "value_percent": "",
        "formula": "video_duration_seconds / processing_time_seconds",
        "raw_value": round(totals["video_duration_seconds"] / totals["processing_time_seconds"], 2)
        if totals["processing_time_seconds"] else 0,
    }]

    os.makedirs(args.output_dir, exist_ok=True)

    write_csv(
        os.path.join(args.output_dir, "per_frame_predictions.csv"),
        all_predictions,
    )

    write_csv(
        os.path.join(args.output_dir, "per_video_summary.csv"),
        all_video_summaries,
    )

    write_csv(
        os.path.join(args.output_dir, "metrics_summary.csv"),
        metrics,
    )

    print("\nEvaluation complete.")
    print(f"Results saved in: {args.output_dir}")

    print("\nSummary Metrics")
    print("=" * 60)

    for metric in metrics:
        if metric.get("value_percent") != "":
            print(f"{metric['metric']}: {metric['value_percent']}%")
        else:
            print(f"{metric['metric']}: {metric.get('raw_value', '')}")

    print("\nGenerated files:")
    print(" - evaluation_results/per_frame_predictions.csv")
    print(" - evaluation_results/per_video_summary.csv")
    print(" - evaluation_results/metrics_summary.csv")


if __name__ == "__main__":
    main()