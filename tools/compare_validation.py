"""
Manuel etiket (dl_*_phases.json) vs sistem ciktisi (dl_*_system.json) karsilastirma.

Sistem 'hata var' dedi mi? -> errors[eid].confidence >= UI_CONFIDENCE_THRESHOLD
Manuel 'hata var' dedi mi? -> rep_labels[i][eid] == True

Cikti: per-error confusion matrix + per-variant breakdown + V1 karari.
"""
import json
from pathlib import Path

FIXTURE_DIR = Path("tests/fixtures/validation")
SYSTEM_DIR = FIXTURE_DIR / "system_output"

UI_CONFIDENCE_THRESHOLD = 0.30   # bu esigi gecince sistem 'hata' diyor
V1_FP_THRESHOLD = 0.20
V1_ACC_THRESHOLD = 0.70

def system_says_error(rep: dict, error_id: str) -> bool:
    err = rep.get("errors", {}).get(error_id)
    if not err:
        return False
    return err.get("confidence", 0) >= UI_CONFIDENCE_THRESHOLD

def main():
    label_files = sorted(FIXTURE_DIR.glob("deadlift_*_phases.json"))
    if not label_files:
        print(f"Manuel etiket bulunamadi: {FIXTURE_DIR}/deadlift_*_phases.json")
        return

    rows = []                # (video, variant, rep_id, error_id, manual, system)
    rep_count_issues = []

    for label_path in label_files:
        stem = label_path.stem.replace("_phases", "")
        sys_path = SYSTEM_DIR / f"{stem}_system.json"
        if not sys_path.exists():
            print(f"SKIP {stem}: sistem ciktisi yok (run_validation.py calistir)")
            continue

        manual = json.loads(label_path.read_text(encoding="utf-8"))
        system = json.loads(sys_path.read_text(encoding="utf-8"))
        variant = manual.get("variant", "conventional")
        m_reps = manual.get("rep_labels", [])
        s_reps = system.get("reps", [])

        if len(m_reps) != len(s_reps):
            rep_count_issues.append((stem, len(m_reps), len(s_reps)))

        for m, s in zip(m_reps, s_reps):
            for eid in ("incomplete_lockout", "uncontrolled_descent"):
                rows.append((
                    stem, variant, m["rep_id"], eid,
                    bool(m.get(eid, False)),
                    system_says_error(s, eid),
                ))

    print("=" * 60)
    print("REP COUNT")
    print("=" * 60)
    if rep_count_issues:
        for v, m, s in rep_count_issues:
            print(f"  {v}: manuel={m}  sistem={s}  UYUMSUZ")
    else:
        print("  Tum videolar: rep sayilari esit.")

    triggered = []

    variants = sorted({r[1] for r in rows})
    groups = [("ALL", None)] + [(f"variant={v}", v) for v in variants]

    for group_name, var_filter in groups:
        print(f"\n=== {group_name} ===")
        for eid in ("incomplete_lockout", "uncontrolled_descent"):
            subset = [r for r in rows
                      if r[3] == eid
                      and (var_filter is None or r[1] == var_filter)]
            if not subset:
                continue
            tp = sum(1 for r in subset if r[4] and r[5])
            fp = sum(1 for r in subset if not r[4] and r[5])
            fn = sum(1 for r in subset if r[4] and not r[5])
            tn = sum(1 for r in subset if not r[4] and not r[5])
            n = len(subset)
            acc = (tp + tn) / n if n else 0
            fpr = fp / (fp + tn) if (fp + tn) else 0
            tpr = tp / (tp + fn) if (tp + fn) else None

            print(f"\n  {eid}  n={n}")
            print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
            print(f"    accuracy={acc:.2%}   FP_rate={fpr:.2%}")
            if tpr is not None:
                print(f"    recall={tpr:.2%}")
            else:
                print(f"    recall=(pozitif sample yok)")

            if var_filter is None:
                if fpr > V1_FP_THRESHOLD:
                    triggered.append(f"{eid}: FP_rate {fpr:.0%} > {V1_FP_THRESHOLD:.0%}")
                if acc < V1_ACC_THRESHOLD:
                    triggered.append(f"{eid}: accuracy {acc:.0%} < {V1_ACC_THRESHOLD:.0%}")

    print("\n" + "=" * 60)
    print("V0 / V1 KARARI")
    print("=" * 60)
    if triggered:
        print("V1 TRIGGER BASILDI:")
        for r in triggered:
            print(f"  - {r}")
        print("\nNe yapilabilir:")
        print("  1) Eşik ayarı dene (T_LOCKOUT_THRESHOLD, HIP_VELOCITY_THRESHOLD_PER_MS)")
        print("  2) Veya V1'e geç (antrenor verisi)")
    else:
        print("Tum metrikler V0 esiklerinin altinda.")
        print("V0 ship'e hazir.")

if __name__ == "__main__":
    main()
