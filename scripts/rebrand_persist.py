from pathlib import Path

REPLACEMENTS = [
    ("PERSIST-AI (left)", "PERSIST-AI (left)"),
    ("Activate PERSIST-AI", "Activate PERSIST-AI"),
    ("PERSIST-AI:", "PERSIST-AI:"),
    ("Left = PERSIST-AI", "Left = PERSIST-AI"),
    ("PERSIST-AI keeps", "PERSIST-AI keeps"),
    ("PERSIST-AI kept", "PERSIST-AI kept"),
    ("PERSIST-AI maintains", "PERSIST-AI maintains"),
    ("PERSIST-AI KEEPS", "PERSIST-AI KEEPS"),
    ("PERSIST-AI panel", "PERSIST-AI panel"),
    ("PERSIST-AI TrackManager", "PERSIST-AI TrackManager"),
    ("vs PERSIST-AI", "vs PERSIST-AI"),
    ("# PERSIST-AI", "# PERSIST-AI"),
    ("PERSIST-AI Metrics", "PERSIST-AI Metrics"),
    ("PERSIST-AI Demo", "PERSIST-AI Demo"),
    ("PERSIST-AI interactive", "PERSIST-AI interactive"),
    ("PERSIST-AI Street", "PERSIST-AI Street"),
    ("VIDEO3_PERSIST_SPLIT", "VIDEO3_PERSIST_SPLIT"),
    ("PERSIST_KEEPS", "PERSIST_KEEPS"),
    ("PERSIST_GHOST", "PERSIST_GHOST"),
    ("PERSIST-AI (object permanence)", "PERSIST-AI (object permanence)"),
    ("for PERSIST-AI demos", "for PERSIST-AI demos"),
    ("Baselines vs PERSIST-AI", "Baselines vs PERSIST-AI"),
    ("PERSIST-AI is an", "PERSIST-AI is an"),
    ("PERSIST-AI models", "PERSIST-AI models"),
    ("PERSIST-AI auto-falls", "PERSIST-AI auto-falls"),
    ("PERSIST-AI maintains latent", "PERSIST-AI maintains latent"),
    ("title={PERSIST-AI}", "title={PERSIST-AI}"),
    ('title="PERSIST-AI"', 'title="PERSIST-AI"'),
    ('help="PERSIST-AI', 'help="PERSIST-AI'),
    ("PERSIST-AI —", "PERSIST-AI —"),
    ("PERSIST-AI = what", "PERSIST-AI = what"),
    ("Detection tells us what is visible. PERSIST-AI", "Detection tells us what is visible. PERSIST-AI"),
    ('cv2.putText(vis, "PERSIST-AI"', 'cv2.putText(vis, "PERSIST-AI"'),
    ('label), (right, "PERSIST-AI")', 'label), (right, "PERSIST-AI")'),
    ('((left, "BASELINE"), (right, "PERSIST-AI"))', '((left, "BASELINE"), (right, "PERSIST-AI"))'),
    ("PERSIST-AI REAL", "PERSIST-AI REAL"),
    ("PERSIST-AI on locked", "PERSIST-AI on locked"),
    ("| PERSIST-AI", "| PERSIST-AI"),
    ("PERSIST-AI REAL_MOT17", "PERSIST-AI REAL_MOT17"),
    ("PERSIST_AI_REAL_MOT17", "PERSIST_AI_REAL_MOT17"),
    ("PERSIST-AI: Object Permanence", "PERSIST-AI: Object Permanence"),
    ("PERSIST-AI: ghost", "PERSIST-AI: ghost"),
    ("PERSIST-AI (left) vs", "PERSIST-AI (left) vs"),
    ("PERSIST-AI side-by-side", "PERSIST-AI side-by-side"),
    ("Baseline vs PERSIST-AI", "Baseline vs PERSIST-AI"),
    ("Extra overlays for the PERSIST-AI", "Extra overlays for the PERSIST-AI"),
    ("masked baseline + PERSIST-AI", "masked baseline + PERSIST-AI"),
    ("PERSIST-AI tracks", "PERSIST-AI tracks"),
    ("PERSIST-AI track", "PERSIST-AI track"),
    ("PERSIST-AI ghost", "PERSIST-AI ghost"),
    ("PERSIST-AI vs", "PERSIST-AI vs"),
    ("| PERSIST-AI |", "| PERSIST-AI |"),
    ("PERSIST-AI`", "PERSIST-AI`"),
    ("PERSIST-AI,", "PERSIST-AI,"),
    ("PERSIST-AI.", "PERSIST-AI."),
    ("PERSIST-AI ", "PERSIST-AI "),
    (" PERSIST-AI", " PERSIST-AI"),
    ('class="persist-label"', 'class="persist-label"'),
    ("persist-label", "persist-label"),
]

SKIP = {".venv", "results", "__pycache__", ".git", "debug-6ac46f.log"}
EXTS = {".py", ".md", ".html", ".js", ".css", ".yaml"}

for path in Path(".").rglob("*"):
    if any(part in SKIP for part in path.parts):
        continue
    if path.suffix not in EXTS:
        continue
    text = path.read_text(encoding="utf-8")
    if "lumen" in path.as_posix() and path.name in {"brand.py"}:
        continue
    orig = text
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        print("updated", path)
