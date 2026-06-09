import json
from pathlib import Path

from omi_stt.nemo_runtime import transcribe_nemo

rows = [json.loads(line) for line in Path("manifest.jsonl").read_text().splitlines() if line.strip()]
paths = [row["audio_filepath"] for row in rows]
texts = transcribe_nemo(paths, repo_id="omi-health/omi-med-stt-v1")
for row, text in zip(rows, texts):
    print(json.dumps({**row, "prediction": text}, ensure_ascii=False))
