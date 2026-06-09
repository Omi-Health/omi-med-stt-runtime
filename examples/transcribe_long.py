from pathlib import Path

from omi_stt.audio import make_chunks
from omi_stt.merge import merge_transcripts
from omi_stt.nemo_runtime import transcribe_nemo

chunks = make_chunks(Path("consult.wav"), chunk_seconds=25, overlap=3)
texts = transcribe_nemo([c.path for c in chunks], repo_id="omi-health/omi-med-stt-v1")
print(merge_transcripts(texts))
