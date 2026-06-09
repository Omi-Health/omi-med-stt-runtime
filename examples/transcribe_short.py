from omi_stt.nemo_runtime import transcribe_nemo

texts = transcribe_nemo(["audio.wav"], repo_id="omi-health/omi-med-stt-v1")
print(texts[0])
