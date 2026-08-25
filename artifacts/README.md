# Artifacts

Trained adapter checkpoints. `adapters/tinyllama-qat/` is a peft LoRA adapter (r=4, alpha=8,
`q_proj`/`v_proj`) produced by the legacy QAT prototype on `TinyLlama/TinyLlama-1.1B-Chat-v0.6`.

Binary weights (~4.6 MB) are committed directly. If more checkpoints are added, migrate them to
Git LFS (`git lfs install && git lfs track "artifacts/**/*.bin"`) to keep clones small.
