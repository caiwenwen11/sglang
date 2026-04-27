# Diffusion CI branch reconcile - 2026-04-27

- 旧工作目录 `/Users/mick/repos/sglang-qwen2509-align` 在 `qwen2509-align-worktree`，只保留了后续 Qwen2509 对齐改动，缺少 PR #23714 中 FLUX、layered、workflow、threshold 等 diffusion-ci 语义修复。
- PR #23714 HEAD 与 `origin/diffusion-ci` 一致，hash `f0a08ce12`。
- 新建工作目录 `/Users/mick/repos/sglang-diffusion-ci-with-qwen2509`，分支 `diffusion-ci-with-qwen2509`，base 为 `origin/diffusion-ci`，未设置 upstream。
- 已把当前 Qwen2509 4 个文件的未提交精度修复叠到新分支并解决冲突:
  - `configs/pipeline_configs/qwen_image.py`
  - `runtime/models/dits/qwen_image.py`
  - `runtime/models/encoders/qwen2_5vl.py`
  - `runtime/pipelines/qwen_image.py`
- 当前该分支包含 PR #23714 全量 diffusion-ci 改动，同时叠加 Qwen2509 后续修复。Qwen2509 当前最终图 PSNR 约 21.26，对齐约 70.9%。
