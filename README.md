uv run python scripts/agentic_tender_pipeline.py `
  --input-root downloads/batch_runs `
  --output-dir outputs/tender_structured `
  --base-url http://10.4.25.56:8000/v1 `
  --model Qwen/Qwen3-30B-A3B-GPTQ-Int4 `
  --api-key EMPTY `
  --workers 4 `
  --heartbeat-seconds 15