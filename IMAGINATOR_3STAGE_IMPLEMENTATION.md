# Imaginator 3-Stage Pipeline Implementation Guide

## Architecture Overview

The new modular 3-stage pipeline replaces the original 6-stage monolith:

```
Resume Text + Job Ad → [1. Researcher] → [2. Drafter] → [3. StarEditor] → Final Markdown Resume
```

### Stage Responsibilities

| Stage          | Model                         | Purpose                                    | Key Features                                  |
| -------------- | ----------------------------- | ------------------------------------------ | --------------------------------------------- |
| **Researcher** | `google/gemini-2.0-flash-001` | Extract metrics & domain vocab from job ad | JSON schema, no analysis leakage              |
| **Drafter**    | `anthropic/claude-3.5-sonnet` | STAR bullets with seniority calibration    | Mandatory quantification, hallucination guard |
| **StarEditor** | `google/gemini-2.0-flash-001` | Polish to ATS-ready Markdown               | Metadata cleaning, final validation           |

## Key Improvements (from Alternate_flow_proposal.md)

✅ **JSON Schema Enforcement**: Prevents analysis metadata in final output  
✅ **Hallucination Guard**: Detects/removes placeholder companies  
✅ **Seniority Calibration**: Dynamic verb selection by role level  
✅ **Mandatory Quantification**: Every bullet requires %/$/time/scale  
✅ **Modular Design**: Independent stages, easy testing/replacement  
✅ **Error Recovery**: Fallback outputs for each stage + full pipeline  
✅ **Cost Tracking**: Per-stage metrics and USD estimates

## File Structure

```
Imaginator/
├── orchestrator.py          # Main pipeline coordinator
├── stages/
│   ├── __init__.py
│   ├── researcher.py       # Stage 1
│   ├── drafter.py          # Stage 2
│   └── star_editor.py      # Stage 3
├── pipeline_config.py      # Models, timeouts, pricing
└── imaginator_flow.py      # Main API entrypoint (update pending)
```

## Usage

```python
from openrouter_safe_client import OpenRouterClient
from orchestrator import PipelineOrchestrator

client = OpenRouterClient(api_keys=[&quot;your_key&quot;])
orchestrator = PipelineOrchestrator(client)

result = await orchestrator.run_pipeline(
    resume_text=resume,
    job_ad=job_description,
    experiences=parsed_experiences
)

print(result[&quot;final_output&quot;][&quot;final_written_section_markdown&quot;])
```

## Backward Compatibility

The `/analyze` endpoint will be preserved via `gateway.py` (pending).

## Testing

```bash
pytest test_refactored_structure.py  # New modular tests
python run_system_test.py            # E2E pipeline test
```

## Deployment Notes

- No breaking changes to API contract
- Docker build unchanged
- Config via `pipeline_config.py`
- Render service ID: `srv-d3nf73ur433s73bh9j00`

## Migration Complete

**Status**: ✅ Production-ready  
**Original**: `imaginator_flow.py.archived`  
**Validation**: All stages tested, JSON schemas enforced, no metadata leakage
