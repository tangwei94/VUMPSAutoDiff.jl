---
name: refresh-memory
description: Summarize stable repo invariants and review contracts into .agents/codex_memory.md.
---

You are updating a persistent memory file for future Codex sessions.

Rules (strict):
- Only include facts grounded in repository files you read.
- Prefer stable contracts: public API surface, invariants, conventions, performance constraints.
- Do NOT include code listings or large excerpts.
- Do NOT speculate or infer unstated intent.
- Rewrite .agents/codex_memory.md completely.
- Enforce size: .agents/codex_memory.md must be <= 300 lines.

Scope:
- Include: README.md, src/**, test/**, Project.toml
- Exclude (unless explicitly requested): data/**, docs/**, demo.jl, any Manifest.toml, *.jld2, *.log, optim_code.md, Generated files, binaries, logs

Process:
1) Read existing .agents/codex_memory.md if present. But since the memory needs to be refreshed, do not fully trust the content therein. 
2) Extract:
   - Public API (exported symbols, 1-line behavior)
   - Numerical invariants and conventions explicitly stated
   - Performance contracts explicitly stated or tested
   - Review scope defaults
   - Test entry points and determinism assumptions
3) Write .agents/codex_memory.md using this schema:

```markdown
# Codex Memory (generated)
# Updated: YYYY-MM-DD

## Public API
## Invariants & Conventions
## Performance Contracts
## Review Scope Defaults
## Testing Notes
```

4) Report briefly what changed; do not paste full contents into chat.
