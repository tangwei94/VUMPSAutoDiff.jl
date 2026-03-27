# Codex Agent Instructions

## General Instructions
- Do not commit any plans in `docs/plans`.  

## Julia-Specific Instructions
### scope
- Scope is limited to `README.md`, `src/**`, `test/**`, and `Project.toml` unless explicitly asked otherwise.
- Do not inspect or modify files outside that scope unless explicitly asked, except for package-specific in-bounds paths documented below.
### persistent julia repl
- Use a persistent Julia REPL for tests and checks.
- Never start a new Julia process when an existing REPL session can be reused.
- Use Revise and `include()` to reload code after edits.
- Restart Julia only if required by:
  - struct or type-definition changes
  - macro or generated-function changes
  - dependency changes in `Project.toml` or `Manifest.toml`
  - Revise failing to pick up changes correctly
- If restarting Julia, explain why.

## Package-Specific Instructions
- empty