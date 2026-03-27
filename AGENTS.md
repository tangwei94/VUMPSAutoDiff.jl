# Agent Instructions

## General Instructions
- Do not commit any plans in `docs/plans`.  
- When you describe your designs/plans, you should be as specific as possible. Describe in great detail what you would do.  

## Julia-Specific Instructions
### scope
- Scope is limited to `README.md`, `src/**`, `test/**`, and `Project.toml` unless explicitly asked otherwise.
- Do not inspect or modify files outside that scope unless explicitly asked, except for package-specific in-bounds paths documented below.
### tests and persistent julia REPL
- Do not edit/run tests unless you are explicitly asked to do so. 
- When you run tests, always use a persistent Julia REPL rather than the command line julia command. 
- Never start a new Julia REPL process when an existing REPL session can be reused.
- Use Revise and `include()` to reload code in REPL after edits.
- Restart Julia only if required by:
  - struct or type-definition changes
  - macro or generated-function changes
  - dependency changes in `Project.toml` or `Manifest.toml`
  - Revise failing to pick up changes correctly
- If restarting Julia, explain why.

## Package-Specific Instructions
- empty