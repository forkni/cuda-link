# Code Style Instructions

Charlie already reads CLAUDE.md for project-specific guidance.

## Additional Rules

- [R1] Respect TouchDesigner naming conventions in `td_exporter/`: PascalCase class names, camelCase method names — these are intentional, not lint violations
- [R2] Follow existing patterns in the codebase
- [R3] Run tests before submitting PRs
- [R4] Ensure all lint checks pass before committing
- [R5] Use type hints for all public functions
- [R6] All CUDA-dependent tests must use `@pytest.mark.requires_cuda` — never mock the GPU in these tests
