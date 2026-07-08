# API Agent Context

- Keep FastAPI app assembly in `src/api/main.py`.
- Keep app-owned configuration in `src/api/config`.
- Keep cross-cutting API concerns in `src/api/common`.
- Keep reusable provider and schema infrastructure in `src/api/platform`.
- Keep capability-specific code in `src/api/capabilities`.

