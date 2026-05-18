---
id: builtin:debug_loop
name: Debug Loop
description: Diagnose a failure by reproducing it, isolating the cause, and verifying the fix.
trigger: debug error failure failing test bug regression exception traceback diagnose
confidence: 0.85
---

# Debug Loop

Use this procedure when behavior differs from expectations, a command fails, or
tests expose a regression.

1. State the observed failure and the expected behavior.
2. Reproduce the failure with the smallest reliable command or example.
3. Inspect the error output and the code path it points to.
4. Form one concrete hypothesis about the root cause.
5. Make the smallest coherent fix that addresses that cause.
6. Re-run the failing check, then run a nearby broader check when practical.
7. Record any remaining uncertainty instead of hiding it.
