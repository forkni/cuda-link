# Correctness

## Use this lane when

- Use when the pull request can affect runtime behavior, externally observable results, stored state, data transformations, error handling, concurrency, retries, ordering, or lifecycle behavior.
- Skip when the change has no plausible effect on runtime behavior or externally meaningful state.

## What this lane should catch

- **Incorrect behavior on a reachable path**
  - **Report when:** Changed code produces an incorrect result or control-flow decision for a concrete input, state, configuration, or dependency result.
  - **Evidence:** Show how the path is reached, what behavior is expected, what the changed code does instead, and the resulting consequence.
  - **Do not report:** Hypothetical edge cases without a plausible execution path or observable consequence.

- **Broken state, data, or lifecycle invariants**
  - **Report when:** A changed state transition, write, ordering decision, retry, or concurrent operation can leave state invalid, lose or duplicate work, or violate an invariant relied on by later behavior.
  - **Evidence:** Identify the expected invariant, the changed transition that violates it, and a caller, test, schema, interface, or nearby implementation that establishes the expectation.

- **Unhandled failures or boundary conditions**
  - **Report when:** A reachable error, cancellation, empty or boundary value, fallback, retry, or partial-failure path can now return the wrong result, conceal a required failure, or leave required work incomplete.
  - **Evidence:** Identify the triggering condition, trace the affected path, and explain the concrete incorrect outcome.
