# Slash Command Testing & Optimization Protocol

**Version:** 1.0
**Last Updated:** 2025-11-20
**Purpose:** Systematic framework for testing, evaluating, and optimizing Claude Code slash commands

---

## Table of Contents

1. [Overview & Purpose](#overview--purpose)
2. [Command Quality Framework](#command-quality-framework)
3. [Testing Methodology](#testing-methodology)
4. [Evaluation Criteria](#evaluation-criteria)
5. [Optimization Cycle](#optimization-cycle)
6. [Project-Specific Customization](#project-specific-customization)
7. [Real-World Examples](#real-world-examples)
8. [Command Lifecycle Management](#command-lifecycle-management)
9. [Quick Reference](#quick-reference)

---

## Overview & Purpose

### Why Systematic Testing Matters

Slash commands are powerful automation tools, but **untested commands are dangerous**:

- ❌ **Waste time** with incorrect or incomplete outputs
- ❌ **Provide false confidence** (command runs but produces wrong results)
- ❌ **Accumulate technical debt** (unmaintained commands become obsolete)
- ❌ **Frustrate users** (inconsistent quality erodes trust)

### The Testing → Optimization → Evaluation Cycle

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. TEST → 2. IDENTIFY ISSUES → 3. OPTIMIZE        │
│     ↑                                        ↓      │
│     └────────────── 4. EVALUATE ────────────┘      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Cycle duration:** 2-4 weeks per command
**Expected iterations:** 3-5 cycles to reach "production-ready" quality

### Command Hierarchy

**Global commands** (`.claude/commands/` in user home):

- Apply to ALL projects
- Should be general-purpose
- Require highest quality standards
- Example: `/tdd-cycle`, `/deps-audit`

**Project-specific commands** (`.claude/commands/` in project root):

- Apply to single project
- Can be specialized for project context
- Override global commands if same name
- Example: `/deploy-staging` (specific to your infrastructure)

---

## Command Quality Framework

### Six Core Quality Dimensions

Every slash command should be evaluated across these dimensions:

#### 1. **Clarity** (Is it obvious what the command does?)

**Scoring:**

- ⭐⭐⭐⭐⭐ (5): Purpose clear from name, description, and first use
- ⭐⭐⭐⭐ (4): Requires reading description to understand
- ⭐⭐⭐ (3): Unclear until you've used it once
- ⭐⭐ (2): Confusing even after reading documentation
- ⭐ (1): Misleading or deceptive about functionality

**Tests:**

- [ ] Command name clearly indicates purpose
- [ ] First paragraph explains "what" and "why"
- [ ] Examples provided for common use cases
- [ ] "When to use" vs "when NOT to use" guidance

**Example:**

- ✅ `/deps-audit` - Clear (audits dependencies)
- ❌ `/check-stuff` - Unclear (what stuff?)

#### 2. **Practicality** (Does it solve real problems?)

**Scoring:**

- ⭐⭐⭐⭐⭐ (5): Solves frequent, high-impact problems
- ⭐⭐⭐⭐ (4): Useful for common scenarios
- ⭐⭐⭐ (3): Occasionally helpful
- ⭐⭐ (2): Edge case utility only
- ⭐ (1): Theoretical value, no practical use

**Tests:**

- [ ] Addresses problems you actually have (not hypothetical)
- [ ] Saves significant time vs manual approach
- [ ] Output is immediately actionable
- [ ] Fits into existing workflow

**Anti-pattern:**

- ❌ Commands solving problems you don't have (security theater)
- ❌ Commands for "best practices" you'll never follow

#### 3. **Time-Efficiency** (Fast enough to be useful?)

**Scoring:**

- ⭐⭐⭐⭐⭐ (5): Completes in <5 minutes
- ⭐⭐⭐⭐ (4): Completes in 5-15 minutes
- ⭐⭐⭐ (3): Completes in 15-60 minutes
- ⭐⭐ (2): Completes in 1-4 hours
- ⭐ (1): Takes >4 hours or hangs indefinitely

**Tests:**

- [ ] Document expected completion time upfront
- [ ] Break long operations into phases with checkpoints
- [ ] Provide progress indicators for >5 minute operations
- [ ] Include timeout guidance

**Target times by command type:**

- **Quick checks:** <5 minutes (e.g., lint check)
- **Analysis:** 5-30 minutes (e.g., `/deps-audit`)
- **Implementation:** 30-120 minutes (e.g., `/tdd-cycle` for one feature)
- **Comprehensive:** 2-4 hours (e.g., full system refactoring)

#### 4. **Specificity** (Right level of abstraction?)

**Scoring:**

- ⭐⭐⭐⭐⭐ (5): Perfect level of detail for the task
- ⭐⭐⭐⭐ (4): Slightly too general or specific, but workable
- ⭐⭐⭐ (3): Requires significant adaptation
- ⭐⭐ (2): Wrong abstraction level
- ⭐ (1): Completely mismatched to use case

**Tests:**

- [ ] Commands for Python projects include Python examples
- [ ] Commands for ML projects include PyTorch/FAISS specifics
- [ ] Generic commands avoid technology-specific assumptions
- [ ] Specialization matches user's common tech stack

**Examples:**

- ✅ `/performance-optimization` (Python/ML) - Specific cProfile, PyTorch examples
- ❌ `/performance-optimization` (generic) - Abstract "use profiling tools"

#### 5. **Accuracy** (Produces correct results?)

**Scoring:**

- ⭐⭐⭐⭐⭐ (5): 95%+ accuracy, minimal false positives/negatives
- ⭐⭐⭐⭐ (4): 85-95% accuracy, occasional mistakes
- ⭐⭐⭐ (3): 70-85% accuracy, requires verification
- ⭐⭐ (2): 50-70% accuracy, unreliable
- ⭐ (1): <50% accuracy, misleading

**Tests:**

- [ ] Command recommendations are factually correct
- [ ] Code examples actually work (tested)
- [ ] Security advice doesn't create vulnerabilities
- [ ] Performance claims are evidence-based

**Validation methods:**

- Run code examples to verify they work
- Check external references (docs, CVEs, benchmarks)
- Test on real projects, not toy examples
- Compare output to manual expert analysis

#### 6. **Maintainability** (Will it stay relevant?)

**Scoring:**

- ⭐⭐⭐⭐⭐ (5): References stable APIs, principles over specifics
- ⭐⭐⭐⭐ (4): Mostly stable, minor updates needed quarterly
- ⭐⭐⭐ (3): Requires updates every few months
- ⭐⭐ (2): Rapidly outdated, constant maintenance
- ⭐ (1): Already obsolete

**Tests:**

- [ ] References stable Python/library versions
- [ ] Uses official tools (not deprecated utilities)
- [ ] Focuses on principles over implementation details
- [ ] Includes "Last Updated" date and version

**Example issues:**

- ❌ References `safety` tool (deprecated, moved to paid service)
- ✅ References `pip-audit` (official PyPA tool, actively maintained)

### Overall Quality Score

**Calculation:**

```
Total Score = (Clarity + Practicality + Time-Efficiency + Specificity + Accuracy + Maintainability) / 6
```

**Quality Levels:**

- **4.5-5.0:** Production-ready (ship to global commands)
- **3.5-4.4:** Good (usable, minor improvements needed)
- **2.5-3.4:** Needs work (significant issues)
- **1.5-2.4:** Poor (major revision required)
- **1.0-1.4:** Unusable (complete rewrite)

**Minimum threshold for distribution:** 4.0/5.0

---

## Testing Methodology

### ⚠️ Critical Prerequisite: Restart Required After Command Changes

**IMPORTANT**: Claude Code (VSCode) loads slash commands at startup. If you modify a command file, you **MUST** restart VSCode/Claude Code before testing.

**Symptoms of stale command cache:**

- Changes not reflected when running `/your-command`
- Old version of command still executes
- Template variables like `$ARGUMENTS` not updated

**Fix:**

1. Save all changes to `.claude/commands/*.md` files
2. Completely close VSCode/Claude Code (not just reload window)
3. Reopen VSCode/Claude Code
4. Verify command changes by running the command

**Testing checklist:**

- [ ] All command file changes saved
- [ ] VSCode/Claude Code restarted (closed and reopened)
- [ ] Command file timestamp verified (check file modification date)
- [ ] Ready to begin testing with fresh command cache

**Pro tip**: Keep a scratch file open with test invocations like `/deps-audit` to quickly verify the command loads correctly after restart.

---

### Phase 1: Smoke Testing (5 minutes)

**Goal:** Verify command can be invoked and completes without crashing.

**Steps:**

1. **Invoke command with minimal input:**

   ```bash
   /your-command "basic test input"
   ```

2. **Check for immediate failures:**
   - [ ] Command recognized by Claude
   - [ ] No syntax errors in command file
   - [ ] Command produces some output (not error)
   - [ ] Output completes (doesn't hang indefinitely)

3. **Check output format:**
   - [ ] Output is readable
   - [ ] No truncated sentences/code
   - [ ] Proper markdown formatting

**Pass criteria:** Command runs to completion without errors.

**If smoke test fails:**

- Check `.claude/commands/` file exists and is readable
- Verify markdown syntax is valid
- Check for missing `$ARGUMENTS` variable
- Look for syntax errors in code examples

### Phase 2: Correctness Testing (30-60 minutes)

**Goal:** Verify command produces factually correct, useful output.

#### Test 2.1: Example Code Validation

**For commands with code examples:**

1. **Extract all code examples from command:**

   ```bash
   grep -A 20 '```python' .claude/commands/your-command.md > test_examples.txt
   ```

2. **Test each code example:**

   ```python
   # Create test file for each example
   # Run with: python test_example_1.py

   # Example 1: Should run without errors
   # Example 2: Should produce expected output
   # Example 3: Should handle edge cases
   ```

3. **Record results:**
   - [ ] All examples execute without errors
   - [ ] Examples produce expected outputs
   - [ ] Examples handle edge cases mentioned

**Pass criteria:** 95%+ of code examples work as-is.

#### Test 2.2: Recommendation Validation

**For commands with recommendations:**

1. **Identify all recommendations:**
   - Security advice
   - Performance optimizations
   - Best practices
   - Tool suggestions

2. **Verify each recommendation:**
   - [ ] Is factually accurate (check official docs)
   - [ ] Is still current (tool not deprecated)
   - [ ] Is appropriate for use case
   - [ ] Doesn't introduce vulnerabilities

3. **Test critical recommendations:**

   ```python
   # If command recommends: "Use pip-audit for vulnerability scanning"
   # Test: Install pip-audit, run on test project, verify it works
   ```

**Pass criteria:** All recommendations are accurate and current.

#### Test 2.3: End-to-End Testing

**Test command on real project:**

1. **Select representative test case:**
   - Small Python project (for Python commands)
   - Real codebase (not toy example)
   - Common use case (not edge case)

2. **Execute command:**

   ```bash
   /your-command "implement feature X"
   ```

3. **Evaluate output quality:**
   - [ ] Output addresses the request
   - [ ] Steps are actionable
   - [ ] Code examples are relevant
   - [ ] Advice is project-appropriate

4. **Measure time-to-value:**
   - Time to invoke command: ___ seconds
   - Time to receive output: ___ minutes
   - Time to apply output: ___ minutes
   - **Total time:** ___ minutes

5. **Compare to manual approach:**
   - Manual time estimate: ___ minutes
   - Command time: ___ minutes
   - **Time saved:** _**minutes (**_%)

**Pass criteria:** Command saves >30% time vs manual approach.

### Phase 2.5: Implementation Validation (15-30 minutes)

**Goal:** Verify that command-generated artifacts, scripts, or files match documentation and follow best practices.

**When to use this phase:**

- ✅ Commands that create files (helper scripts, configs, templates)
- ✅ Commands with directory structure setup
- ✅ Commands with automation scripts
- ❌ Commands that only provide advice/documentation

#### Test 2.5.1: Artifact Verification

**For commands that generate files:**

1. **Verify all promised artifacts exist:**

   ```bash
   # Example: /deps-audit creates tools/summarize_audit.py
   ls -lh tools/summarize_audit.py  # Should exist
   ```

   - [ ] All files mentioned in docs are present
   - [ ] Files are in documented locations
   - [ ] File permissions are correct
   - [ ] Files are git-tracked or git-ignored as appropriate

2. **Test artifact functionality:**

   ```python
   # Run generated script to verify it works
   python tools/summarize_audit.py --help  # Should not crash
   python tools/summarize_audit.py test_input.json  # Should produce output
   ```

   - [ ] Scripts execute without errors
   - [ ] Scripts handle documented use cases
   - [ ] Scripts produce expected output format
   - [ ] Error handling works as documented

3. **Validate artifact quality:**
   - [ ] Code follows project style guide
   - [ ] Includes proper error handling
   - [ ] Has helpful error messages
   - [ ] Works on documented platforms (Windows/Linux/Mac)

**Pass criteria:** All artifacts work as documented, 100% platform compatibility for documented platforms.

#### Test 2.5.2: Documentation Drift Detection

**Check for mismatches between docs and implementation:**

1. **Example output verification:**

   ```bash
   # If command shows example output with emoji (📦, ⚠️)
   # But actual output uses ASCII ([PACKAGE], [VULN])
   # → Documentation drift detected
   ```

   - [ ] Example outputs match actual outputs
   - [ ] Code snippets work as shown
   - [ ] File paths match actual locations
   - [ ] Command invocations work as documented

2. **Feature completeness:**
   - [ ] All documented features implemented
   - [ ] No undocumented features (hidden behaviors)
   - [ ] All documented options work
   - [ ] All documented flags recognized

3. **Version consistency:**
   - [ ] Tool versions in docs match recommended versions
   - [ ] API examples use current API syntax
   - [ ] Deprecated features flagged in docs

**Pass criteria:** <3% documentation drift (≤1 issue per ~30 documented items).

#### Test 2.5.3: Directory Structure Validation

**For commands that create/organize directories:**

1. **Verify directory structure:**

   ```bash
   tree audit_reports/  # Should match documented structure
   # Expected:
   # audit_reports/
   # ├── README.md
   # ├── YYYY-MM-DD-audit.json
   # └── archive/ (optional)
   ```

   - [ ] All documented directories exist
   - [ ] Directory names match exactly
   - [ ] README files present where documented
   - [ ] Git-ignore rules applied correctly

2. **Test artifact organization:**
   - [ ] Test files go to correct directories
   - [ ] No pollution of project root
   - [ ] Archives/backups organized properly
   - [ ] Cleanup instructions work

3. **Verify git integration:**

   ```bash
   git status  # Should not show temp files
   git check-ignore audit_reports/*.json  # Should be ignored
   ```

   - [ ] Git-ignore rules prevent committing sensitive data
   - [ ] README files tracked by git
   - [ ] Directory structure committed (empty .gitkeep if needed)

**Pass criteria:** Directory structure matches docs 100%, no project root pollution, git-ignore rules working.

### Phase 3: Usability Testing (60 minutes)

**Goal:** Verify command is actually helpful in real-world usage.

#### Test 3.1: First-Time User Experience

**Simulate new user:**

1. **Read command without context:**
   - [ ] Purpose is clear within 30 seconds
   - [ ] When to use is obvious
   - [ ] Examples are provided
   - [ ] Limitations are documented

2. **Try to use command:**
   - [ ] First invocation succeeds
   - [ ] Output is self-explanatory
   - [ ] Next steps are clear
   - [ ] No dead-ends or confusion

**Pass criteria:** New user can successfully use command within 5 minutes.

#### Test 3.2: Expert User Efficiency

**Test with experienced user:**

1. **Measure efficiency:**
   - Time to invoke: ___ seconds
   - Time to parse output: ___ seconds
   - Time to act on output: ___ minutes
   - **Total workflow time:** ___ minutes

2. **Identify friction points:**
   - Unnecessary steps?
   - Redundant information?
   - Missing keyboard shortcuts?
   - Could be more concise?

**Pass criteria:** Expert can complete workflow in <50% of manual time.

#### Test 3.3: Error Handling

**Test failure modes:**

1. **Invalid input:**

   ```bash
   /your-command ""
   /your-command "impossible request"
   /your-command "ambiguous input"
   ```

   - [ ] Provides helpful error messages
   - [ ] Suggests valid alternatives
   - [ ] Doesn't crash or hang

2. **Edge cases:**
   - Empty project (no files)
   - Large project (1000+ files)
   - Missing dependencies
   - Unexpected file structure

**Pass criteria:** Command handles errors gracefully with helpful messages.

### Phase 4: Performance Testing (30 minutes)

**Goal:** Verify command completes in reasonable time.

#### Test 4.1: Benchmark Command Execution

**Measure execution time:**

```python
import time

iterations = 5
times = []

for i in range(iterations):
    start = time.time()
    # Execute: /your-command "test input"
    elapsed = time.time() - start
    times.append(elapsed)

print(f"Average: {sum(times)/len(times):.2f}s")
print(f"Min: {min(times):.2f}s")
print(f"Max: {max(times):.2f}s")
```

**Target times:**

- Quick commands: <30 seconds average
- Analysis commands: <5 minutes average
- Implementation commands: <30 minutes average

**Pass criteria:** 95th percentile execution time meets target.

#### Test 4.2: Scalability Testing

**Test with varying project sizes:**

| Project Size | Files | Execution Time | Pass/Fail |
|--------------|-------|----------------|-----------|
| Tiny | 1-10 | ___ seconds | |
| Small | 10-100 | ___ seconds | |
| Medium | 100-500 | ___ minutes | |
| Large | 500-2000 | ___ minutes | |

**Pass criteria:** Execution time scales linearly (not exponentially).

### Phase 5: Security Remediation (30-60 minutes)

**Goal:** Apply security fixes discovered during command testing, demonstrating dogfooding (using your own tools).

**When to use this phase:**

- ✅ After testing `/deps-audit` or security-related commands
- ✅ When vulnerabilities are discovered in command examples
- ✅ When command recommends security updates
- ❌ For commands unrelated to security

**Philosophy:** If your command finds security issues, you must fix them before shipping the command. This proves the command works and demonstrates best practices.

#### Test 5.1: Vulnerability Remediation Workflow

**For security audit commands:**

1. **Save baseline audit:**

   ```bash
   .venv/Scripts/pip-audit --format json > audit_reports/before-fixes-YYYY-MM-DD.json
   ```

   - [ ] Baseline captured before any changes
   - [ ] Baseline includes full vulnerability details
   - [ ] Baseline saved to git-ignored location

2. **Create security fix branch:**

   ```bash
   git checkout -b security-fixes-YYYY-MM-DD
   ```

   - [ ] Branch created from clean development/main
   - [ ] Branch name includes date for tracking
   - [ ] Branch isolated from other work

3. **Apply fixes ONE AT A TIME:**

   ```bash
   # Fix 1: Update authlib
   pip install --upgrade authlib==1.6.5
   python -m pytest tests/

   # Fix 2: Update pip
   pip install --upgrade pip==25.3
   python -m pytest tests/

   # Continue for each CVE...
   ```

   - [ ] Each fix applied individually
   - [ ] Tests run after EACH fix
   - [ ] Rollback plan ready if tests fail
   - [ ] Fix version documented in commit message

4. **Save post-fix audit:**

   ```bash
   .venv/Scripts/pip-audit --format json > audit_reports/after-fixes-YYYY-MM-DD.json
   ```

   - [ ] All CVEs resolved or documented as deferred
   - [ ] Audit saved for comparison
   - [ ] Remaining issues explained in commit message

5. **Update dependency constraints:**

   ```toml
   # pyproject.toml
   [project.dependencies]
   authlib = ">=1.6.5"  # Fixes CVE-2025-59420 (JWS bypass)
   pip = ">=25.3"       # Fixes CVE-2025-8869 (path traversal)
   ```

   - [ ] Minimum versions set to patched versions
   - [ ] CVE IDs documented in comments
   - [ ] Version constraints prevent regression

**Pass criteria:** All discovered CVEs fixed or documented as deferred with justification.

#### Test 5.2: Testing Post-Remediation

**Verify fixes don't break functionality:**

1. **Run full test suite:**

   ```bash
   pytest tests/ -v --cov
   ```

   - [ ] All unit tests pass
   - [ ] All integration tests pass
   - [ ] Code coverage maintained or improved
   - [ ] No new test failures introduced

2. **Test critical workflows:**
   - [ ] Main application entry points work
   - [ ] ML functionality intact (if applicable)
   - [ ] CUDA/GPU access working (if applicable)
   - [ ] CLI commands functional
   - [ ] API endpoints responding (if applicable)

3. **Verify no dependency conflicts:**

   ```bash
   pip check
   ```

   - [ ] No broken dependencies reported
   - [ ] All packages compatible
   - [ ] No circular dependency issues

**Pass criteria:** 100% test pass rate, no functionality regression.

#### Test 5.3: Documentation of Deferred Fixes

**For CVEs that can't be fixed immediately:**

1. **Create deferred CVE log:**

   ```markdown
   # audit_reports/deferred-cves-YYYY-MM-DD.md

   ## Deferred CVE Fixes

   ### CVE-2025-XXXXX (package-name X.Y.Z)
   - **Severity:** High
   - **Reason for deferral:** Fixed version (X.Y.Z+1) breaks OAuth flow (see issue #123)
   - **Mitigation:** Running locally only, no untrusted inputs processed
   - **Review date:** YYYY-MM-DD (3 months, check for stable fix)
   - **Monitoring:** Added to quarterly review checklist
   ```

   - [ ] All deferred CVEs documented
   - [ ] Deferral reason provided
   - [ ] Mitigation strategy explained
   - [ ] Review date set (max 90 days)
   - [ ] Added to maintenance schedule

2. **Risk assessment:**
   - [ ] Exploitation likelihood assessed (Low/Medium/High)
   - [ ] Impact severity documented
   - [ ] Acceptable risk level justified
   - [ ] Stakeholders informed (if applicable)

**Pass criteria:** All deferred fixes documented with clear justification and review schedule.

### Phase 6: Regression Testing (15 minutes)

**Goal:** Verify that optimization changes didn't break previously working functionality.

**When to use this phase:**

- ✅ After every optimization cycle
- ✅ After fixing bugs from previous testing
- ✅ Before committing updated command
- ✅ Before distributing command to users

**Philosophy:** Every change risks introducing new bugs. Regression testing catches these before users do.

#### Test 6.1: Re-run Core Test Cases

**Execute abbreviated test suite:**

1. **Smoke test (2 minutes):**

   ```bash
   /your-command "basic test input"
   ```

   - [ ] Command still executes
   - [ ] No new syntax errors
   - [ ] Output format unchanged (unless intentionally modified)
   - [ ] Completion time similar to before

2. **Critical path testing (5 minutes):**

   ```bash
   # Test main use case from Phase 2 testing
   /your-command "same input as Cycle 1"
   ```

   - [ ] Core functionality works
   - [ ] Previously passing scenarios still pass
   - [ ] Output quality maintained or improved
   - [ ] No new errors introduced

3. **Edge case verification (5 minutes):**

   ```bash
   # Test previously-fixed edge cases
   /your-command ""              # Empty input
   /your-command "edge case X"   # Known edge case
   ```

   - [ ] Previously fixed bugs still fixed
   - [ ] Error handling still graceful
   - [ ] Edge cases don't regress

**Pass criteria:** All previously passing tests still pass, no new failures introduced.

#### Test 6.2: Optimization Impact Verification

**Confirm optimizations had intended effect:**

1. **Compare before/after metrics:**

   | Metric | Before Optimization | After Optimization | Change |
   |--------|-------------------|-------------------|--------|
   | Quality Score | 3.8/5.0 | 4.5/5.0 | +0.7 ✅ |
   | Execution Time | 2.1s | 1.8s | -14% ✅ |
   | False Positives | 3 | 0 | -100% ✅ |
   | User Friction | 4 issues | 1 issue | -75% ✅ |

   - [ ] Quality score improved or maintained
   - [ ] No performance regression
   - [ ] Bugs actually fixed
   - [ ] Target metrics achieved

2. **Verify specific fixes:**

   ```bash
   # If Cycle 1 found: "Unicode crash on Windows"
   # Verify: No Unicode error when running helper script
   python tools/summarize_audit.py audit_reports/test.json
   # Expected: Successful execution, ASCII output
   ```

   - [ ] Each documented issue resolved
   - [ ] Fix confirmed by re-testing failure scenario
   - [ ] Root cause addressed (not just symptom)

**Pass criteria:** All optimization goals achieved, no regressions introduced.

#### Test 6.3: Version Control Validation

**Verify git hygiene:**

1. **Check commit quality:**

   ```bash
   git log -1 --format="%s%n%b"  # Last commit message
   ```

   - [ ] Commit message follows conventional commits format
   - [ ] Commit explains what changed and why
   - [ ] Commit references any issues/tickets
   - [ ] Commit is atomic (single logical change)

2. **Verify no sensitive data committed:**

   ```bash
   git diff --staged | grep -E "(password|token|key|secret)"
   ```

   - [ ] No hardcoded credentials
   - [ ] No API keys or tokens
   - [ ] No sensitive audit outputs
   - [ ] Git-ignore rules working

3. **Check file organization:**

   ```bash
   git status --short
   ```

   - [ ] No untracked files in project root
   - [ ] Test artifacts in proper directories
   - [ ] No leftover temp files
   - [ ] Clean working tree (or explained)

**Pass criteria:** Clean git history, no sensitive data, proper file organization.

---

## Evaluation Criteria

### Output Quality Metrics

#### Metric 1: Actionability Score

**Definition:** Percentage of output that can be directly acted upon.

**Measurement:**

1. Count total recommendations/steps in output
2. Count actionable items (concrete code, commands, steps)
3. Calculate: `Actionable / Total * 100%`

**Target:** >80% actionability

**Example:**

```markdown
❌ Low actionability (30%):
"Consider optimizing your code for better performance. Look into caching strategies and efficient algorithms."

✅ High actionability (90%):
"Add @lru_cache decorator to expensive_function() on line 45:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_function(input):
    ...
```

This will cache up to 1000 results, reducing repeated computations by 80%."

```

#### Metric 2: Accuracy Rate

**Definition:** Percentage of recommendations that are factually correct.

**Measurement:**
1. Select 20 random recommendations from output
2. Verify each against authoritative sources
3. Calculate: `Correct / Total * 100%`

**Target:** >95% accuracy

**Verification sources:**
- Official documentation
- Peer-reviewed benchmarks
- Expert review
- Test execution

#### Metric 3: Time-to-Value

**Definition:** Time from command invocation to actionable results.

**Measurement:**
```

Time-to-Value = Command_Execution_Time + Output_Parsing_Time

```

**Target times by command type:**
- Quick checks: <1 minute
- Analysis: <10 minutes
- Implementation: <60 minutes

#### Metric 4: False Positive Rate

**Definition:** Percentage of recommendations that are wrong or misleading.

**Measurement:**
1. Apply all recommendations from command
2. Test results (does code work? do tests pass?)
3. Count failures: `Failures / Total_Recommendations * 100%`

**Target:** <5% false positive rate

**Common false positives:**
- Suggesting deprecated packages
- Recommending incompatible versions
- Security advice that introduces vulnerabilities
- Performance "optimizations" that slow things down

### User Satisfaction Indicators

#### Indicator 1: Repeat Usage Rate

**Definition:** Do users invoke command again after first use?

**Measurement:**
- Track command invocations over 30 days
- Count unique users
- Calculate: `Users_Invoking_>1_Time / Total_Users * 100%`

**Target:** >60% repeat usage

**Interpretation:**
- <30%: Command not useful (one-and-done)
- 30-60%: Occasionally useful
- 60-80%: Regularly useful
- >80%: Essential tool

#### Indicator 2: Net Promoter Score (NPS)

**Definition:** Would users recommend this command?

**Survey question:**
> "On a scale of 0-10, how likely are you to recommend `/your-command` to a colleague?"

**Calculation:**
- Promoters (9-10): Count as +1
- Passives (7-8): Count as 0
- Detractors (0-6): Count as -1
- NPS = (Promoters - Detractors) / Total * 100

**Target:** NPS >50 (excellent), NPS 0-49 (good), NPS <0 (poor)

#### Indicator 3: Time Savings (Self-Reported)

**Survey question:**
> "How much time did `/your-command` save you compared to doing this manually?"

**Responses:**
- Saved >50% time: Excellent
- Saved 20-50% time: Good
- Saved 0-20% time: Marginal
- No time saved: Poor
- Took longer: Failure

**Target:** >70% of users report saving >20% time

---

## Optimization Cycle

### Step 1: Identify Issues (Systematic Problem Categorization)

#### Issue Categories

**Category A: Clarity Issues**
- Symptoms: Users confused about command purpose
- Evidence: Low first-use success rate, repeated questions
- Examples:
  - Unclear command name
  - Missing "when to use" guidance
  - No examples provided

**Category B: Accuracy Issues**
- Symptoms: Recommendations don't work, code fails
- Evidence: High false positive rate, user complaints
- Examples:
  - Code examples with syntax errors
  - Outdated tool recommendations
  - Factually incorrect advice

**Category C: Performance Issues**
- Symptoms: Command too slow, times out
- Evidence: Long execution times, user abandonment
- Examples:
  - Command hangs for large projects
  - Generates excessive output
  - Unnecessary agent orchestration

**Category D: Specificity Issues**
- Symptoms: Output too generic or too specialized
- Evidence: Low actionability, requires heavy adaptation
- Examples:
  - Generic advice not applicable to user's stack
  - Too many enterprise assumptions
  - Missing context for user's use case

**Category E: Usability Issues**
- Symptoms: Hard to use, friction in workflow
- Evidence: Low repeat usage, users abandon mid-task
- Examples:
  - Too many steps required
  - Output format hard to parse
  - Missing error handling

#### Issue Prioritization Matrix

|  | High Impact | Medium Impact | Low Impact |
|--|-------------|---------------|------------|
| **Quick Fix** | 🔴 **P0:** Fix immediately | 🟡 **P1:** Fix this week | 🟢 **P2:** Fix this month |
| **Medium Effort** | 🟡 **P1:** Fix this week | 🟢 **P2:** Fix this month | ⚪ **P3:** Backlog |
| **High Effort** | 🟡 **P1:** Fix this week | 🟢 **P2:** Fix this month | ⚪ **P3:** Backlog |

**Priority definitions:**
- **P0 (Critical):** Blocks command usage entirely
- **P1 (High):** Significantly degrades experience
- **P2 (Medium):** Minor annoyance, workarounds exist
- **P3 (Low):** Nice-to-have improvement

### Step 2: Prioritize Improvements (Impact vs Effort Matrix)

#### Impact Assessment

**High Impact** (affects >50% of users or >50% of use cases):
- Fixing factual errors
- Correcting broken code examples
- Adding missing "when to use" guidance
- Reducing execution time by >50%

**Medium Impact** (affects 20-50% of users):
- Improving error messages
- Adding more examples
- Better output formatting
- Moderate performance improvements

**Low Impact** (affects <20% of users):
- Polish and refinement
- Edge case handling
- Advanced features
- Cosmetic improvements

#### Effort Estimation

**Quick Fix** (<2 hours):
- Fixing typos
- Updating deprecated tool references
- Adding examples
- Clarifying language

**Medium Effort** (2-8 hours):
- Restructuring command flow
- Adding new sections
- Comprehensive testing
- Major example additions

**High Effort** (>8 hours):
- Complete command rewrite
- Architecture changes
- Multi-person review required
- Extensive validation needed

### Step 3: Implement Changes (Focused Iterations)

#### Iteration Process

**Iteration 1: Fix Critical Issues (P0)**
- Focus: Make command usable
- Duration: 1-2 days
- Target: Move from "unusable" to "barely usable"

**Iteration 2: Address High-Priority Issues (P1)**
- Focus: Make command reliable
- Duration: 3-5 days
- Target: Move from "barely usable" to "good"

**Iteration 3: Polish and Refine (P2)**
- Focus: Make command excellent
- Duration: 1-2 weeks
- Target: Move from "good" to "production-ready"

**Iteration 4+: Continuous Improvement (P3)**
- Focus: Optimize and maintain
- Duration: Ongoing
- Target: Keep command current and relevant

#### Change Documentation

**For each change, document:**

```markdown
## Change #N: [Brief Description]

**Issue:** [What problem does this solve?]
**Category:** [Clarity/Accuracy/Performance/Specificity/Usability]
**Priority:** [P0/P1/P2/P3]

**Before:**
[Describe or show old behavior]

**After:**
[Describe or show new behavior]

**Testing:**
- [ ] Smoke tested
- [ ] Code examples verified
- [ ] End-to-end tested
- [ ] Performance measured

**Impact:**
- Time saved: +X minutes
- Accuracy improved: +Y%
- User satisfaction: +Z points
```

### Step 4: Validate Improvements (Before/After Comparison)

#### Validation Checklist

**For each change:**

1. **Re-run all tests:**
   - [ ] Phase 1: Smoke testing
   - [ ] Phase 2: Correctness testing
   - [ ] Phase 3: Usability testing
   - [ ] Phase 4: Performance testing

2. **Measure improvement:**
   - Quality score before: ___/5.0
   - Quality score after: ___/5.0
   - **Improvement:** +___ points

3. **Verify no regressions:**
   - [ ] Previous functionality still works
   - [ ] No new errors introduced
   - [ ] Performance not degraded
   - [ ] Documentation still accurate

4. **Get user feedback:**
   - [ ] Test with 2-3 users
   - [ ] Collect satisfaction scores
   - [ ] Document any remaining issues

**Acceptance criteria:**

- Quality score increases by ≥0.5 points
- No regressions introduced
- User satisfaction improves

### Step 5: Document Learnings (Build Knowledge Base)

#### Learning Documentation Template

```markdown
# Command Optimization: [Command Name]

## Initial Assessment
- **Quality Score:** ___/5.0
- **Primary Issues:** [List top 3 issues]
- **User Feedback:** [Summary of complaints/requests]

## Optimization Process

### Iteration 1
- **Date:** YYYY-MM-DD
- **Changes:** [What was changed]
- **Results:** [What improved]
- **Quality Score:** ___/5.0

### Iteration 2
[Repeat for each iteration]

## Key Learnings

### What Worked Well
- [Learning 1]
- [Learning 2]
- [Learning 3]

### What Didn't Work
- [Mistake 1]
- [Mistake 2]

### Reusable Patterns
- [Pattern 1: When X, do Y]
- [Pattern 2: Avoid Z in context A]

## Final State
- **Quality Score:** ___/5.0
- **Improvement:** +___ points
- **Status:** [Production-ready / Needs more work]

## Recommendations for Similar Commands
[Advice for creating/optimizing similar commands]
```

---

## Project-Specific Customization

### When to Customize Commands

#### Customization Triggers

**Trigger 1: Technology Stack Mismatch**

- Global command assumes Node.js, you use Python
- Global command assumes REST APIs, you use GraphQL
- Global command assumes SQL, you use NoSQL

**Action:** Create project-specific version with correct tech stack.

**Trigger 2: Team Conventions**

- Global command suggests standard approach, your team uses different conventions
- Code style doesn't match your linting rules
- Different test framework preferences

**Action:** Fork command and adapt to team conventions.

**Trigger 3: Domain-Specific Requirements**

- ML projects need GPU/CUDA considerations
- Financial projects need compliance checks
- Healthcare projects need HIPAA considerations

**Action:** Add domain-specific guidance to command.

**Trigger 4: Scale Differences**

- Global command optimized for small projects
- Your project has 1M+ lines of code
- Performance characteristics completely different

**Action:** Optimize command for your scale.

### Customization Process

#### Step 1: Copy Global Command

```bash
# Copy global command to project directory
cp ~/.claude/commands/deps-audit.md .claude/commands/deps-audit.md

# Or start from scratch with template
cat > .claude/commands/my-custom-command.md << 'EOF'
---
model: claude-sonnet-4-0
---

# Custom Command for [Project Name]

**Purpose:** [What this command does]
**Based on:** [Global command name, if any]
**Customizations:** [What's different from global]

## Instructions

$ARGUMENTS

[Your custom instructions here]
EOF
```

#### Step 2: Document Customizations

**Add customization header:**

```markdown
---
model: claude-sonnet-4-0
---

# Dependency Audit (Python/ML - Project Specific)

**Base Command:** `deps-audit` (global)
**Customized For:** claude-context-local (semantic code search)
**Last Updated:** 2025-11-20

## Project-Specific Customizations

### 1. Focus on ML Dependencies
- PyTorch 2.6+ with CUDA 12.4 compatibility
- Transformers 4.51+ with security patches
- FAISS index format stability
- Tree-sitter parser security

### 2. Custom Update Strategy
- NEVER auto-update PyTorch (CUDA compatibility)
- Test transformers in isolation before updating
- Pin major versions: torch>=2.6.0,<3.0.0

### 3. Project-Specific Checks
- Verify 82 tests pass after updates
- Check semantic search quality didn't degrade
- Validate FAISS index compatibility

[Rest of command...]
```

#### Step 3: Test Customized Command

**Run full testing protocol:**

- [ ] Phase 1: Smoke testing
- [ ] Phase 2: Correctness testing (especially project-specific parts)
- [ ] Phase 3: Usability testing (with team members)
- [ ] Phase 4: Performance testing

**Verify customizations:**

- [ ] Project-specific examples work
- [ ] Team conventions followed
- [ ] Domain requirements addressed
- [ ] No conflicts with global command

#### Step 4: Maintain Custom Command

**Synchronization strategy:**

```markdown
## Maintenance Log

### 2025-11-20 - Created custom version
- Based on global deps-audit v1.0
- Added PyTorch/ML focus
- Documented 3 key customizations

### 2025-12-15 - Updated from global
- Global command updated to v1.1
- Merged: New pip-audit examples
- Kept: PyTorch-specific guidance
- Status: In sync with global

### 2026-01-10 - Project-specific update
- Added GPT-4 fine-tuning checks
- Updated for PyTorch 2.7
- Status: Diverged from global (intentional)
```

**Review schedule:**

- Check global command monthly for updates
- Merge non-conflicting improvements
- Document reasons for keeping differences

### Customization Patterns

#### Pattern 1: Add Project Context

**Global command:**

```markdown
Run vulnerability scan on dependencies.
```

**Customized:**

```markdown
Run vulnerability scan on dependencies.

**For this project:**
- Focus on 38 direct dependencies (not transitive)
- Critical packages: torch, transformers, faiss-cpu
- Check against project's Python 3.11 requirement
- Verify compatibility with CUDA 12.4
```

#### Pattern 2: Add Team Workflows

**Global command:**

```markdown
After finding vulnerabilities, update packages.
```

**Customized:**

```markdown
After finding vulnerabilities, update packages.

**Team workflow:**
1. Create feature branch: `deps/update-YYYY-MM-DD`
2. Update in isolated venv: `python -m venv test_env`
3. Run full test suite: 82 tests must pass
4. Check ML performance: No embedding quality regression
5. Create PR using: `scripts/git/commit_enhanced.bat`
6. Tag 2 reviewers: @ml-lead @devops-lead
```

#### Pattern 3: Add Domain Constraints

**Global command:**

```markdown
Review license compatibility.
```

**Customized:**

```markdown
Review license compatibility.

**Medical device compliance (FDA):**
- All dependencies must be GPL-3.0 compatible
- Document all Apache-2.0 licenses (patent clauses)
- Flag any AGPL dependencies (network copyleft)
- Maintain license audit trail for FDA submission
- Generate SBOM (Software Bill of Materials) quarterly
```

---

## Real-World Examples

### Example 1: deps-audit Command Optimization

#### Initial Assessment

**Version 0.1** (From external repository)

```markdown
**Quality Score:** 2.8/5.0

Breakdown:
- Clarity: 4/5 (name is clear)
- Practicality: 2/5 (enterprise security theater)
- Time-Efficiency: 1/5 (weekly scans, 40+ hours/month maintenance)
- Specificity: 2/5 (generic, not ML-focused)
- Accuracy: 4/5 (technically correct but impractical)
- Maintainability: 3/5 (references deprecated 'safety' tool)
```

**Primary Issues:**

1. **Enterprise Security Theater** (Practicality: P0)
   - Model integrity checking (HTTPS already provides this)
   - Weekly automated scans (overkill for local tools)
   - Slack notifications (over-engineering)
   - Supply chain monitoring for PyTorch (would be headline news)

2. **Wrong Tools** (Maintainability: P0)
   - Recommends `safety` (deprecated, moved to paid service)
   - Should recommend `pip-audit` (official PyPA tool)

3. **Not ML-Specific** (Specificity: P1)
   - No PyTorch/CUDA compatibility guidance
   - No ML dependency update strategies
   - Missing transformers security considerations

#### Optimization Process

**Iteration 1: Remove Security Theater** (2 hours)

```diff
- ### Model Integrity Checking
- Implement hash verification for HuggingFace downloads
- Set up model signing infrastructure
- [600 lines of enterprise security code]

+ ## Phase 5: Practical Security Hardening (Optional)
+ ### Only Do This If You Actually Need It
+ - Local development tool? Skip this.
+ - Public-facing API? Consider basic validation.
```

**Results:**

- Reduced from 776 lines → 379 lines (-51%)
- Time commitment: 40+ hours/month → 30 min setup + 15 min quarterly
- Practicality: 2/5 → 5/5

**Iteration 2: Fix Tools** (1 hour)

```diff
- pip install safety
- safety check --json

+ pip install pip-audit  # Official PyPA tool
+ pip-audit --desc       # Queries OSV database
```

**Results:**

- Maintainability: 3/5 → 5/5
- Accuracy: 4/5 → 5/5

**Iteration 3: Add ML Focus** (3 hours)

```markdown
+ ### PyTorch/CUDA Compatibility
+
+ **⚠️ CRITICAL for ML Projects:**
+
+ ```python
+ python -c "
+ import torch
+ print(f'PyTorch Version: {torch.__version__}')
+ print(f'CUDA Available: {torch.cuda.is_available()}')
+ if torch.cuda.is_available():
+     print(f'CUDA Version: {torch.version.cuda}')
+ "
+ ```
+
+ ### ML Dependency Update Strategy
+
+ 🚫 **NEVER auto-update these packages:**
+ - `torch`, `torchvision`, `torchaudio` (CUDA compatibility)
+ - `transformers` (model behavior changes)
+ - `faiss-cpu` / `faiss-gpu` (index format changes)
```

**Results:**

- Specificity: 2/5 → 5/5
- Practicality: 5/5 (already improved) → 5/5

#### Final Assessment

**Version 1.0** (Optimized)

```markdown
**Quality Score:** 4.8/5.0 (+2.0 points, +71% improvement)

Breakdown:
- Clarity: 4/5 → 5/5 (+1, added "when NOT to use")
- Practicality: 2/5 → 5/5 (+3, removed security theater)
- Time-Efficiency: 1/5 → 5/5 (+4, 30min vs 40+ hours)
- Specificity: 2/5 → 5/5 (+3, ML-focused)
- Accuracy: 4/5 → 5/5 (+1, fixed deprecated tools)
- Maintainability: 3/5 → 4/5 (+1, still need quarterly reviews)
```

**Status:** ✅ Production-ready (4.8 > 4.0 threshold)

#### Key Learnings

**What Worked:**

1. **Critical analysis first** - Identified security theater before coding
2. **Real-world focus** - "What would I actually use?" test
3. **ML-specific examples** - PyTorch, FAISS, transformers guidance
4. **Time-boxing** - "30 min setup + 15 min quarterly" target

**What Didn't Work:**

1. **Initial acceptance** - Almost shipped enterprise version unchanged
2. **Trusting external sources** - External repo had different use case
3. **Skipping testing** - Found `safety` deprecation during review, not testing

**Reusable Pattern:**
> **Pattern: "Security Theater Detection"**
>
> When reviewing security commands, ask:
>
> 1. What's the actual attack vector? (HTTPS compromise? Very unlikely)
> 2. What's the blast radius? (Local tool vs internet service)
> 3. What's the cost/benefit? (40 hours/month vs catching 1 CVE/year)
>
> If answers are "unlikely", "local", "terrible ratio" → it's security theater.

### Example 2: tdd-cycle Command Simplification

#### Initial Assessment

**Version 0.1** (From external repository)

```markdown
**Quality Score:** 3.2/5.0

Breakdown:
- Clarity: 3/5 (too many phases confusing)
- Practicality: 4/5 (TDD is practical, but over-complicated)
- Time-Efficiency: 2/5 (12 phases takes hours)
- Specificity: 3/5 (generic, needs pytest examples)
- Accuracy: 4/5 (correct but impractical)
- Maintainability: 4/5 (stable concepts)
```

**Primary Issues:**

1. **Over-Orchestration** (Time-Efficiency: P0)
   - 12 phases with agent coordination
   - Architecture review PER TEST (overkill)
   - Integration tests as separate workflow
   - "Performance and edge case tests" phase

2. **Missing Practical Guidance** (Specificity: P1)
   - No pytest examples
   - No "when NOT to use TDD"
   - No incremental approach guidance
   - Abstract test patterns

3. **Intimidating Complexity** (Clarity: P1)
   - "Use Task tool with subagent_type=" (unfamiliar)
   - Extended thinking annotation (confusing)
   - Coverage thresholds without context
   - Success criteria too strict (100% code test-first)

#### Optimization Process

**Iteration 1: Simplify to Core TDD** (4 hours)

```diff
- ## Phase 1: Test Specification and Design
- ### 1. Requirements Analysis
- - Use Task tool with subagent_type="architect-review"
- ### 2. Test Architecture Design
- - Use Task tool with subagent_type="test-automator"
-
- ## Phase 2: RED - Write Failing Tests
- ### 3. Write Unit Tests (Failing)
- - Use Task tool with subagent_type="test-automator"
- ### 4. Verify Test Failure
- - Use Task tool with subagent_type="code-reviewer"

+ ## Phase 1: RED - Write a Failing Test
+
+ ### Step 1: Write ONE Failing Test
+
+ **Focus**: Write the simplest test that specifies desired behavior.
+
+ ```python
+ import pytest
+
+ def test_search_returns_relevant_results():
+     # Arrange
+     engine = SearchEngine()
+     query = "python decorators"
+
+     # Act
+     results = engine.search(query)
+
+     # Assert
+     assert len(results) > 0
+ ```
```

**Results:**

- Clarity: 3/5 → 5/5 (+2, concrete examples)
- Time-Efficiency: 2/5 → 5/5 (+3, 30-60 min vs hours)
- Reduced from 203 lines → 437 lines (+115% but more practical)

**Iteration 2: Add Pytest Specifics** (2 hours)

```markdown
+ ## Pytest Tips for TDD
+
+ ### Useful Pytest Commands
+
+ ```bash
+ # Run specific test
+ pytest tests/test_search.py::test_search_returns_results -v
+
+ # Stop on first failure
+ pytest -x
+
+ # Run with coverage
+ pytest --cov=src --cov-report=term-missing tests/
+ ```
+
+ ### Useful Pytest Fixtures
+
+ ```python
+ @pytest.fixture
+ def search_engine():
+     """Shared search engine instance for tests"""
+     return SearchEngine()
+ ```
```

**Results:**

- Specificity: 3/5 → 5/5 (+2, pytest examples)
- Practicality: 4/5 → 5/5 (+1, immediately usable)

**Iteration 3: Add "When NOT to Use"** (1 hour)

```markdown
+ ## When to Use TDD
+
+ ✅ **Good use cases:**
+ - Complex business logic (algorithms, state machines)
+ - Critical functionality (authentication, payment processing)
+ - Bug fixes (reproduce bug with test first)
+
+ ❌ **Don't use TDD for:**
+ - UI/UX experimentation (design unclear)
+ - Prototyping/spikes (exploring solutions)
+ - Trivial getters/setters
+ - One-off scripts
```

**Results:**

- Clarity: 5/5 (already improved) → 5/5
- Practicality: 5/5 (already improved) → 5/5

#### Final Assessment

**Version 1.0** (Optimized)

```markdown
**Quality Score:** 4.8/5.0 (+1.6 points, +50% improvement)

Breakdown:
- Clarity: 3/5 → 5/5 (+2, concrete examples)
- Practicality: 4/5 → 5/5 (+1, pytest specifics)
- Time-Efficiency: 2/5 → 5/5 (+3, streamlined)
- Specificity: 3/5 → 5/5 (+2, Python/pytest)
- Accuracy: 4/5 → 5/5 (+1, tested examples)
- Maintainability: 4/5 → 4/5 (stable)
```

**Status:** ✅ Production-ready (4.8 > 4.0 threshold)

#### Key Learnings

**What Worked:**

1. **Simplification** - 12 phases → 3 core phases
2. **Concrete examples** - Real pytest code, not abstract patterns
3. **Practical guidance** - "When NOT to use" prevents misuse
4. **Incremental approach** - One test at a time is actually how TDD works

**What Didn't Work:**

1. **Agent orchestration** - Too complex for simple TDD workflow
2. **Architecture reviews** - Overkill for writing tests
3. **Strict thresholds** - "100% test-first" intimidates beginners

**Reusable Pattern:**
> **Pattern: "Simplification Test"**
>
> When reviewing complex commands, ask:
>
> 1. Can I explain this in 3 sentences?
> 2. Can a beginner use this without reading 200 lines?
> 3. Are there concrete examples in the first 100 lines?
>
> If any answer is "no" → simplify.

---

## Command Lifecycle Management

### Version Control for Commands

#### Semantic Versioning for Commands

**Version format:** `MAJOR.MINOR.PATCH`

**Examples:**

- `1.0.0` - Initial production release
- `1.1.0` - New feature added (backward compatible)
- `1.0.1` - Bug fix (no functionality changes)
- `2.0.0` - Breaking change (requires user updates)

**Version header in command:**

```markdown
---
model: claude-sonnet-4-0
version: 1.2.3
last-updated: 2025-11-20
---

# Command Name

**Version:** 1.2.3
**Changelog:** See [CHANGELOG.md](./CHANGELOG.md)
```

#### Change Documentation

**CHANGELOG.md format:**

```markdown
# Changelog - /deps-audit

## [1.2.0] - 2025-12-15

### Added
- PyTorch 2.7 compatibility checks
- GPU memory profiling examples

### Changed
- Updated pip-audit to v2.0 syntax
- Improved ML dependency update workflow

### Deprecated
- Old `safety check` examples (tool deprecated)

### Fixed
- Incorrect CUDA version detection
- Broken link to PyTorch docs

## [1.1.0] - 2025-11-20

### Added
- ML-specific security guidance
- PyTorch/CUDA compatibility section

### Removed
- Enterprise security theater (model signing, weekly scans)

## [1.0.0] - 2025-11-01

Initial production release.
```

### Deprecation Strategy

#### Deprecation Process

**Phase 1: Mark as Deprecated** (Month 1)

```markdown
---
model: claude-sonnet-4-0
status: DEPRECATED
replaced-by: /new-command
deprecation-date: 2025-11-20
removal-date: 2026-02-20
---

# ⚠️ DEPRECATED: Old Command Name

**Status:** Deprecated as of 2025-11-20
**Will be removed:** 2026-02-20 (3 months)
**Use instead:** `/new-command`

**Why deprecated:**
- [Reason 1]
- [Reason 2]

**Migration guide:** [See below](#migration-guide)

[Rest of command for backward compatibility...]
```

**Phase 2: Add Warnings** (Month 2)

Add prominent warnings at the start of output:

```markdown
⚠️ WARNING: This command is deprecated and will be removed on 2026-02-20.
Please use `/new-command` instead.
Migration guide: [link]
```

**Phase 3: Remove Command** (Month 3)

Replace command file with redirect:

```markdown
---
model: claude-sonnet-4-0
---

# Command Removed: /old-command

This command was removed on 2026-02-20.

**Use instead:** `/new-command`

**Migration guide:**
1. Replace `/old-command` with `/new-command`
2. Update any documentation references
3. See [migration guide](link) for details

**Need the old command?**
Archived version: [link to git history]
```

### Migration Paths

#### Migration Guide Template

```markdown
# Migration Guide: /old-command → /new-command

## Quick Start

**Old:**
```bash
/old-command "analyze dependencies"
```

**New:**

```bash
/new-command "analyze dependencies"
```

## What Changed

### Removed Features

- Feature 1 (reason: security theater)
- Feature 2 (reason: deprecated tool)

### New Features

- Feature 3 (benefit: faster execution)
- Feature 4 (benefit: better accuracy)

### Changed Behavior

- Behavior 1: Was X, now Y (reason)
- Behavior 2: Was A, now B (reason)

## Step-by-Step Migration

1. **Update command invocations:**
   - Search codebase for `/old-command`
   - Replace with `/new-command`

2. **Update documentation:**
   - Update README references
   - Update team wiki
   - Notify team members

3. **Test new command:**
   - Run on sample project
   - Verify output meets expectations
   - Report any issues

## Breaking Changes

### Change 1: [Description]

**Impact:** [Who is affected]
**Workaround:** [How to handle]

### Change 2: [Description]

**Impact:** [Who is affected]
**Workaround:** [How to handle]

## Need Help?

- Issues: [link to issue tracker]
- Questions: [link to discussion forum]
- Support: [email or chat]

```

---

## Quick Reference

### Command Testing Checklist

**Copy-paste this checklist for each command:**

```markdown
## Command Testing Checklist: /your-command

**Date:** YYYY-MM-DD
**Tester:** [Your Name]
**Command Version:** X.Y.Z

### Phase 1: Smoke Testing (5 min)
- [ ] Command recognized by Claude
- [ ] Runs without errors
- [ ] Produces readable output
- [ ] Completes in reasonable time

### Phase 2: Correctness Testing (30-60 min)
- [ ] Code examples execute successfully
- [ ] Recommendations are accurate
- [ ] End-to-end test on real project passes
- [ ] Time-to-value measured: ___ minutes

### Phase 3: Usability Testing (60 min)
- [ ] First-time user succeeds within 5 minutes
- [ ] Expert user workflow efficient
- [ ] Error handling is graceful
- [ ] "When to use" guidance is clear

### Phase 4: Performance Testing (30 min)
- [ ] Execution time within target
- [ ] Scales linearly with project size
- [ ] No timeouts or hangs

### Quality Assessment
- Clarity: ___/5
- Practicality: ___/5
- Time-Efficiency: ___/5
- Specificity: ___/5
- Accuracy: ___/5
- Maintainability: ___/5

**Overall Score:** ___/5.0

**Status:**
- [ ] Production-ready (≥4.0)
- [ ] Needs work (3.0-3.9)
- [ ] Requires major revision (<3.0)

### Issues Found
1. [Issue 1] - Priority: P0/P1/P2/P3
2. [Issue 2] - Priority: P0/P1/P2/P3

### Next Steps
- [ ] Fix P0 issues
- [ ] Address P1 issues
- [ ] Document learnings
- [ ] Update command version
```

### Common Issues and Quick Fixes

| Issue | Symptom | Quick Fix | Time |
|-------|---------|-----------|------|
| **Deprecated tool** | References `safety`, `npm audit` | Update to `pip-audit`, `npm audit fix` | 15 min |
| **Enterprise assumptions** | Weekly scans, Slack alerts | Remove or mark as "Optional" | 30 min |
| **Missing examples** | Abstract advice only | Add 3-5 concrete code examples | 60 min |
| **Wrong tech stack** | Assumes Node.js, project is Python | Replace examples with Python | 90 min |
| **Too generic** | Could apply to any project | Add project-specific context | 45 min |
| **Too slow** | Takes >5 min for simple tasks | Remove unnecessary steps | 60 min |
| **Confusing structure** | User doesn't know where to start | Add "Quick Start" section at top | 30 min |
| **No "when not to use"** | Users misapply command | Add anti-patterns section | 20 min |

### Improvement Patterns Library

**Pattern 1: Add Practical Time Estimates**

```diff
- Run dependency audit.

+ Run dependency audit.
+ **Time:** 30 minutes initial setup + 15 minutes quarterly
```

**Pattern 2: Add "When NOT to Use"**

```diff
+ ## When to Use This Command
+
+ ✅ **Good use cases:**
+ - [Scenario 1]
+ - [Scenario 2]
+
+ ❌ **Don't use for:**
+ - [Anti-pattern 1]
+ - [Anti-pattern 2]
```

**Pattern 3: Add Concrete Examples**

```diff
- Optimize your code for performance.

+ Optimize your code for performance.
+
+ **Example:**
+ ```python
+ # Before: O(n²) complexity
+ for i in items:
+     for j in items:
+         if i == j:
+             count += 1
+
+ # After: O(n) complexity
+ count = len([x for x in items if x in set(items)])
+ ```
```

**Pattern 4: Add Project Context**

```diff
- Check dependencies for vulnerabilities.

+ Check dependencies for vulnerabilities.
+
+ **For this project:**
+ - Focus on 38 direct dependencies
+ - Critical: torch, transformers, faiss
+ - Python 3.11+ required
```

**Pattern 5: Remove Security Theater**

```diff
- Set up 24/7 monitoring with Slack alerts
- Implement model integrity checking
- Scan for supply chain attacks daily

+ Run quarterly security review:
+ 1. pip-audit --desc (5 min)
+ 2. Review PyTorch CVEs (10 min)
+ 3. Update if needed (30 min)
```

---

## Artifact Management Guidelines

### Purpose

Commands that generate files (helper scripts, configs, templates) must manage artifacts systematically to prevent:

- ❌ Project root pollution (test files scattered everywhere)
- ❌ Sensitive data committed to git (audit outputs with CVEs)
- ❌ Orphaned artifacts (old test files never cleaned up)
- ❌ Documentation drift (examples showing wrong file paths)

### Artifact Directory Structure

**Best Practice:** All command-generated artifacts go in dedicated directories with clear ownership.

**Example Structure:**

```
project_root/
├── audit_reports/               # /deps-audit outputs
│   ├── README.md               # Usage guide (git-tracked)
│   ├── 2025-11-20-audit.json  # Audit outputs (git-ignored)
│   └── archive/               # Old reports (git-ignored)
├── test_reports/               # /tdd-cycle outputs
│   ├── README.md               # Usage guide (git-tracked)
│   ├── coverage.xml            # Coverage data (git-ignored)
│   └── .gitignore              # Ignore patterns (git-tracked)
├── performance_profiles/       # /performance-optimization outputs
│   ├── README.md               # Usage guide (git-tracked)
│   ├── profile_2025-11-20.prof # Profile data (git-ignored)
│   └── flamegraphs/           # Visualizations (git-ignored)
└── .gitignore                  # Root git-ignore (git-tracked)
```

### Git-Ignore Patterns

**For each artifact directory:**

1. **Ignore all generated files:**

   ```gitignore
   # Dependency audit reports (generated by /deps-audit)
   audit_reports/*.json
   audit_reports/**/*.json
   ```

2. **Track README files:**

   ```gitignore
   # Track usage documentation
   !audit_reports/README.md
   !audit_reports/.gitignore
   ```

3. **Use specific patterns (not wildcards):**

   ```gitignore
   # ✅ GOOD: Specific pattern
   audit_reports/*.json

   # ❌ BAD: Too broad (might ignore wanted files)
   *.json
   ```

### Artifact README Template

**Every artifact directory needs a README explaining:**

```markdown
# [Directory Name]

**Purpose:** Stores [type of artifacts] generated by `[/command-name]`.

**Contents:**
- `YYYY-MM-DD-[type].json`: [Description]
- `before-fixes-YYYY-MM-DD.json`: [Description]
- `after-fixes-YYYY-MM-DD.json`: [Description]
- `archive/`: [Description]

**Usage:**
```bash
# Generate new artifact
[command invocation]

# Process artifact
[processing command]
```

**Cleanup Policy:**

- **Keep:** Current + previous 2 artifacts
- **Archive:** Reports older than 90 days
- **Delete:** Archived reports older than 1 year

**Git Status:** All `*.json` files are git-ignored. Only this README is tracked.

```

### Cleanup Instructions

**Include in command documentation:**

```markdown
## Cleanup (Every Quarter)

Keep your `artifact_directory/` lean:

```bash
# 1. List all artifacts by date
ls -lt audit_reports/*.json

# 2. Keep current + previous 2 (for comparison)
# Delete the rest:
rm audit_reports/2025-08-*.json
rm audit_reports/2025-09-*.json

# 3. Archive important baselines (optional)
mkdir -p audit_reports/archive
mv audit_reports/before-fixes-*.json audit_reports/archive/
```

**Recommended retention:**

- Current audit + previous 2: Always keep
- Before/after pairs: Keep for 90 days
- Historical baselines: Archive indefinitely (compressed)

```

### Documentation Path Consistency

**All examples MUST use artifact directory paths:**

```markdown
# ✅ CORRECT: Uses artifact directory
.venv/Scripts/pip-audit --format json > audit_reports/2025-11-20-audit.json
python tools/summarize_audit.py audit_reports/2025-11-20-audit.json

# ❌ WRONG: Pollutes project root
.venv/Scripts/pip-audit --format json > audit.json
python tools/summarize_audit.py audit.json
```

**Phase 2.5 testing validates:**

- [ ] All examples use artifact directory paths
- [ ] No examples write to project root
- [ ] Git-ignore patterns work correctly
- [ ] Cleanup instructions provided

### Sensitive Data Prevention

**Audit artifacts often contain sensitive information:**

1. **CVE details** → May reveal attack vectors
2. **Package versions** → May reveal unpatched vulnerabilities
3. **Dependency tree** → May reveal proprietary dependencies

**Mitigation:**

```gitignore
# Always git-ignore audit outputs
audit_reports/*.json
*-audit.json
test-audit*.json

# Exception: Anonymized summaries OK to commit
!audit_reports/summary-report.md
```

**Verification:**

```bash
# Before committing, check for sensitive patterns
git diff --staged | grep -E "(CVE-|GHSA-|password|token|key)"
# Should return nothing
```

---

## Optimization Changelog Template

### Purpose

Track optimization history to:

- ✅ Document what changed and why
- ✅ Measure improvement over testing cycles
- ✅ Prevent regression (know what was fixed)
- ✅ Share learnings with team

### Changelog Format

**Location:** Add to bottom of command file as comment block

```markdown
---

<!-- CHANGELOG: Optimization History

## Version 1.0 (2025-11-20) - Production-Ready
**Quality Score:** 4.5/5.0 (↑ from 3.8)
**Testing Cycles:** 2

### Changes from v0.1 (Cycle 1 → Cycle 2)
**Priority 1 Fixes (Breaking Issues):**
- [x] Replaced $ARGUMENTS placeholder with actual content (Line 25-31)
- [x] Fixed Windows Unicode crash: Added safe_print() to helper script (tools/summarize_audit.py:63-70)
- [x] Updated time estimates: "30 min initial + 15 min quarterly" → "10 min initial + 5 min quarterly"

**Priority 2 Enhancements (Quality Improvements):**
- [x] Created helper script for human-readable summaries (tools/summarize_audit.py, 185 lines)
- [x] Added Windows Unicode workaround documentation (Lines 57-71)
- [x] Added troubleshooting section for common issues (Lines 468-534)
- [x] Improved example clarity with actual pip-audit invocations

**Bugs Fixed:**
- Helper script KeyError on skipped dependencies (torch, local packages)
- Helper script UnicodeEncodeError from emoji characters
- Inaccurate example showing 5-min scans (actual: <30 seconds)

**Testing Results:**
| Metric | Cycle 1 (v0.1) | Cycle 2 (v1.0) | Improvement |
|--------|----------------|----------------|-------------|
| Quality Score | 3.8/5.0 | 4.5/5.0 | +18% ✅ |
| First-use Success | 3/10 | 9/10 | +200% ✅ |
| Execution Time | 2.1s | <1s | -52% ✅ |
| False Positives | 2 | 0 | -100% ✅ |
| Unicode Errors | 3 | 0 | -100% ✅ |

**Lessons Learned:**
- Always test helper scripts on actual audit outputs (KeyError found in testing)
- Windows users are ~40% of userbase, Unicode workarounds essential
- Time estimates must match reality (users distrust 5-min claims for 10-sec tasks)
- Example output must match actual output (emoji vs ASCII caused confusion)

**Next Cycle Focus:**
- Add Phase 6: Apply Fixes Safely workflow
- Test quarterly maintenance checklist with real users
- Validate license compliance section accuracy

## Version 0.1 (2025-11-15) - Initial Draft
**Quality Score:** 3.8/5.0
**Testing Cycles:** 1

### Initial Issues Found (Cycle 1)
**Smoke Test:**
- ✅ Command recognized and executes
- ❌ Template variable $ARGUMENTS not replaced
- ✅ Output format acceptable

**Correctness Test:**
- ❌ Windows Unicode crash on --desc flag
- ❌ Helper script not created yet (6,000+ char JSON unreadable)
- ✅ pip-audit invocations correct

**Usability Test:**
- ❌ First-time user confused (3/10 success)
- ❌ Time estimates wildly inaccurate ("5 minutes" → <30 seconds actual)
- ✅ Quarterly workflow clear

**Overall Status:** Needs work (3.8/5.0), requires optimization cycle.

-->
```

### Abbreviated Changelog (For Minor Revisions)

**For small fixes (typos, clarifications, <10 lines changed):**

```markdown
---

<!-- CHANGELOG: Minor Revisions

## v1.0.1 (2025-11-22)
- Fixed typo in Phase 3 instructions (Line 145)
- Clarified git checkout command syntax (Line 310)
- Updated link to pip-audit docs (Line 62)

## v1.0.0 (2025-11-20)
- Initial production release (see full changelog above)

-->
```

### Changelog in Separate File

**For commands with extensive optimization history:**

Create `.claude/commands/CHANGELOG-command-name.md`:

```markdown
# /deps-audit Optimization History

## Version 1.0 (2025-11-20) - Production-Ready

[Full detailed changelog as shown above]

## Version 0.2 (2025-11-18) - Beta Testing

[Previous version changes]

## Version 0.1 (2025-11-15) - Initial Draft

[Initial version notes]
```

**Reference in command:**

```markdown
---
**Version:** 1.0
**Last Updated:** 2025-11-20
**Full optimization history:** See .claude/commands/CHANGELOG-deps-audit.md
```

### Changelog Best Practices

1. **Record BEFORE/AFTER metrics** → Prove improvement
2. **Link to line numbers** → Make fixes easy to find
3. **Explain WHY, not just WHAT** → Future maintainers need context
4. **Document bugs found** → Prevent regression
5. **Track testing cycles** → Show iteration count
6. **Note lessons learned** → Avoid repeating mistakes
7. **Update version number** → Semantic versioning (major.minor.patch)

### Version Numbering

**Semantic Versioning for Commands:**

- **Major (1.0 → 2.0):** Breaking changes, complete rewrite, different use cases
- **Minor (1.0 → 1.1):** New features, significant enhancements, new sections
- **Patch (1.0.0 → 1.0.1):** Bug fixes, typo corrections, clarifications

**Example:**

- `0.1` - Initial draft (not production-ready)
- `0.2` - After first optimization cycle
- `1.0` - Production-ready (quality ≥4.0)
- `1.1` - Added Phase 6 section
- `1.0.1` - Fixed typo in Phase 3

---

## Appendix

### Testing Tools

**Recommended tools for command testing:**

```bash
# Markdown validation
npm install -g markdownlint-cli
markdownlint .claude/commands/*.md

# Code example extraction
# Extract Python examples from markdown:
grep -Pzo '```python.*?```' command.md > examples.py

# Spell checking
npm install -g cspell
cspell .claude/commands/*.md

# Link checking
npm install -g markdown-link-check
markdown-link-check .claude/commands/*.md
```

### Metrics Tracking Template

**Spreadsheet for tracking command quality over time:**

| Command | Version | Date | Clarity | Practicality | Time-Eff | Specificity | Accuracy | Maintainability | Overall | Status |
|---------|---------|------|---------|--------------|----------|-------------|----------|-----------------|---------|--------|
| deps-audit | 0.1 | 2025-11-01 | 4 | 2 | 1 | 2 | 4 | 3 | 2.7 | Poor |
| deps-audit | 1.0 | 2025-11-20 | 5 | 5 | 5 | 5 | 5 | 4 | 4.8 | Prod-ready |
| tdd-cycle | 0.1 | 2025-11-01 | 3 | 4 | 2 | 3 | 4 | 4 | 3.3 | Needs work |
| tdd-cycle | 1.0 | 2025-11-20 | 5 | 5 | 5 | 5 | 5 | 4 | 4.8 | Prod-ready |

**Track weekly:**

- Commands tested: ___
- Issues found: ___
- Issues fixed: ___
- Average quality score: ___

### Further Reading

**Resources for command optimization:**

1. **Documentation Best Practices**
   - [Write the Docs](https://www.writethedocs.org/)
   - [Microsoft Style Guide](https://learn.microsoft.com/en-us/style-guide/)

2. **Testing Methodologies**
   - [Test-Driven Documentation](https://www.writethedocs.org/guide/writing/testing-documentation/)
   - [Usability Testing](https://www.nngroup.com/articles/usability-testing-101/)

3. **Technical Writing**
   - [Google Developer Documentation Style Guide](https://developers.google.com/style)
   - [The Elements of Style (Strunk & White)](https://en.wikipedia.org/wiki/The_Elements_of_Style)

---

## Document History

**Version 1.0** (2025-11-20)

- Initial release
- Comprehensive testing protocol
- Real-world examples from deps-audit and tdd-cycle
- Project-specific customization guidance

---

**Questions or suggestions?** File an issue or submit a PR to improve this protocol.
