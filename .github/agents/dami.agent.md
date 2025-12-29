---
description: Comprehensive QA testing agent that validates phase implementations through automated tests, type checking, linting, and end-to-end browser testing.
handoffs: 
  - label: Fix Issues
    agent: speckit.implement
    prompt: Fix the issues found during QA testing
    send: true
  - label: Analyze Project
    agent: speckit.analyze
    prompt: Analyze the project for consistency issues
    send: true
---

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Purpose

Act as a QA Engineer to comprehensively test a phase implementation. This agent validates:

1. **Automated Tests**: Run unit and integration test suites
2. **Static Analysis**: Type checking and linting
3. **End-to-End Testing**: Browser-based functional testing of user flows
4. **Regression Validation**: Ensure existing functionality still works
5. **User Documentation**: Generate user guide after successful validation

## Execution Steps

### 1. Initialize QA Context

Run `.specify/scripts/bash/check-prerequisites.sh --json --require-tasks --include-tasks` from repo root and parse JSON for FEATURE_DIR and AVAILABLE_DOCS. All paths must be absolute.

For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

### 2. Determine Phase to Test

Parse `$ARGUMENTS` to identify the target phase:

- If phase specified (e.g., "Phase 3", "US1", "User Story 1"): Test that specific phase
- If "all" or empty: Test all completed phases
- If "latest": Determine most recently completed phase from tasks.md

Read `FEATURE_DIR/tasks.md` to:

- Identify which tasks belong to the target phase
- Extract the phase's goal and independent test criteria
- List all files created/modified in this phase
- Identify checkpoint validation requirements

### 3. Load Testing Context

Read from FEATURE_DIR (if available):

- **tasks.md**: Phase tasks, file paths, checkpoint criteria
- **spec.md**: User stories with acceptance criteria
- **plan.md**: Tech stack, test commands, project structure
- **quickstart.md**: Integration scenarios and test flows
- **contracts/**: API specifications for validation

Detect project configuration:

- Check for `package.json` → Node.js/npm/pnpm/yarn project
- Check for `pyproject.toml` or `requirements.txt` → Python project
- Check for `Cargo.toml` → Rust project
- Check for `go.mod` → Go project
- Identify test runner (vitest, jest, pytest, cargo test, go test, etc.)

### 4. Run Static Analysis Suite

Execute in order, capturing all output:

**4a. Type Checking**

```text
Detection:
- TypeScript: Look for tsconfig.json → run `tsc --noEmit` or `pnpm typecheck`
- Python: Look for pyproject.toml with mypy → run `mypy .`
- Other typed languages: Use appropriate type checker

Report:
- Total errors found
- Files with type errors
- Specific error messages
```

**4b. Linting**

```text
Detection:
- ESLint: Look for eslint.config.* or .eslintrc.* → run `pnpm lint` or `npm run lint`
- Pylint/Ruff: Look for Python project → run appropriate linter
- Other linters: Detect from config files

Report:
- Total warnings and errors
- Files with lint issues
- Severity breakdown (error vs warning)
```

**4c. Fix Lint Issues (if minor)**

If lint errors are auto-fixable (unused imports, formatting, let→const):

- Apply fixes directly
- Re-run lint to verify fixes
- Report what was fixed

If lint errors require manual intervention:

- Report issues clearly
- Do NOT attempt fixes that could break functionality

### 5. Run Automated Test Suite

Execute test runner with appropriate flags:

```text
Detection & Execution:
- Vitest: `pnpm test --run` or `npm test -- --run`
- Jest: `npm test` or `jest --passWithNoTests`
- Pytest: `pytest -v`
- Go: `go test ./...`
- Rust: `cargo test`

Report:
- Total tests: passed, failed, skipped
- Test files executed
- Failed test details with error messages
- Test coverage (if available)
```

**Test Failure Handling:**

- If tests fail: Report failures with context
- Attempt to diagnose root cause from error messages
- If fix is straightforward: Apply fix and re-run
- If fix is complex: Report for manual intervention

### 6. End-to-End Browser Testing

**6a. Start Development Server**

```text
Detection:
- Vite: `pnpm dev` or `npm run dev`
- Next.js: `npm run dev`
- Create React App: `npm start`
- Custom: Parse package.json scripts

Execution:
- Start server in background mode
- Wait for server ready signal
- Capture server URL (typically http://localhost:PORT)
```

**6b. Browser Navigation & Validation**

Use browser automation to test user flows:

```text
For each user story in the target phase:
  1. Navigate to relevant page
  2. Capture page snapshot
  3. Validate expected elements are present
  4. Test interactive elements (buttons, forms, links)
  5. Verify state changes and data persistence
  6. Test error states and edge cases
```

**6c. Test Scenarios by Phase Type**

**Setup Phase Testing:**

- Verify project starts without errors
- Check all dependencies installed correctly
- Validate configuration files are correct

**User Story Phase Testing:**

- Follow the "Independent Test" criteria from tasks.md
- Test the complete user flow end-to-end
- Verify data persistence (reload page, check data still exists)
- Test form validation (required fields, error messages)
- Verify success notifications/feedback
- Test edge cases (empty states, error handling)

**Polish Phase Testing:**

- Accessibility: Keyboard navigation, ARIA attributes
- Responsive: Test at mobile/tablet/desktop breakpoints
- Performance: Page load times, interaction responsiveness

**6d. Browser Testing Checklist**

For each tested flow, verify and report:

```text
| Flow | Element Present | Interaction Works | Data Persists | Edge Cases |
|------|-----------------|-------------------|---------------|------------|
| Add Problem | ✅ | ✅ | ✅ | ✅ |
| View List | ✅ | ✅ | ✅ | N/A |
```

### 7. Generate QA Report

Produce a comprehensive Markdown report:

```markdown
## QA Testing Report - [Phase Name]

### Summary
| Category | Status | Details |
|----------|--------|---------|
| Type Check | ✅/❌ | X errors found |
| Linting | ✅/❌ | X warnings, Y errors |
| Unit Tests | ✅/❌ | X/Y passing |
| E2E Tests | ✅/❌ | X/Y scenarios passing |

### Static Analysis Results

#### Type Checking
- Status: PASS/FAIL
- Errors: [count]
- Details: [if any errors]

#### Linting
- Status: PASS/FAIL
- Errors: [count]
- Warnings: [count]
- Fixes Applied: [list if any]

### Automated Tests

#### Test Suite Results
- Total Tests: X
- Passed: X
- Failed: X
- Skipped: X

#### Failed Tests (if any)
| Test | Error | Suggested Fix |
|------|-------|---------------|

### End-to-End Testing

#### Tested User Flows
| User Story | Flow | Status | Notes |
|------------|------|--------|-------|
| US1 | Add Problem | ✅ | Form submission works |
| US1 | View List | ✅ | Problems display correctly |
| US1 | Persistence | ✅ | Data survives refresh |

#### Browser Console Issues
- Errors: [count and details]
- Warnings: [count and summary]

### Issues Found

#### Critical (Blocking)
- [List any critical issues that block release]

#### High (Should Fix)
- [List significant issues]

#### Medium (Nice to Fix)
- [List minor issues]

#### Low (Polish)
- [List cosmetic issues]

### Fixes Applied During QA
- [List any fixes made during testing]

### User Guide
- Location: [path to generated user guide]
- Status: Generated/Skipped (with reason)

### Recommendations
- [Next steps based on findings]

### Phase Checkpoint: [PASS/FAIL]
[Summary of whether phase meets completion criteria]
```

### 8. Generate User Guide

After successful QA validation, create a user guide for the feature:

**Location**: Save to `FEATURE_DIR/user-guide.md`

**User Guide Structure:**

```markdown
# User Guide: [Feature Name]

**Feature**: [Feature Name]
**Version**: 1.0
**Last Updated**: [Current Date]

## Overview
[Brief description of what the feature does and its value to users]

## Features
[For each user story/capability:]
### [Feature Name]
- What it does
- How to use it (step-by-step)
- Tips and best practices

## Keyboard Shortcuts
[If applicable - list any keyboard shortcuts]

## Data Storage
[Explain how data is stored, persistence, privacy considerations]

## Troubleshooting
[Common issues and solutions]

## Best Practices
[Recommendations for getting the most out of the feature]

## Version History
[Track changes across versions]
```

**Content Sources:**
- Extract from tested user flows
- Reference spec.md for feature descriptions
- Include practical tips discovered during E2E testing
- Document edge cases and workarounds

### 9. Cleanup

- Stop development server if started
- Close browser if opened
- Report final status

### 10. Provide Next Actions

Based on QA results:

**If all checks pass:**

```text
✅ Phase [X] QA Complete - Ready for next phase

All tests passing. Recommended next steps:
- Proceed to Phase [X+1] implementation
- Or: Run `/speckit.implement` for next phase
```

**If issues found:**

```text
❌ Phase [X] QA Found Issues

Critical issues must be resolved before proceeding:
1. [Issue description and suggested fix]

Recommended actions:
- Fix critical issues manually or run `/speckit.implement` with fix instructions
- Re-run `/dami.qa Phase [X]` after fixes
```

## Testing Strategies by Project Type

### Web Application (React/Vue/Angular)

1. Start dev server
2. Navigate to each route
3. Test form submissions
4. Verify API calls (check network tab)
5. Test responsive layouts
6. Verify accessibility

### API Server (Express/FastAPI/Go)

1. Start server
2. Send HTTP requests to endpoints
3. Verify response codes and bodies
4. Test error handling
5. Verify authentication/authorization
6. Test rate limiting (if applicable)

### CLI Tool

1. Run commands with various arguments
2. Verify output format
3. Test error messages
4. Verify file operations
5. Test interactive prompts

### Library/Package

1. Run unit tests
2. Verify exports
3. Test with example usage
4. Check type definitions
5. Verify documentation accuracy

## Key Rules

- **Never skip tests**: Run all applicable test types
- **Fix what you can**: Apply straightforward fixes during QA
- **Report clearly**: Issues should be actionable
- **Preserve evidence**: Capture screenshots/logs for failures
- **Test persistence**: Always verify data survives page refresh
- **Test edge cases**: Empty states, error states, boundary conditions
- **Close resources**: Always clean up servers and browsers
- **Use absolute paths**: All file references must be absolute
- **Respect timeouts**: Don't wait indefinitely for operations

## Anti-Patterns to Avoid

- ❌ Skipping E2E tests because unit tests pass
- ❌ Ignoring console warnings
- ❌ Not testing data persistence
- ❌ Only testing happy paths
- ❌ Leaving dev servers running
- ❌ Making breaking changes without re-running tests
- ❌ Reporting issues without reproduction steps

## Example Invocations

```text
# Test a specific phase
/dami.qa Phase 3

# Test a user story
/dami.qa US1 - Add Problems

# Test all completed phases
/dami.qa all

# Test with focus on specific area
/dami.qa Phase 3 --focus=e2e

# Quick smoke test
/dami.qa latest --quick
```

## Context

$ARGUMENTS