---
description: How to systematically fix issues in the gravitational lensing codebase
---

# Issue Fix Workflow

## Rules for Fixing Issues

### 1. Fix One Issue at a Time
- Focus on a single issue before moving to the next
- Complete the full fix cycle before proceeding
- Verify the fix doesn't break other functionality

### 2. Research Before Fixing
Before making any changes:
// turbo-all

#### For Physics/Scientific Issues:
- Search the web for the correct physics formulas (e.g., NFW profile, critical surface density)
- Reference peer-reviewed papers when possible
- Verify units and dimensional analysis

#### For Syntax/Implementation Issues:
- Check official documentation (Python, PyTorch, FastAPI, etc.)
- Look for best practices and modern patterns
- Verify compatibility with project dependencies

### 3. Make Improvements
- Use web research and AI knowledge to improve code
- Add better error handling
- Improve documentation and comments
- Optimize performance where possible

### 4. Remove Duplicate Code
- Check for duplicate code patterns
- Extract common functionality into shared utilities
- Use DRY (Don't Repeat Yourself) principles

## Fix Cycle for Each Issue

```
1. RESEARCH
   └── Search web for correct implementation
   └── Verify physics/logic correctness
   └── Check best practices

2. ANALYZE
   └── View the affected file(s)
   └── Check for duplicate code
   └── Identify related issues

3. FIX
   └── Make the correction
   └── Add/improve documentation
   └── Handle edge cases

4. VERIFY
   └── Check imports work
   └── Run related tests if available
   └── Ensure no regressions
```

## Priority Order

### Phase 1: Critical (Runtime Failures)
1. Issue #6 - API Key auth broken
2. Issue #7 - SQLAlchemy 2.0 compatibility
3. Issue #8 - Import errors
4. Issue #9 - Duplicate UI code
5. Issue #5 - z_lens attribute
6. Issue #3 - Sigma_crit formula (PHYSICS)
7. Issue #37 - NFW formula (PHYSICS)

### Phase 2: Security Hardening
8. Issue #12 - CORS
9. Issue #13 - Optional auth
10. Issue #39 - Hardcoded admin password

### Phase 3: Physics Corrections
11. Issue #4 - PINN units
12. Issue #14 - Hardcoded concentration
13. Issue #17 - Random r_s

### Phase 4-6: Remaining issues...
