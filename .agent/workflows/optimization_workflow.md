---
description: Workflow for optimizing a page/feature (UI & Backend)
---

# Page/Feature Optimization Workflow

For each page or feature identified in the task list, follow these steps judiciously.

1.  **Preparation**
    - Identify the target file(s) (e.g., `app/pages/02_Simple_Lensing.py`).
    - Identify the corresponding backend logic or API endpoints.
    - Check for existing tests in `tests/`.

2.  **Design Research**
    - Search for modern design references specifically for this type of scientific/dashboard interface (e.g., "modern dashboard ui streamlit", "scientific visualization ui design").
    - Note down specific CSS effects (glassmorphism, gradients, animations) to apply.

3.  **Implementation**
    - **Backend**: Ensure the logic is robust. Fix any "Phase 12" import issues or database dependencies. Add error handling.
    - **Frontend**:
        - Apply `styles.inject_custom_css()`.
        - Use standard components (`render_header`, `render_card`) where possible.
        - Add custom animations (fadeIn, slideIn) to key elements.
        - Improve layout (spacing, columns, expanders).

4.  **Verification (Iterative)**
    - **Test Run**: Run the Streamlit app locally (if possible) or visually inspect the code structure.
    - **Automated Tests**: Run relevant pytest tests.
    - **Screenshots (Simulated)**: If using browser tool, capture screenshots. *Note: Since we are in an agentic environment, we might rely on code review and test outputs if live browsing of local app is restricted.*

5.  **Refinement**
    - Analyze the results/screenshots.
    - Tweak CSS or logic as needed.
    - Repeat verification.

6.  **Final Check**
    - Ensure no regressions in other parts of the app.
    - Mark the task as complete in `task.md`.
