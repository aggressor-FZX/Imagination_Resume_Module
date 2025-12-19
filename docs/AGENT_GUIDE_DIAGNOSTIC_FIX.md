# Guide: Live Monitoring, Diagnostic, and Crash Fix Process

This guide outlines the end-to-end process for monitoring a live application, diagnosing a critical UI crash, implementing a fix, and verifying it using automated tools.

## 1. Live Monitoring & Baseline Observation
*   **Launch Browser:** Use Playwright or Chrome DevTools MCP to navigate to the live URL (e.g., `app.cogitometric.org`).
*   **Capture Logs:** Continuously monitor `browser_console_messages` and `browser_network_requests`.
*   **Initial Check:** Verify that key endpoints (ML APIs, Auth) are returning expected status codes (200 OK, or 401 for guest sessions).

## 2. Reproduction of the Error
*   **Simulate User Flow:** Perform the exact sequence of actions that leads to the reported issue (e.g., uploading specific files, entering data, clicking buttons).
*   **Identify the Crash:** Look for "Uncaught Errors" or "Minified React Errors" in the console logs.
*   **Contextual Analysis:** Examine the network response body just before the crash. Compare the schema of a successful run vs. a failed run.

## 3. Diagnostic (Root Cause Analysis)
*   **Error Mapping:** Map the React Error code (e.g., #31: Objects are not valid as React children) to the specific data being rendered.
*   **Schema Mismatch Identification:** Identify where the backend response structure (JSON) has changed (e.g., a string array becoming an object array).
*   **Code Inspection:** Locate the component (e.g., `AnalysisResults.tsx`) and the specific line (e.g., `listitem.map`) where the unexpected object is being passed to JSX.

## 4. Fix Implementation
*   **Defensive Rendering:** Add type checks and mapping logic to extract renderable strings from objects (e.g., `typeof item === 'object' ? item.skill : item`).
*   **Normalization:** Update the data transformation layer (e.g., `adaptAnalysisToData` in `ResumeBuilder.tsx`) to ensure child components receive a consistent data structure regardless of backend fluctuations.
*   **TypeScript/Linting:** Use `Record<string, unknown>` and proper type guards to satisfy build constraints without using `any`.

## 5. Deployment & Verification
*   **Automated Reproduction Script:** Write a Playwright test (e.g., `e2e/repro_crash.spec.ts`) that mocks the problematic API response and asserts that the UI remains visible and responsive.
*   **Push & Deploy:** Commit the changes and push to the deployment branch (e.g., `master`).
*   **Live Validation:** After deployment is complete (check `mcp_render_get_deploys`), re-run the user flow on the production site using the same input data that previously caused the crash.

## 6. Cleanup
*   **Remove Mock Tests:** Delete the reproduction script once the live fix is verified to keep the codebase clean.
*   **Documentation:** Update logs or guides to reflect the fix and the new expected behavior.
