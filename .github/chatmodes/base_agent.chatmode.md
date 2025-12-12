---
description: "General purpose chat mode with standard tools and safety guardrails for command execution."

tools:
  ['edit/createFile', 'edit/createDirectory', 'edit/editFiles', 'search', 'runCommands', 'runTasks', 'chrome-devtools/*', 'playwright-browser-automation/*', 'render/*', 'usages', 'problems', 'changes', 'testFailure', 'openSimpleBrowser', 'fetch', 'extensions']
---

Define the purpose of this chat mode and how AI should behave: response style, available tools, focus areas, and any mode-specific instructions or constraints.

Remember you are in a WSL environment unless otherwise specified.

## Command Safety Guardrails

- Wrap any exploratory or diagnostic commands (for example `npx`, MCP server probes, or other external processes) in a short `timeout` (≤10s) **or** launch them in a background terminal to prevent the VS Code agent session from hanging.
  - **Exception:** Long-running `task-master-ai` commands (installed locally via npm) should be launched directly in a background terminal without a timeout so they can complete their handshake.
- If a command must run longer than the timeout, start it in a background terminal and monitor the output separately; never block the foreground agent thread waiting on unbounded work.
- Prefer capturing outputs with `timeout <seconds> <command> || true` when quick verification is enough; escalate to background execution only when sustained processes are required.
- Before executing any Python or pip command, explicitly activate the project virtual environment (`source .venv/bin/activate`) so that the `.venv` tooling is always used instead of the system Python runtime. Use UV pip for package installs where possible.

## MCP Server Handling

- **NEVER run MCP servers directly in terminal** - MCP servers (like `task-master-ai`, `context7`, etc.) are designed to be managed by VS Code's MCP system, not executed as standalone commands
- **MCP servers hang when run manually** - They expect VS Code to initiate the MCP protocol handshake and will block indefinitely if run directly
- **Testing MCP servers** - Use `npm view <package>` to verify package existence, but never run the server command directly
- **MCP configuration** - Only modify `.vscode/mcp.json` for server configuration; VS Code handles the actual server lifecycle

## Upgrade host JSON Schema validator (optional / maintainer action)

If your host validator rejects schemas that declare Draft 2020-12 (for example tools using `$dynamicRef`), upgrading the host JSON Schema library to an implementation that supports Draft 2020-12 is the cleanest long-term fix. The `ajv` library (v8+) supports Draft 2020-12 via a separate entrypoint.

Minimal install and initialization steps (run as maintainer/root on the host where the validator runs):

```bash
# Install Ajv and helpers
npm install ajv@latest ajv-formats
```

Sample Node initialization using the Draft-2020 Ajv entrypoint:

```js
// require the 2020 build
const Ajv2020 = require("ajv/dist/2020");
const addFormats = require("ajv-formats");

const ajv = new Ajv2020();
addFormats(ajv);

// Use ajv.validate or ajv.compile on tool schemas that use draft-2020-12 features
// Example:
// const validate = ajv.compile(schema);
// const valid = validate(instance);
```

If you prefer not to upgrade the host, consider these alternatives:

- change tool schemas to target Draft-07 or remove the `$schema` line
- avoid draft-2020-only keywords like `$dynamicRef` or `unevaluatedProperties` in tool schemas

## Context7 Usage Protocol

- **Prioritize Context7 for Specialized Tools**: When working with any library, framework, or tool that is not part of the standard Python library (e.g., `asyncio`, `json`, `os`), you **must** first use the `Context7` tool to get up-to-date documentation and best practices.
- **Examples**: Before using packages like `FastAPI`, `pydantic`, `Docker`, `Render`, or any other third-party library, query `Context7` for implementation guidance.
- **Rationale**: This ensures that the agent's actions are based on the latest official documentation, reducing errors and improving the quality of the generated code and configurations.

## MCP Memory Protocol

- **Purpose**: To maintain a persistent knowledge graph of the project, including key files, functions, architectural components, and user preferences. This helps in retaining context across sessions and making more informed decisions.

- **When to Use**:

  - **On Project Start**: Create entities for core files and components to establish a baseline understanding.
  - **When Introducing New Concepts**: Add entities for new libraries, modules, or significant functions.
  - **To Remember User Preferences**: Add observations to relevant entities to record specific instructions (e.g., "DeepSeek should be the last resort in the fallback chain.").
  - **To Understand Relationships**: Create relations between entities to map out dependencies and interactions (e.g., `function_A` _uses_ `service_B`).

- **Best Practices**:
  - **Be Specific**: Use clear and descriptive names for entities and relations.
  - **Keep it Updated**: As the project evolves, update the knowledge graph to reflect the changes.
  - **Query Before Acting**: Use the memory to recall information before making decisions.
