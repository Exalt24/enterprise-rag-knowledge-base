import { defineConfig } from "@playwright/test";

// The source view test runs the real backend over the demo corpus (backend/eval/demo_server.py: in-memory index, hashing
// embedder, a stand-in answer) and the real front end against it. Set E2E_PYTHON to a Python that has the backend
// requirements installed (requirements-ci.txt is enough); it defaults to `python`.
const API_PORT = 8011;
const WEB_PORT = 3100;
const python = process.env.E2E_PYTHON ?? "python";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://localhost:${WEB_PORT}`, trace: "retain-on-failure" },
  webServer: [
    {
      command: `${python} ../backend/eval/demo_server.py --port ${API_PORT}`,
      url: `http://127.0.0.1:${API_PORT}/`,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: `npm run dev -- --port ${WEB_PORT}`,
      url: `http://localhost:${WEB_PORT}/`,
      env: { NEXT_PUBLIC_API_URL: `http://127.0.0.1:${API_PORT}/api` },
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
