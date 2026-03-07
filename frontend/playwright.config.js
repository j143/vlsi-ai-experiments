// frontend/playwright.config.js
// ==============================
// Playwright E2E test configuration.
//
// Tests run against the combined Flask server (api/server.py) which serves the
// pre-built React SPA from frontend/dist/ via STATIC_DIR.
//
// Start the server before running tests:
//   STATIC_DIR=frontend/dist python api/server.py &
//   cd frontend && npx playwright test

import { defineConfig, devices } from '@playwright/test';

const BASE_URL = process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:5000';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  timeout: 30_000,
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: 'playwright-report' }],
  ],
  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    // Video recording is retained only on failure; disable in local dev with
    // PLAYWRIGHT_VIDEO=off to speed up test runs.
    video: process.env.PLAYWRIGHT_VIDEO === 'off' ? 'off' : 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  // No webServer block — the server must be started externally before tests run.
});
