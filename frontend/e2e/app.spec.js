/**
 * frontend/e2e/app.spec.js
 * ========================
 * Playwright end-to-end tests for the VLSI-AI Design Studio UI.
 *
 * The tests run against the combined Flask server (api/server.py) that
 * serves the pre-built React SPA from frontend/dist/ and exposes the REST
 * API at /api/*.  ngspice is NOT required; the server automatically falls
 * back to the synthetic runner when ngspice is absent.
 *
 * Each test is scoped narrowly so failures are easy to diagnose.
 */

import { test, expect } from '@playwright/test';

// ---------------------------------------------------------------------------
// Helper: wait for the status bar to show "Ready" (either mode)
// ---------------------------------------------------------------------------
async function waitForReady(page) {
  await expect(page.getByText(/Ready \((?:ngspice|synthetic)\)/)).toBeVisible({
    timeout: 15_000,
  });
}

// ---------------------------------------------------------------------------
// 1. Page loads
// ---------------------------------------------------------------------------
test.describe('Page load', () => {
  test('renders the app header', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('VLSI-AI Design Studio')).toBeVisible();
  });

  test('renders the sub-title', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText(/Bandgap Reference/)).toBeVisible();
  });

  test('status bar is visible', async ({ page }) => {
    await page.goto('/');
    // Either "Connecting…" or one of the "Ready" states
    const bar = page.getByText(/Connecting…|Ready \((?:ngspice|synthetic)\)/);
    await expect(bar).toBeVisible({ timeout: 10_000 });
  });

  test('status reaches Ready state', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
  });
});

// ---------------------------------------------------------------------------
// 2. API health (via /api/status)
// ---------------------------------------------------------------------------
test.describe('API /api/status', () => {
  test('returns ok:true', async ({ request }) => {
    const resp = await request.get('/api/status');
    expect(resp.ok()).toBeTruthy();
    const json = await resp.json();
    expect(json.ok).toBe(true);
    expect(typeof json.ngspice_available).toBe('boolean');
  });
});

// ---------------------------------------------------------------------------
// 3. Sidebar navigation
// ---------------------------------------------------------------------------
test.describe('Sidebar navigation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
  });

  test('Optimization tab is active by default', async ({ page }) => {
    await expect(page.getByRole('button', { name: 'Optimization' })).toBeVisible();
    // The "Run Optimizer" button is only in the Optimization tab
    await expect(page.getByRole('button', { name: /Run Optimizer/i })).toBeVisible();
  });

  test('navigates to Layout Viewer tab', async ({ page }) => {
    await page.getByRole('button', { name: 'Layout Viewer' }).click();
    await expect(page.getByText('Layout Viewer — Bandgap Core')).toBeVisible();
  });

  test('navigates to Verification tab', async ({ page }) => {
    await page.getByRole('button', { name: 'Verification' }).click();
    await expect(page.getByText('Spec Verification Summary')).toBeVisible();
  });

  test('navigates to Logs tab', async ({ page }) => {
    await page.getByRole('button', { name: 'Logs' }).click();
    await expect(page.getByText('Optimizer Log')).toBeVisible();
  });

  test('returns to Optimization tab', async ({ page }) => {
    await page.getByRole('button', { name: 'Logs' }).click();
    await page.getByRole('button', { name: 'Optimization' }).click();
    await expect(page.getByRole('button', { name: /Run Optimizer/i })).toBeVisible();
  });
});

// ---------------------------------------------------------------------------
// 4. Optimization tab — controls
// ---------------------------------------------------------------------------
test.describe('Optimization tab controls', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
  });

  test('Spec Targets card is visible', async ({ page }) => {
    await expect(page.getByText('Spec Targets')).toBeVisible();
  });

  test('Design Variables card is visible', async ({ page }) => {
    await expect(page.getByText('Design Variables')).toBeVisible();
  });

  test('sliders are present', async ({ page }) => {
    const sliders = page.locator('input[type="range"]');
    await expect(sliders).toHaveCount(3);
  });

  test('live estimate panel is present', async ({ page }) => {
    // The Spec Targets card always shows the Vref spec bar label
    await expect(page.getByText('Vref (mV)').first()).toBeVisible();
  });

  test('live estimate shows Vref after status loads', async ({ page }) => {
    // Wait for the live status indicator — appears once the surrogate estimate arrives
    await expect(
      page.getByText(/Live from surrogate|Estimating/)
    ).toBeVisible({ timeout: 10_000 });
    // At least one spec bar should show a numeric value (e.g. "1187.5 mV")
    await expect(page.getByText(/\d+\.\d+ mV/).first()).toBeVisible({ timeout: 3_000 });
  });

  test('Run Optimizer button is enabled after status loads', async ({ page }) => {
    const btn = page.getByRole('button', { name: /Run Optimizer/i });
    await expect(btn).not.toBeDisabled({ timeout: 10_000 });
  });
});

// ---------------------------------------------------------------------------
// 5. Optimization — run the optimizer (synthetic, small budget)
// ---------------------------------------------------------------------------
test.describe('Optimizer run (synthetic)', () => {
  test('clicking Run Optimizer shows "Running…" then finishes', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);

    const btn = page.getByRole('button', { name: /Run Optimizer/i });
    await expect(btn).not.toBeDisabled({ timeout: 10_000 });

    await btn.click();

    // Button should briefly say "Running…"
    await expect(page.getByRole('button', { name: /Running…/i })).toBeVisible({
      timeout: 5_000,
    });

    // Wait for the optimizer to finish (50 synthetic iterations, ~30 s max)
    await expect(page.getByRole('button', { name: /Run Optimizer/i })).toBeVisible({
      timeout: 60_000,
    });
  });

  test('optimizer results appear in the candidates table', async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);

    const btn = page.getByRole('button', { name: /Run Optimizer/i });
    await btn.click();

    // Wait for at least one result row starting with "BG-"
    await expect(page.getByText(/BG-\d+/).first()).toBeVisible({ timeout: 60_000 });
  });
});

// ---------------------------------------------------------------------------
// 6. Layout Viewer tab
// ---------------------------------------------------------------------------
test.describe('Layout Viewer tab', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await waitForReady(page);
    await page.getByRole('button', { name: 'Layout Viewer' }).click();
  });

  test('Layout Viewer heading is visible', async ({ page }) => {
    await expect(page.getByText('Layout Viewer — Bandgap Core')).toBeVisible();
  });

  test('Layer Legend card is visible', async ({ page }) => {
    await expect(page.getByText('Layer Legend')).toBeVisible();
  });

  test('DRC violations stat is present', async ({ page }) => {
    await expect(page.getByText(/DRC:/)).toBeVisible({ timeout: 10_000 });
  });

  test('layout SVG renders', async ({ page }) => {
    // The LayoutViewer renders an <svg> when patch data is available
    await expect(page.locator('svg').first()).toBeVisible({ timeout: 10_000 });
  });
});

// ---------------------------------------------------------------------------
// 7. API endpoints (direct HTTP checks via Playwright request)
// ---------------------------------------------------------------------------
test.describe('API endpoints', () => {
  test('POST /api/simulate returns vref_V', async ({ request }) => {
    const resp = await request.post('/api/simulate', {
      data: {
        params: { N: 8, R1: 180000, R2: 18000, W_P: 5e-6, L_P: 1e-6 },
        use_synthetic: true,
      },
    });
    expect(resp.ok()).toBeTruthy();
    const json = await resp.json();
    expect(typeof json.vref_V).toBe('number');
    expect(json.spec_checks).toBeDefined();
  });

  test('GET /api/layout/preview returns patch data', async ({ request }) => {
    const resp = await request.get('/api/layout/preview?seed=1&patch_size=32');
    expect(resp.ok()).toBeTruthy();
    const json = await resp.json();
    expect(json.patch_size).toBe(32);
    expect(Array.isArray(json.patch)).toBe(true);
  });

  test('POST /api/optimize (small budget) returns history', async ({ request }) => {
    const resp = await request.post('/api/optimize', {
      data: { budget: 3, n_init: 2, seed: 0, use_synthetic: true },
      timeout: 30_000,
    });
    expect(resp.ok()).toBeTruthy();
    const json = await resp.json();
    expect(json.history.length).toBe(3);
    expect(Array.isArray(json.convergence)).toBe(true);
  });

  test('POST /api/netlist/export returns SPICE text', async ({ request }) => {
    const resp = await request.post('/api/netlist/export', {
      data: { params: { N: 8, R1: 180000, R2: 18000, W_P: 5e-6, L_P: 1e-6 } },
    });
    expect(resp.ok()).toBeTruthy();
    const json = await resp.json();
    expect(json.ok).toBe(true);
    expect(json.netlist_text.length).toBeGreaterThan(0);
  });

  test('GET /api/project/files?kind=netlists lists files', async ({ request }) => {
    const resp = await request.get('/api/project/files?kind=netlists');
    expect(resp.ok()).toBeTruthy();
    const json = await resp.json();
    expect(json.ok).toBe(true);
    expect(Array.isArray(json.files)).toBe(true);
  });
});
