import { defineConfig } from '@playwright/test';

/**
 * The browser suite exists for one thing the Node suite structurally cannot
 * do: Web Audio. web/ does not implement biquads — it hands f0/Q/gain to
 * native BiquadFilterNodes — so the only way to check that the resulting
 * response matches SciPy is to run it in a real browser.
 */
export default defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: 'http://127.0.0.1:8788',
  },
  webServer: {
    command: 'python3 -m http.server 8788 --directory web --bind 127.0.0.1',
    url: 'http://127.0.0.1:8788/index.html',
    reuseExistingServer: !process.env.CI,
    stdout: 'ignore',
  },
});
