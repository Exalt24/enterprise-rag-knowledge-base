import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";

// Asks a question on the demo corpus, clicks a citation and checks that the passage the panel highlights is the cited
// chunk: the same text the API returns for that citation, and text that is really in the document on disk.
// The answer text is a stand-in (see backend/eval/demo_server.py); the citations are what hybrid retrieval returned.

const API = "http://127.0.0.1:8011/api";
const CORPUS = path.resolve(__dirname, "../../backend/eval/corpus");
const squash = (text: string) => text.replace(/\s+/g, " ").trim();

async function ask(page: Page, question: string) {
  await page.goto("/");
  // React must have hydrated before the question is typed, or the controlled input throws the text away
  await page.waitForLoadState("networkidle");
  await page.getByPlaceholder(/Ask a question/).fill(question);
  await expect(page.getByRole("button", { name: "Ask" })).toBeEnabled();
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.getByTestId("citation").first()).toBeVisible();
}

test("a citation opens the exact chunk it cites, highlighted, with its neighbours", async ({ page, request }) => {
  await ask(page, "who approves a change request");

  const citation = page.getByTestId("citation").first();
  const fileName = (await citation.locator("span.font-medium").innerText()).trim();
  const docId = await citation.getAttribute("data-doc-id");
  const chunkId = await citation.getAttribute("data-chunk-id");
  expect(docId).toMatch(/^[0-9a-f]{64}$/);
  expect(Number(chunkId)).toBeGreaterThanOrEqual(0);

  await citation.click();
  const panel = page.getByTestId("source-panel");
  await expect(panel).toBeVisible();

  // what the API holds for this citation
  const response = await request.get(`${API}/documents/${docId}/chunks/${chunkId}?context=2`);
  expect(response.ok()).toBeTruthy();
  const body = await response.json();

  // the highlighted text is the cited chunk, word for word
  const highlighted = panel.getByTestId("cited-passage");
  await expect(highlighted).toBeVisible();
  expect(squash(await highlighted.innerText())).toBe(squash(body.chunk.text));
  expect(squash(body.chunk.text).length).toBeGreaterThan(40);

  // and that chunk is text from the document on disk, not something the panel made up
  const onDisk = squash(readFileSync(path.join(CORPUS, fileName), "utf8"));
  expect(onDisk).toContain(squash(body.chunk.text));

  // it is the passage the citation previewed
  const preview = (await citation.innerText()).includes(fileName);
  expect(preview).toBeTruthy();

  // the document name, the location and the neighbours
  await expect(panel.getByTestId("source-file")).toHaveText(fileName);
  await expect(panel.getByTestId("source-location")).toContainText(`Chunk ${Number(chunkId) + 1} of ${body.total_chunks}`);
  await expect(panel.getByTestId("context-before")).toHaveCount(body.before.length);
  await expect(panel.getByTestId("context-after")).toHaveCount(body.after.length);
  for (let i = 0; i < body.before.length; i++) {
    expect(squash(await panel.getByTestId("context-before").nth(i).innerText())).toBe(squash(body.before[i].text));
  }
  for (let i = 0; i < body.after.length; i++) {
    expect(squash(await panel.getByTestId("context-after").nth(i).innerText())).toBe(squash(body.after[i].text));
  }
  expect(body.before.length + body.after.length).toBeGreaterThan(0);

  // Escape closes the panel and puts focus back on the citation
  await page.keyboard.press("Escape");
  await expect(panel).toBeHidden();
  await expect(citation).toBeFocused();
});

test("every citation of an answer opens its own chunk", async ({ page, request }) => {
  await ask(page, "how is a risk escalated");
  const count = await page.getByTestId("citation").count();
  expect(count).toBeGreaterThan(1);
  for (let i = 0; i < count; i++) {
    const citation = page.getByTestId("citation").nth(i);
    const docId = await citation.getAttribute("data-doc-id");
    const chunkId = await citation.getAttribute("data-chunk-id");
    await citation.click();
    const body = await (await request.get(`${API}/documents/${docId}/chunks/${chunkId}?context=2`)).json();
    await expect(page.getByTestId("cited-passage")).toHaveText(body.chunk.text, { useInnerText: true });
    await page.getByRole("button", { name: "Close source panel" }).click();
  }
});

test("a PDF citation shows its page and a workbook citation shows its sheet and cells", async ({ page }) => {
  await ask(page, "interconnection study feeder megawatts network upgrade");
  const pdf = page.getByTestId("citation").filter({ hasText: "site-report.pdf" }).first();
  await expect(pdf).toBeVisible();
  await pdf.click();
  await expect(page.getByTestId("source-location")).toContainText(/^Page \d/);
  await page.getByRole("button", { name: "Close source panel" }).click();

  await ask(page, "equipment schedule item quantity unit cost due day");
  const sheet = page.getByTestId("citation").filter({ hasText: "equipment-schedule.xlsx" }).first();
  await expect(sheet).toBeVisible();
  await sheet.click();
  await expect(page.getByTestId("source-location")).toContainText(/^Schedule!A\d+:D\d+/);
});

test("screenshot for the docs", async ({ page }) => {
  test.skip(!process.env.E2E_SCREENSHOT, "set E2E_SCREENSHOT to a file path to write the picture the docs use");
  await page.setViewportSize({ width: 1280, height: 900 });
  await ask(page, "interconnection study feeder megawatts network upgrade");
  await page.getByTestId("citation").filter({ hasText: "site-report.pdf" }).first().click();
  await expect(page.getByTestId("cited-passage")).toBeVisible();
  await page.getByTestId("cited-passage").evaluate((element) => element.scrollIntoView({ block: "center" }));
  // the Next.js dev badge is not part of the app
  await page.screenshot({ path: process.env.E2E_SCREENSHOT, style: "nextjs-portal { display: none !important }" });
});
