Kingfisher — Lightweight UI/UX Testing Checklist

Purpose: a compact, pragmatic checklist you can run in ~30–90 minutes. This version assumes you'll notice obvious breakages; items focus on core workflows and observable regressions. Mark [x] = pass, [!] = noticed problem, [-] = not applicable.

---

**CORE — Quick Run (do these in order)**

Step 0 (STARTUP & tutorial)
- [ ] Launch Kingfisher; if shown, skim the Legal/Terms UI and accept.
- [ ] Open the tutorial/help and use Skip / Enter / Arrow keys / Escape to navigate — it renders and you can complete the workflow.
- [ ] Re-open Kingfisher; the legal screen should not reappear.

Step 1 (ANALYZE FOLDERS → queue → run)
- [ ] Open Analyze Folders, pick a root, confirm folder tree loads and analyzed folders show a distinct label/state.
- [ ] Add 2–3 folders to the queue and start analysis; confirm the queue shows the folders and a running progress for the active folder.
- [ ] Pause/resume or cancel the active folder and confirm behavior is sensible (progress stops/resumes or folder shows cancelled).

Step 2 (BROWSE SCENES & OPEN IN EDITOR)
- [ ] Open a folder with scenes; confirm scene cards load (thumbnails/species/quality visible) and browsing/scrolling is responsive.
- [ ] Open a scene, preview images, and double-click an image to open in the configured editor (system default or installed editor). If editor not found, app should fall back gracefully.

Step 3 (RATINGS / METADATA / SAVE)
- [ ] Change a star rating on a scene (or in a scene dialog). Confirm visual change and that saving/exporting CSV updates persisted data (you can re-open folder to verify persistence).
- [ ] Merge two scenes quickly and confirm merged scene appears and persists.

Step 4 (CULLING — quick smoke)
- [ ] Open the Culling Assistant for a folder: confirm Accept/Reject columns appear and Auto-Categorize roughly groups images.
- [ ] Manually move a couple of images between Accept and Reject, open the Preview pane and verify metadata shown (filename/species/quality).
- [ ] Click "Done Culling" and exercise either Move Rejects or Write XMP (one action). Confirm the action runs and reports completion; check that physical moves or `.xmp` sidecars appear as expected.
- [ ] If XMP conflicts appear, the conflict prompt should list files and allow Overwrite/Skip.

Step 5 (SETTINGS & RESTART)
- [ ] Open Settings, change an obvious preference (e.g., editor or zoom), save, restart the app, and confirm the change persisted.

Step 6 (FEEDBACK & SUPPORT — final check)
- [ ] Open support surfaces and confirm they point to GitHub issues / releases / repository pages.
- [ ] Confirm there is no telemetry opt-in prompt and settings describe telemetry as disabled.

---

**GRANULAR — If you have more time (optional checks you can spot quickly)**
- [ ] Reorder queue items and confirm analysis respects the new order.
- [ ] Use filters (species/rating) in the scene grid and confirm filters narrow results correctly.
- [ ] Try multi-select (Ctrl/Shift) and run Merge or other bulk actions.
- [ ] Replay the tutorial from the Welcome panel and confirm it still guides you through the core workflow without bundled demo data.

---

**MINOR — Spot checks (only if you notice something odd)**
- [ ] Toasts and toasts’ dismiss behavior look reasonable and don't overlap unreadably.
- [ ] Tutorials render without clipped text on typical displays.
- [ ] UI remains usable at smaller window sizes (no broken buttons).
- [ ] XMP sidecars are readable text/XML and include `xmp:Rating` for rated images.

---

Notes:
- This light checklist is deliberately short — it relies on you noticing problems during normal use and marking [!] when something is wrong.
- Telemetry is intentionally disabled in Kingfisher; verify support and release links instead.

Last updated: 2026-03-04
