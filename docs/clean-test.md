# Clean-account test on a real Windows PC

This checklist proves that the release zip works for a user who has never had
Python, the Visual C++ runtime, the models or a ShotFinder index. It uses a
**new local Windows user account** on the same PC: a fresh profile with its own
`%LOCALAPPDATA%`, without a virtual machine.

The automated version of this check runs on a clean GitHub Actions
`windows-latest` runner for every `v*` tag. This manual pass is the final check
on real hardware, with a real tray, a real mouse and real video codecs.

## 0. Prepare from your normal account

You need administrator rights to create and delete the test account.

1. Get the release zip. Either download `ShotFinder-0.1.1-win64.zip` from the
   [releases page](https://github.com/brighinatommaso-eng/shotfinder/releases),
   or download the `ShotFinder-windows` artifact from the green workflow run.
   Note the SHA-256 from the release notes.
2. Create `C:\Users\Public\Documents\ShotFinder-test\` and copy the zip there so
   the new account can reach it without a browser download.
3. Copy the three sample files from the repository folder
   `shotfinder\selftest_fixtures\` into
   `C:\Users\Public\Documents\ShotFinder-test\samples\`:

   | File | What it proves |
   | --- | --- |
   | `invoice.png` | OCR text: `ACME INVOICE 2024` |
   | `pattern.png` | visual match for `a colorful test pattern with bars` |
   | `talk.mp4` | speech: `quick brown fox`, spoken at about 0.3 s |

4. Make sure you know the password/PIN of your administrator account: you will
   sign out and back in several times.

## 1. Create the test account

1. **Settings â†’ Accounts â†’ Other users â†’ Add account**.
2. Click **I don't have this person's sign-in information**, then **Add a user
   without a Microsoft account**.
3. Username `ShotFinderTest`, choose a simple local password (or no password).
4. Leave the account type as **Standard user**. ShotFinder needs no
   administrator rights.
5. Sign out of your account and sign in as `ShotFinderTest`. The first sign-in
   takes a few minutes while Windows creates the profile.

## 2. Confirm the profile is clean

In the new account, open PowerShell and check:

```powershell
Get-Command python -ErrorAction SilentlyContinue   # must print nothing
Test-Path "$env:LOCALAPPDATA\ShotFinder"            # must print False
Get-Process ShotFinder -ErrorAction SilentlyContinue # must print nothing
```

The machine may still have `msvcp140.dll` in `System32` because other software
installed the redistributable. That is expected. The point of the test is that
ShotFinder must run **without** relying on it; after extracting, check that the
bundle carries its own copies:

```powershell
$internal = "$env:USERPROFILE\Desktop\ShotFinder\_internal"
Get-ChildItem "$internal\msvcp140.dll",
              "$internal\vcruntime140_1.dll",
              "$internal\concrt140.dll" -ErrorAction SilentlyContinue
```

`concrt140.dll` may be absent if nothing imports it; the build fails instead if
a required VC++ DLL is missing.

## 3. Extract the zip

1. Copy `ShotFinder-0.1.1-win64.zip` from
   `C:\Users\Public\Documents\ShotFinder-test\` to the Desktop.
2. Right-click the zip â†’ **Properties**. If **Unblock** is shown, tick it and
   click **Apply**. (A zip downloaded by a browser gets this; a file copied from
   a local folder may not.)
3. Right-click â†’ **Extract Allâ€¦**, extract to the Desktop. You get
   `Desktop\ShotFinder\ShotFinder.exe`.
4. Verify the download:

   ```powershell
   Get-FileHash .\ShotFinder-0.1.1-win64.zip -Algorithm SHA256
   ```

   The hash must match the one in the release notes.

## 4. First launch, SmartScreen and the model download

1. Double-click `ShotFinder.exe`. There is no installer and no console window.
2. Windows SmartScreen shows **Windows protected your PC** because the exe is
   unsigned. Click **More info â†’ Run anyway**. This dialog appears only once per
   downloaded file.
3. The **Welcome to ShotFinder** dialog appears once. It explains that everything
   stays on this PC, lists Screenshots, Pictures, Videos and Downloads with
   checkboxes, and mentions the one-time ~635 MB model download. Leave the
   defaults (or untick what you do not want) and click **Start**.
4. A ShotFinder icon appears in the notification area. It may be hidden behind
   the **^** overflow button; open the overflow and drag it onto the taskbar if
   you want it visible.
5. Wait for the model download. The tray tooltip/menu shows progress; it is
   ~635 MB. Text/OCR search works while it downloads; visual and spoken search
   switch on when it finishes.
6. Check that the models landed in the new profile:

   ```powershell
   Get-ChildItem "$env:LOCALAPPDATA\ShotFinder\models" -Recurse -File |
     Select-Object FullName, Length
   ```

   Expect `clip_image_fp32.onnx`, `clip_text_int8.onnx`, `tokenizer.json`, and
   `whisper-base\model.bin` (plus the small config/tokenizer files).

## 5. Add the sample folder

1. Right-click the tray icon â†’ **Settingsâ€¦**.
2. Under **Watched folders**, click **Add folder** and choose
   `C:\Users\Public\Documents\ShotFinder-test\samples`.
3. Close Settings. Wait until indexing finishes: reopen Settings and check
   **Index** â€” it should show 3 files, 0 errors, and the video's frames/speech.
   The first video takes a few seconds.

## 6. Hotkey and the three queries

1. Press `Ctrl+Alt+Space`. The search window opens with the caret in the query
   field. If another app owns that combination, ShotFinder picks a fallback; the
   tray menu shows the active one. Note which combination worked.
2. **Text:** type `ACME INVOICE`. `invoice.png` appears as a confident match with
   the snippet `[ACME] [INVOICE] 2024`.
3. **Visual:** clear the field and type `a colorful test pattern with bars`.
   `pattern.png` appears under **Related by appearance** (visual-only results are
   suggestions, never confident matches).
4. **Spoken:** clear the field and type `quick brown fox`. `talk.mp4` appears
   with the spoken line and a timestamp around `0.3 s`.
5. Press `Esc`. Press the hotkey again to confirm the window reopens.

## 7. Video preview

1. Click the `talk.mp4` result. The player opens at or just before the matched
   line; because this line is at 0.3 s, it starts at the beginning of the clip.
2. Press `Space` to pause and play, then drag the seek bar (or use the arrow
   keys). The frame changes and the audio follows.
3. Note any black/blank frame: it means the machine's codecs cannot decode that
   file, not that the index is wrong. Close the player with `Esc`.

## 8. Quit with a physical mouse

1. Right-click the tray icon with the mouse (not the keyboard), then click
   **Quit**.
2. The icon disappears. Open Task Manager (`Ctrl+Shift+Esc`) and confirm there is
   no `ShotFinder.exe` process.
3. In `%LOCALAPPDATA%\ShotFinder`, the `index.db-wal` file should be gone or
   nearly empty: the database closed cleanly.

## 9. Restart keeps the index

1. Double-click `ShotFinder.exe` again.
2. The welcome dialog must **not** appear (the config already exists).
3. The tray menu must **not** show **Download models** (the models are present).
4. Open Settings â†’ **Index**: the file/video counts are the same as before.
5. Press `Ctrl+Alt+Space`, search `ACME INVOICE`, and confirm `invoice.png`
   returns immediately without a new scan.

## 10. Delete the test account

1. Quit ShotFinder from the tray.
2. Sign out of `ShotFinderTest` and sign in as your administrator account.
3. **Settings â†’ Accounts â†’ Other users â†’ ShotFinderTest â†’ Remove**, then choose
   **Delete account and data**. This deletes `C:\Users\ShotFinderTest`,
   including `%LOCALAPPDATA%\ShotFinder` (index, models, logs).
4. If Windows leaves `C:\Users\ShotFinderTest` behind, delete it manually.
5. Delete `C:\Users\Public\Documents\ShotFinder-test\` and the Desktop copy of
   the zip.
6. Confirm the administrator account has no ShotFinder startup entry or tray
   icon from this test.

## What to report

For every failure, note the step number, what you expected, what happened, and a
screenshot. The most useful details are:

- the exact SmartScreen text or a missing DLL name;
- whether the welcome dialog appeared once and only once;
- whether the model download finished, stalled or retried;
- the tray icon, hotkey and each query result;
- whether the video preview played and seeked;
- whether Quit left a process behind;
- whether the index survived the restart.

Before deleting the account, copy the log to the shared folder:

```powershell
Copy-Item "$env:LOCALAPPDATA\ShotFinder\logs\shotfinder.log" `
          "C:\Users\Public\Documents\ShotFinder-test\shotfinder.log"
```

This manual test uses the same Windows installation as your normal account, so
it cannot prove a machine that has never had the VC++ redistributable. That gap
is covered by `tools\check_runtime_deps.py` (run by every build) and by the
automated selftest on a fresh `windows-latest` runner.
