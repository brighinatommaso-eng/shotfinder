# Clean-machine test in Windows Sandbox

Windows Sandbox is a disposable Windows install. Nothing in it survives a
restart, and it never touches your real index or your real folders, so it is
the honest way to check that the release zip works on a machine that has never
seen Python, the models or ShotFinder.

The sandbox needs **Windows 10/11 Pro, Enterprise or Education**. It is not
available on Home.

## 1. Enable Windows Sandbox (once)

GUI: **Start → "Turn Windows features on or off" → tick "Windows Sandbox" →
OK → restart**.

PowerShell (administrator):

```powershell
Enable-WindowsOptionalFeature -Online -FeatureName "Containers-DisposableClientVM" -All
```

## 2. Prepare the folder to test

Build first, from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\build.ps1
```

You get:

- `dist\ShotFinder\` — the unpacked app
- `dist\ShotFinder-0.1.0-win64.zip` — what users download

## 3. Map the build into the sandbox

Save this as `shotfinder-test.wsb` next to `dist` and double-click it. Adjust
the `HostFolder` path if your checkout lives somewhere else.

```xml
<Configuration>
  <Networking>Enable</Networking>
  <MappedFolders>
    <MappedFolder>
      <HostFolder>C:\Programs\ShotFinder\dist</HostFolder>
      <SandboxFolder>C:\Users\WDAGUtilityAccount\Desktop\dist</SandboxFolder>
      <ReadOnly>true</ReadOnly>
    </MappedFolder>
  </MappedFolders>
  <LogonCommand>
    <Command>explorer.exe C:\Users\WDAGUtilityAccount\Desktop\dist</Command>
  </LogonCommand>
</Configuration>
```

The folder is read-only on purpose: the app must write its index and models to
`%LOCALAPPDATA%\ShotFinder`, never next to the executable.

If you want to test the zip exactly as a user receives it, copy
`ShotFinder-0.1.0-win64.zip` into the sandbox, right-click → **Extract All**,
then run the extracted `ShotFinder\ShotFinder.exe`. A zip downloaded from the
internet needs **Properties → Unblock** first, otherwise Windows marks every
file as blocked.

## 4. What to check

Run `dist\ShotFinder\ShotFinder.exe`. There is no console window: the app lives
in the tray.

1. **Welcome dialog** — appears once, on the first launch. It explains what the
   app does, states that everything stays on this PC, lists Screenshots,
   Pictures, Videos and Downloads with checkboxes, and mentions the ~635 MB
   one-time download. Click **Start**.
2. **Download** — the tray icon shows progress. Text and OCR search work
   immediately; visual and spoken search switch on when the models finish.
   On a fresh sandbox this takes a few minutes.
3. **Tray menu** — right-click the icon: *Search*, *Pause indexing*, *Download
   models* (only while missing), *Open data folder*, *Settings…*, *Quit*.
4. **Settings** — open it, check that the folders are listed, toggle *Index
   video frames* off and on, change the hotkey to something free, and close the
   window. Changes apply without a restart.
5. **Search** — press `Ctrl+Alt+Space` (or the configured hotkey).
   - Text: type a word you know is written in one of your screenshots.
   - Visual: type what a photo shows, e.g. `a red car at night`.
   - Spoken: type a phrase said in a video, e.g. `quick brown fox`.
6. **Video result** — click it: the player opens two seconds before the matched
   frame. Space pauses, arrow keys seek.
7. **Quit** — tray menu → *Quit*. The icon disappears and the process ends. The
   sandbox can then be closed, which throws everything away.

## 5. What to report

Please note anything that differs from the above, especially:

- a Windows error dialog, a missing DLL name, or an empty search window;
- the model download failing or never finishing (the app retries 3 times, then
  shows a notification with a retry action);
- a video that opens as a black rectangle (codec support differs between real
  hardware and the sandbox's virtual GPU);
- the tray icon not appearing.

Logs are in `%LOCALAPPDATA%\ShotFinder\logs\shotfinder.log` inside the sandbox.
Copy the file out through the mapped folder before closing the sandbox.
