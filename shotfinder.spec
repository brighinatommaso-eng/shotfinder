# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller one-folder build for ShotFinder.

Run through ``tools/build.ps1``; the spec is committed so the release build is
reproducible. Models are never bundled: the app downloads them to
``%LOCALAPPDATA%\\ShotFinder\\models`` on first run.

Two binary sets need explicit help:

* ``onnxruntime`` and ``ctranslate2`` ship their DLLs inside the package and
  load them by path at import time, so they must stay in the same relative
  folder inside ``_internal``.
* ``sqlite_vec`` loads ``vec0.dll`` from its own package directory.
"""

import sqlite_vec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

# SPECPATH is the directory that contains this spec file.
ROOT = Path(SPECPATH)

# cudnn64_9.dll is a 250 MB CUDA dependency; ShotFinder runs on the CPU.
binaries = [
    *collect_dynamic_libs("onnxruntime"),
    *(
        item
        for item in collect_dynamic_libs("ctranslate2")
        if not item[0].lower().endswith("cudnn64_9.dll")
    ),
]

datas = [
    (str(Path(sqlite_vec.__file__).with_name("vec0.dll")), "sqlite_vec"),
    # faster-whisper resolves the Silero VAD model relative to its package
    # directory; without it every transcription fails at runtime.
    *collect_data_files("faster_whisper"),
]

hiddenimports = [
    # winrt modules are imported lazily inside the OCR extractor.
    *collect_submodules("winrt"),
    "imageio_ffmpeg.binaries",
]

# Qt modules the app never imports; keeping them out saves hundreds of MB.
excluded_qt = [
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtGraphs",
    "PySide6.QtGraphsWidgets",
    "PySide6.QtHelp",
    "PySide6.QtHttpServer",
    "PySide6.QtLocation",
    "PySide6.QtNfc",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSql",
    "PySide6.QtStateMachine",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtUiTools",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
    "PySide6.QtXml",
]

excluded = [
    # Export-only toolchain (torch/transformers) and test runners.
    "torch",
    "torchvision",
    "transformers",
    "sentence_transformers",
    "onnx",
    "accelerate",
    "sklearn",
    "scipy",
    "pandas",
    "matplotlib",
    "IPython",
    "pytest",
    "tests",
    "tkinter",
    *excluded_qt,
]

a = Analysis(
    [str(ROOT / "tools" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ShotFinder",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "tools" / "shotfinder.ico"),
    version=str(ROOT / "tools" / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ShotFinder",
)
