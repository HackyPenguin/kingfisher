# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_all

datas = [('models', 'models'), ('kestrel_telemetry.py', '.'), ('folder_inspector.py', '.'), ('cli.py', '.'), ('VERSION.txt', '.'), ('kestrel_analyzer', 'kestrel_analyzer'), ('visualizer.html', '.'), ('visualizer.css', '.'), ('visualizer.js', '.'), ('csv_parser.js', '.'), ('culling.html', '.'), ('logo.png', '.'), ('logo.ico', '.'), ('settings_utils.py', '.'), ('editor_launch.py', '.'), ('queue_manager.py', '.'), ('api_bridge.py', '.')]
binaries = []
hiddenimports = ['pywebview', 'certifi','PIL','exifread','settings_utils','editor_launch','queue_manager','api_bridge']
binaries += collect_dynamic_libs('torch')
binaries += collect_dynamic_libs('onnxruntime')
binaries += collect_dynamic_libs('tensorflow')
tmp_ret = collect_all('msvc-runtime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['visualizer.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['runtime_hook.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Kingfisher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon='logo_peregrine.icns',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Kingfisher',
    icon='logo_peregrine.icns',
)

app = BUNDLE(
    coll,
    name='Kingfisher.app',
    icon='logo_peregrine.icns',
    bundle_identifier='org.Kingfisher',
)
