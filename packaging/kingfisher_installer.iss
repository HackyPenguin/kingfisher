; Kingfisher Installer - Inno Setup Script
; Packages the Kingfisher desktop application

#define MyAppName "Kingfisher"
#define MyAppPublisher "Kingfisher"
#define MyAppURL "https://github.com/HackyPenguin/kingfisher"
#define TutorialURL "https://github.com/HackyPenguin/kingfisher/blob/main/README.md#tutorial-5-steps-to-better-culling"

#ifndef AppVersion
  #define AppVersion "alpha-YYYY.MM.DD.HH.MM"
#endif

#ifndef ReleaseName
  #define ReleaseName "Kingfisher aYYYY.MM.DD.HH.MM"
#endif

#ifndef ReleaseDir
  #define ReleaseDir "..\\release"
#endif

; WebView2 runtime installation removed to reduce installer failures

[Setup]
AppId=org.Kingfisher
AppName={#MyAppName}
AppVersion={#AppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\dist\installer
OutputBaseFilename={#ReleaseName}-{#AppVersion}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
DisableProgramGroupPage=yes
WizardImageFile=..\assets\logo.png
WizardSmallImageFile=..\assets\logo.png

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon_kingfisher"; Description: "Create desktop shortcut for Kingfisher"; GroupDescription: "Desktop shortcuts:"; Flags: checkedonce

[Files]
; Unified Kingfisher bundle (one-dir from PyInstaller)
Source: "..\analyzer\dist\Kingfisher\*"; DestDir: "{app}\Kingfisher"; Flags: recursesubdirs createallsubdirs ignoreversion

; Documentation
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu icon (single unified app)
Name: "{group}\Kingfisher"; Filename: "{app}\Kingfisher\Kingfisher.exe"; WorkingDir: "{app}\Kingfisher"; IconFilename: "{app}\Kingfisher\_internal\\logo.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

; Desktop icon
Name: "{autodesktop}\Kingfisher"; Filename: "{app}\Kingfisher\Kingfisher.exe"; WorkingDir: "{app}\Kingfisher"; Tasks: desktopicon_kingfisher; IconFilename: "{app}\Kingfisher\_internal\\logo.ico"

[Run]
; Open tutorial webpage after install
Filename: "{#TutorialURL}"; Description: "View online tutorial"; Flags: shellexec postinstall skipifsilent nowait

; Option to launch after install (unified)
Filename: "{app}\Kingfisher\Kingfisher.exe"; Description: "Launch Kingfisher"; Flags: nowait postinstall skipifsilent unchecked

[Code]
// WebView2 installer integration removed to avoid forcing downloads during setup.
// No-op InitializeWizard kept for compatibility.
procedure InitializeWizard;
begin
  // Intentionally empty — installer will not download WebView2.
end;
