; Two installer variants from this script:
;   iscc InnoSetupInstallerBuild.iss
;       → Output\MilanaSetup.exe
;         (BLIP + EasyOCR packed; optional Tasks checkbox at install)
;   iscc /DNoModels InnoSetupInstallerBuild.iss
;       → Output\MilanaSetup-nomodels.exe
;         (no image-model weights in the installer, no prompt)
; Or run compile_innosetup.bat to build both.

[Setup]
AppName=Milana
AppVersion=2026.08
AppPublisher=iishnitsa
AppPublisherURL=https://github.com/iishnitsa/milana
DefaultDirName={userpf}\Milana
DefaultGroupName=Milana
#ifdef NoModels
OutputBaseFilename=MilanaSetup-nomodels
#else
OutputBaseFilename=MilanaSetup
#endif
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\data\icons\icon.ico
UninstallDisplayIcon={app}\data\icons\icon.ico
AppID={{1df4711d-8269-4498-abc2-ba6400484db6}}
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
DisableDirPage=no
ExtraDiskSpaceRequired=0
AllowRootDirectory=no
AllowUNCPath=no
UsedUserAreasWarning=no
AllowNetworkDrive=no
AllowNoIcons=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"
Name: "startmenuicon"; Description: "Create a Start Menu shortcut"; GroupDescription: "Additional icons:"
#ifndef NoModels
; Optional ~1GB BLIP + EasyOCR weights for image OCR/caption
Name: "models"; Description: "Install image recognition models (OCR / captions, ~1 GB)"; GroupDescription: "Optional components:"; Flags: checkedonce
#endif

[Files]
; App (exclude models from recursive data so they are optional)
; Exclude data\chats AND top-level chats (would land as {app}\chats next to exe)
; Also never ship milana_boot.log (debug leftover)
Source: "..\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs; Excludes: "__pycache__, __pycache__\*, tests, install\Output, install\Output\*, mvenv, mvenv\*, *.pyc, *.pyo, *.pyd, launcher.py, data\settings.db, run_ui.cmd, run_ui.sh, *.lnk, build, build\*, dist, dist\*, Milana.lnk, Output, Output\*, .git, .git\*, .gitattributes, .vscode, .vscode\*, .idea, .idea\*, *.log, milana_boot.log, data\milana_boot.log, *.bak, *.tmp, thumbs.db, *.db, data\chats, chats, chats\*, data\models, launch_milana.cmd, run_milana.sh, *.run, build_installer"

Source: "..\_internal\*"; DestDir: "{app}\_internal"; Flags: recursesubdirs createallsubdirs

Source: "..\data\*"; DestDir: "{app}\data"; Flags: recursesubdirs createallsubdirs; Excludes: "*.db, chats, models"

Source: "..\data\icons\icon.ico"; DestDir: "{app}\data\icons"; Flags: skipifsourcedoesntexist
Source: "..\data\icons\icon.png"; DestDir: "{app}\data\icons"; Flags: skipifsourcedoesntexist

#ifndef NoModels
; Models only if task selected (omitted entirely from the -nomodels build)
Source: "..\data\models\*"; DestDir: "{app}\data\models"; Flags: recursesubdirs createallsubdirs skipifsourcedoesntexist; Tasks: models
#endif

[Dirs]
Name: "{app}\data\chats"
Name: "{app}\data\models"

[Icons]
Name: "{group}\Milana"; Filename: "{app}\Milana.exe"; IconFilename: "{app}\data\icons\icon.ico"; Tasks: startmenuicon
Name: "{userdesktop}\Milana"; Filename: "{app}\Milana.exe"; IconFilename: "{app}\data\icons\icon.ico"; Tasks: desktopicon
Name: "{group}\Uninstall Milana"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\Milana.exe"; Description: "Launch Milana"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\data"
