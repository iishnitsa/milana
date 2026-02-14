; ===========================================
; Inno Setup Script for Milana Application
; ============================================

[Setup]
AppName=Milana
AppVersion=2026.02
AppPublisher=iishnitsa
AppPublisherURL=https://github.com/iishnitsa/milana
DefaultDirName={pf}\Milana
DefaultGroupName=Milana
OutputBaseFilename=MilanaSetup
Compression=lzma2
SolidCompression=yes
SetupIconFile=..\data\icons\icon.ico
UninstallDisplayIcon={app}\data\icons\icon.ico
AppID={{74bec6ac-6270-4f46-a420-eb3a0d86a788}}
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

DisableDirPage=no
ExtraDiskSpaceRequired=0
AllowRootDirectory=no
AllowUNCPath=no
UsedUserAreasWarning=no

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; GroupDescription: "Additional icons:"
Name: "startmenuicon"; Description: "Create a Start Menu shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs; Excludes: "__pycache__, __pycache__\*, tests, install, install\*, mvenv, mvenv\*, *.pyc, *.pyo, *.pyd, launcher.py, data\settings.db, README.md, run_ui.cmd, run_ui.sh, *.lnk, build, build\*, dist, dist\*, *.iss, Milana.lnk, Output, Output\*, .git, .git\*, .gitignore, .gitattributes, .vscode, .vscode\*, .idea, .idea\*, *.log, *.bak, *.tmp, thumbs.db, requirements.txt, *.db, data\chats"

Source: "..\_internal\*"; DestDir: "{app}\_internal"; Flags: recursesubdirs createallsubdirs

Source: "..\data\*"; DestDir: "{app}\data"; Flags: recursesubdirs createallsubdirs; Excludes: "*.db, chats"

Source: "..\data\icons\icon.ico"; DestDir: "{app}\data\icons"; Flags: skipifsourcedoesntexist
Source: "..\data\icons\icon.png"; DestDir: "{app}\data\icons"; Flags: skipifsourcedoesntexist

[Dirs]
Name: "{app}\data\chats"

[Icons]
Name: "{group}\Milana"; Filename: "{app}\Milana.exe"; IconFilename: "{app}\data\icons\icon.ico"; Tasks: startmenuicon
Name: "{userdesktop}\Milana"; Filename: "{app}\Milana.exe"; IconFilename: "{app}\data\icons\icon.ico"; Tasks: desktopicon
Name: "{group}\Uninstall Milana"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\Milana.exe"; Description: "Launch Milana"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\data"