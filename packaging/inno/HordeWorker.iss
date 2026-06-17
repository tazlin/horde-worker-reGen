; Inno Setup script for the AI Horde Worker graphical (double-click) Windows installer.
;
; This is the non-technical install path: a user downloads HordeWorker-Setup.exe, double-clicks it, and
; clicks through a normal Next/Next/Finish wizard, never touching a command line. It is a *wrapper* around
; the same release bundle the one-line installer and winget already use (the CI staging directory), not a
; second implementation: it only lays down files, seeds config, persists the detected GPU backend, and
; creates shortcuts plus an uninstaller. The heavy first-run work (managed Python + PyTorch) is deferred to
; first launch, exactly as today, so the existing browser wizard and Downloads tab handle that UX.
;
; Build locally (see packaging/inno/README.md):
;   iscc packaging\inno\HordeWorker.iss /DStageDir=<path-to-stage> /DMyAppVersion=1.2.3
; CI passes /DStageDir and /DMyAppVersion and an absolute output dir via ISCC's /O switch.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif
; Default points at a sibling "stage" dir at the repo root (../../stage from this .iss); CI overrides it.
#ifndef StageDir
  #define StageDir "..\..\stage"
#endif

[Setup]
; A stable AppId is what lets a later installer upgrade in place and gives a single Add/Remove Programs
; entry. Do not change it once released.
AppId={{2A6B13F1-1070-4369-A40F-4132AE735E60}
AppName=AI Horde Worker
AppVersion={#MyAppVersion}
AppPublisher=Haidra-Org
AppPublisherURL=https://github.com/Haidra-Org/horde-worker-reGen
; Per-user install: {autopf} resolves to %LOCALAPPDATA%\Programs in non-admin mode, so the default
; raises no UAC prompt. The destination page stays enabled so users can move it to another drive (the
; .venv + PyTorch floor is ~10-15 GB, and models add far more). PrivilegesRequired=lowest keeps it that
; way, while ...OverridesAllowed=dialog still lets an admin opt into a machine-wide location.
DefaultDirName={autopf}\AIHordeWorker
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
OutputBaseFilename=HordeWorker-Setup
OutputDir=dist
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; Disclosure pages, sourced from the same staged bundle the one-liner and winget show so they can never
; drift: the plain-language "what/where/from" notice as the Info-before page, and the aggregated
; third-party license notices as the standard accept page.
InfoBeforeFile={#StageDir}\INSTALL_NOTICE.txt
LicenseFile={#StageDir}\THIRD-PARTY-NOTICES.md
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName=AI Horde Worker
; The finished HordeWorker-Setup.exe is signed post-build by the release workflow 
; (.github/workflows/release.yml) using Azure Trusted Signing over OIDC, which keeps
; Local builds are therefore unsigned and will trip SmartScreen; that is expected for dev builds. 
; Do not re-enable an ISCC SignTool here.

[Tasks]
; Shortcuts are opt-in (unchecked by default), matching the one-line installers' conservative default.
Name: "startmenuicon"; Description: "Create a &Start Menu shortcut"; GroupDescription: "Shortcuts (optional):"; Flags: unchecked
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts (optional):"; Flags: unchecked

[Files]
; Everything in the staged bundle, minus detect-backend.ps1 which is handled explicitly below.
Source: "{#StageDir}\*"; DestDir: "{app}"; Excludes: "detect-backend.ps1"; Flags: recursesubdirs ignoreversion
; Install the shared GPU detector for the one-line installer's benefit / future re-detection...
Source: "{#StageDir}\detect-backend.ps1"; DestDir: "{app}"; Flags: ignoreversion
; ...and also make it available to ExtractTemporaryFile so the wizard can detect the GPU before install.
Source: "{#StageDir}\detect-backend.ps1"; Flags: dontcopy
; Seed bridgeData.yaml from the template on a fresh install only, and never remove it on uninstall, so a
; user's API key and worker name survive reinstalls and upgrades (matches install.ps1).
Source: "{#StageDir}\bridgeData_template.yaml"; DestDir: "{app}"; DestName: "bridgeData.yaml"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
Name: "{autoprograms}\AI Horde Worker"; Filename: "{app}\horde-worker.cmd"; WorkingDir: "{app}"; Comment: "AI Horde Worker dashboard"; Tasks: startmenuicon
Name: "{autodesktop}\AI Horde Worker"; Filename: "{app}\horde-worker.cmd"; WorkingDir: "{app}"; Comment: "AI Horde Worker dashboard"; Tasks: desktopicon

[Run]
; Open the dashboard right after install (a checkbox on the Finished page). First launch builds the
; environment and then opens the browser wizard; the user keeps clicking from there.
Filename: "{app}\horde-worker.cmd"; WorkingDir: "{app}"; Description: "Launch AI Horde Worker"; Flags: postinstall nowait skipifsilent shellexec

[UninstallDelete]
; These are generated after install (not tracked by the installer), so remove them explicitly. bridgeData.yaml
; is intentionally left alone. The uv cache, managed Python, and model weights live in the peered "{app}-data"
; sibling folder (set up by runtime.cmd, see worker_bootstrap\paths.py:data_root), which is outside {app} and
; so is never touched here: a reinstall reuses the cached deps and downloaded models.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\bin"

[Code]
var
  DetectedBackend: String;

function RunDetection(): String;
var
  TmpScript, TmpOut: String;
  Token: AnsiString;
  ResultCode: Integer;
begin
  Result := '';
  ExtractTemporaryFile('detect-backend.ps1');
  TmpScript := ExpandConstant('{tmp}\detect-backend.ps1');
  TmpOut := ExpandConstant('{tmp}\backend.txt');
  if Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
          '-NoProfile -ExecutionPolicy Bypass -File "' + TmpScript + '" -OutFile "' + TmpOut + '"',
          '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if LoadStringFromFile(TmpOut, Token) then
      Result := Trim(String(Token));
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  DetectedBackend := RunDetection();

  if DetectedBackend = 'amd-unsupported' then
  begin
    // AMD on Windows has no working GPU backend today (ROCm is Linux-only, DirectML is removed). Offer a
    // CPU-only install rather than silently doing the ~100x-slower thing without saying so.
    if MsgBox('An AMD GPU was detected. Windows GPU acceleration is currently unavailable for AMD cards'
              + ' (ROCm is Linux-only). The worker can still run on your CPU, but that is roughly 100x'
              + ' slower and is mainly useful for testing.' + #13#10#13#10
              + 'Continue with a CPU-only install?', mbConfirmation, MB_YESNO) = IDYES then
      DetectedBackend := 'cpu'
    else
      Result := False;
  end
  else if DetectedBackend = 'cpu' then
  begin
    MsgBox('No NVIDIA or AMD GPU was detected, so the worker will be installed in CPU-only mode. CPU is'
           + ' roughly 100x slower than a GPU and is mainly for testing. If you do have an NVIDIA GPU,'
           + ' install its drivers first and then re-run this installer.', mbInformation, MB_OK);
  end
  else if DetectedBackend = '' then
  begin
    // Detection failed entirely; default to the CUDA build so an NVIDIA box is not silently downgraded.
    // cu126 is the broadest CUDA-12 build of torch 2.12.0 (runs on any CUDA 12.6+ and on CUDA 13 drivers).
    DetectedBackend := 'cu126';
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    // PyTorch and uv fail on paths containing spaces, so reject one early rather than after a long install.
    if Pos(' ', WizardDirValue) > 0 then
    begin
      MsgBox('Please choose a folder whose full path has no spaces. PyTorch and the uv package manager'
             + ' fail on paths that contain spaces.' + #13#10#13#10
             + 'Tip: a short path like C:\AIHordeWorker works well.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    // Persist the detected backend so the deferred first-launch bootstrap (horde-worker.cmd -> runtime.cmd
    // -> bootstrap.py, which reads bin/backend) installs the right PyTorch build instead of defaulting to
    // CUDA on a CPU-only machine. The wizard still detects here because uv (and thus bootstrap.py) does not
    // exist until first launch.
    if DetectedBackend = '' then
      DetectedBackend := 'cu126';
    ForceDirectories(ExpandConstant('{app}\bin'));
    SaveStringToFile(ExpandConstant('{app}\bin\backend'), DetectedBackend, False);
    // Record that consent was captured by the license page so the deferred first-launch sync
    // (horde-worker.cmd -> runtime.cmd -> bootstrap.py) does not prompt the user a second time.
    SaveStringToFile(ExpandConstant('{app}\bin\install-consent'), 'consent recorded (graphical installer)' + #13#10, False);
  end;
end;
