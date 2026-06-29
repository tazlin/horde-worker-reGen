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
; The owner/repo the self-updater should pull future releases from. Recorded into bin\install-info at
; install time. CI can override with /DRepo=<owner>/<repo>; defaults to the canonical production repo.
#ifndef Repo
  #define Repo "Haidra-Org/horde-worker-reGen"
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
; Default Inno behavior on a re-run is to silently reuse the prior directory and skip the destination
; page, so a user can never relocate or choose to remove an existing install. We turn that off and drive
; the decision explicitly from the custom "existing installation" choice page in [Code] instead, which
; either pins {app} to the previous folder (update in place) or lets the destination page show normally
; (move to a new location).
UsePreviousAppDir=no
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
; The Start Menu shortcut is on by default: a graphical installer creating a Start Menu entry is the
; conventional behavior, and it is what lets a non-technical user relaunch the worker without opening the
; install folder and hunting through the wall of scripts. The desktop icon stays opt-in (unchecked) to keep
; the desktop uncluttered for users who do not want it.
Name: "startmenuicon"; Description: "Create a &Start Menu shortcut"; GroupDescription: "Shortcuts:"
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[InstallDelete]
; Remove the worker's Python import roots before [Files] lays down the new bundle, so a reinstall/upgrade
; over an older release cannot leave a renamed or deleted module behind to shadow the new code. Inno's
; [Files] only adds/overwrites; it never prunes files dropped between versions, so without this a stale
; module would persist and be imported. These dirs hold only bundled source (no user state) and are exactly
; the set the self-updater and one-line installers mirror-prune (worker_bootstrap\updater.py _MIRROR_DIRS).
; User state, .venv, bin, and the peered "-data" models/cache live elsewhere and are untouched.
Type: filesandordirs; Name: "{app}\horde_worker_regen"
Type: filesandordirs; Name: "{app}\worker_bootstrap"

[Files]
; Everything in the staged bundle, minus detect-backend.ps1 and release-versions.txt which are handled
; explicitly below. release-versions.txt is only needed at install time (the version picker reads it
; from {tmp}); it is not installed into the application folder.
Source: "{#StageDir}\*"; DestDir: "{app}"; Excludes: "detect-backend.ps1, release-versions.txt"; Flags: recursesubdirs ignoreversion
; Install the shared GPU detector for the one-line installer's benefit / future re-detection...
Source: "{#StageDir}\detect-backend.ps1"; DestDir: "{app}"; Flags: ignoreversion
; ...and also make it available to ExtractTemporaryFile so the wizard can detect the GPU before install.
Source: "{#StageDir}\detect-backend.ps1"; Flags: dontcopy
; The release version list, baked at build time, so the version picker page does not need a live API call.
Source: "{#StageDir}\release-versions.txt"; Flags: dontcopy skipifsourcedoesntexist
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
// The uninstall registry subkey Inno writes for this AppId. Keep in sync with AppId above (the literal
// braces are part of the key name; do not pass this through ExpandConstant, which would treat them as
// Inno constants).
const
  UninstallSubkey = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{2A6B13F1-1070-4369-A40F-4132AE735E60}_is1';

var
  DetectedBackend: String;
  HasPrior: Boolean;
  PriorDir: String;
  PriorUninstaller: String;
  ChoicePage: TInputOptionWizardPage;
  WantsExit: Boolean;
  VersionPage: TWizardPage;
  VersionCombo: TNewComboBox;
  VersionLabel: TNewStaticText;
  SelectedVersion: String;
  ReleasesFetched: Boolean;

const
  VersionDefaultLabel = 'Latest stable (recommended)';
  ReleaseAssetName = 'horde-worker-reGen.zip';

// Read the release version list from the file baked into the installer at build time
// (packaging/fetch-releases.ps1 writes it into the stage before ISCC runs). Each line is a
// release tag, newest first. Populates VersionCombo with the tags.
// Returns True when at least one tag was loaded from the file.
function LoadReleaseVersions(): Boolean;
var
  VersionsPath: String;
  Lines: TArrayOfString;
  i: Integer;
begin
  Result := False;
  ExtractTemporaryFile('release-versions.txt');
  VersionsPath := ExpandConstant('{tmp}\release-versions.txt');
  if not LoadStringsFromFile(VersionsPath, Lines) then
    exit;
  for i := 0 to GetArrayLength(Lines) - 1 do
  begin
    if Trim(Lines[i]) <> '' then
      VersionCombo.Items.Add(Trim(Lines[i]));
  end;
  // >1 because item 0 is the default label already added at page-creation time.
  Result := VersionCombo.Items.Count > 1;
  if Result then
    VersionCombo.ItemIndex := 0;
end;

function ReadPriorFromRoot(RootKey: Integer): Boolean;
var
  Path, Uninst: String;
begin
  Result := False;
  if RegQueryStringValue(RootKey, UninstallSubkey, 'Inno Setup: App Path', Path) and (Path <> '') then
  begin
    PriorDir := Path;
    if RegQueryStringValue(RootKey, UninstallSubkey, 'UninstallString', Uninst) then
      PriorUninstaller := RemoveQuotes(Uninst);
    Result := True;
  end;
end;

function DetectPriorInstall(): Boolean;
begin
  // A per-user (lowest-privilege) install lands in HKCU; an admin machine-wide install lands in the
  // 64-bit HKLM view (ArchitecturesInstallIn64BitMode). Check both so a re-run finds either one.
  Result := ReadPriorFromRoot(HKCU)
            or ReadPriorFromRoot(HKLM64)
            or ReadPriorFromRoot(HKLM);
end;

function RunPriorUninstaller(const Params: String; Show: Integer; WaitForFinish: Boolean): Boolean;
var
  ResultCode, Waited: Integer;
begin
  Result := True;
  if PriorUninstaller = '' then
    exit;
  Exec(PriorUninstaller, Params, '', Show, ewWaitUntilTerminated, ResultCode);
  if WaitForFinish then
  begin
    // The Inno uninstaller relaunches itself from a temp copy so it can delete its own exe, and the first
    // process returns immediately. Exec's wait is therefore not enough; poll until the original unins exe is
    // actually gone (it is deleted only when uninstall completes), bounded so a failed/cancelled uninstall
    // cannot hang the wizard. Removing the ~10-15 GB .venv can take a while, hence the generous bound.
    Waited := 0;
    while FileExists(PriorUninstaller) and (Waited < 600000) do
    begin
      Sleep(500);
      Waited := Waited + 500;
    end;
    Result := not FileExists(PriorUninstaller);
  end;
end;

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
  HasPrior := DetectPriorInstall();
  DetectedBackend := RunDetection();

  if DetectedBackend = 'amd-unsupported' then
  begin
    // Unsupported AMD on Windows: the detector did not match the ROCm Windows compatibility matrix.
    // Offer a CPU-only install rather than silently doing the ~100x-slower thing without saying so.
    if MsgBox('An AMD GPU was detected, but it did not match a supported Windows ROCm profile.'
              + ' The worker can still run on your CPU, but that is roughly 100x slower and is mainly'
              + ' useful for testing.' + #13#10#13#10
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

procedure InitializeWizard();
begin
  // Only meaningful when a prior install exists (ShouldSkipPage hides it otherwise), but the page must be
  // created unconditionally because page IDs are assigned here.
  ChoicePage := CreateInputOptionPage(wpWelcome,
    'Existing installation found',
    'AI Horde Worker is already installed on this computer.',
    'Choose what to do, then click Next.',
    True,    // Exclusive: radio buttons
    False);  // Render as radio buttons, not a list box
  ChoicePage.Add('Update the existing installation in place (recommended)');
  ChoicePage.Add('Move to a new location (removes the current install first)');
  ChoicePage.Add('Uninstall the existing installation and exit');
  ChoicePage.SelectedValueIndex := 0;

  // Version selection page: appears after the license page, before the destination page.
  // The combo box is populated from release-versions.txt, which is baked into the installer
  // at build time (by packaging/fetch-releases.ps1). If the file is absent (offline build),
  // only the default "Latest stable" choice is shown and the install proceeds normally.
  VersionPage := CreateCustomPage(wpLicense,
    'Choose version',
    'Select the release version to install.');
  VersionLabel := TNewStaticText.Create(VersionPage);
  VersionLabel.Caption := 'Version:';
  VersionLabel.Top := ScaleY(0);
  VersionLabel.Left := ScaleX(0);
  VersionLabel.Parent := VersionPage.Surface;

  VersionCombo := TNewComboBox.Create(VersionPage);
  VersionCombo.Top := VersionLabel.Top + VersionLabel.Height + ScaleY(6);
  VersionCombo.Left := ScaleX(0);
  VersionCombo.Width := ScaleX(300);
  VersionCombo.Style := csDropDownList;
  VersionCombo.Parent := VersionPage.Surface;
  VersionCombo.Items.Add(VersionDefaultLabel);
  VersionCombo.ItemIndex := 0;

  // Fetch the release list now (blocks ~1-2 s on the network). If it fails the page
  // still shows the default choice and the install proceeds with the bundled version.
  ReleasesFetched := LoadReleaseVersions();
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  // Hide the choice page entirely on a clean machine.
  if (PageID = ChoicePage.ID) and (not HasPrior) then
    Result := True;
  // "Update in place" reuses the previous folder, so the destination page would only be confusing.
  if (PageID = wpSelectDir) and HasPrior and (ChoicePage.SelectedValueIndex = 0) then
    Result := True;
end;

procedure CancelButtonClick(CurPageID: Integer; var Cancel, Confirm: Boolean);
begin
  // Suppress the "Exit Setup?" prompt for the programmatic exit after an "uninstall and exit" choice.
  if WantsExit then
    Confirm := False;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  if (HasPrior) and (CurPageID = ChoicePage.ID) then
  begin
    case ChoicePage.SelectedValueIndex of
      0: // Update in place: pin {app} to the previous folder; the destination page is then skipped.
        WizardForm.DirEdit.Text := PriorDir;
      1: // Move to a new location: remove the old install now so its files are not orphaned (a single
         // AppId means the new install would otherwise re-point Add/Remove Programs and strand the old
         // folder). The destination page follows so the user picks the new folder.
        begin
          if MsgBox('The existing installation at:' + #13#10 + '    ' + PriorDir + #13#10#13#10
                    + 'will be removed first, then you can choose a new folder. Downloaded models and the'
                    + ' dependency cache live in the sibling "' + ExtractFileName(PriorDir) + '-data" folder'
                    + ' and are NOT moved, so the new location will re-download them on first launch.'
                    + #13#10#13#10 + 'Continue?', mbConfirmation, MB_YESNO) = IDYES then
          begin
            if RunPriorUninstaller('/SILENT /SUPPRESSMSGBOXES /NORESTART', SW_HIDE, True) then
              HasPrior := False // The old install is gone; treat the rest of setup as a fresh install.
            else
            begin
              // Bail rather than install a second copy on top of a half-removed one (which would also leave
              // the Add/Remove Programs entry pointing at the wrong folder).
              MsgBox('The existing installation could not be removed automatically. Please uninstall it from'
                     + ' Settings > Apps, then run this installer again.', mbError, MB_OK);
              Result := False;
            end;
          end
          else
            Result := False; // Stay on the choice page.
        end;
      2: // Uninstall and exit: run the existing uninstaller interactively, then close the wizard.
        begin
          RunPriorUninstaller('', SW_SHOW, False);
          WantsExit := True;
          Result := False;
          WizardForm.Close;
        end;
    end;
    Exit;
  end;

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

  if CurPageID = VersionPage.ID then
  begin
    // Capture the selected version. An empty string means "use the bundled version" (the
    // default label was selected). Otherwise the value is the GitHub release tag to download.
    if (VersionCombo.ItemIndex > 0) and (VersionCombo.ItemIndex < VersionCombo.Items.Count) then
      SelectedVersion := VersionCombo.Items[VersionCombo.ItemIndex]
    else
      SelectedVersion := '';
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  DownloadUrl, TmpZip, ExtractDir, psCmd, AppDir: String;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    AppDir := ExpandConstant('{app}');

    // Persist the detected backend so the deferred first-launch bootstrap (horde-worker.cmd -> runtime.cmd
    // -> bootstrap.py, which reads bin/backend) installs the right PyTorch build instead of defaulting to
    // CUDA on a CPU-only machine. The wizard still detects here because uv (and thus bootstrap.py) does not
    // exist until first launch.
    if DetectedBackend = '' then
      DetectedBackend := 'cu126';
    ForceDirectories(AppDir + '\bin');
    SaveStringToFile(AppDir + '\bin\backend', DetectedBackend, False);
    // Record that consent was captured by the license page so the deferred first-launch sync
    // (horde-worker.cmd -> runtime.cmd -> bootstrap.py) does not prompt the user a second time.
    SaveStringToFile(AppDir + '\bin\install-consent', 'consent recorded (graphical installer)' + #13#10, False);
    // Record how this worker was installed and from where, so the self-updater can keep the Add/Remove
    // Programs version honest and pull future releases from the right origin.
    SaveStringToFile(AppDir + '\bin\install-info', 'method=exe' + #13#10 + 'repo={#Repo}' + #13#10, False);

    // If the user selected a specific version (not the default "Latest stable"), download that
    // release's bundle and overlay it onto the install. The bundled (latest) files are already in
    // place from [Files]; the overlay replaces them with the chosen version's source. This reuses
    // the same download URL pattern as the one-line installer.
    if SelectedVersion <> '' then
    begin
      DownloadUrl := 'https://github.com/{#Repo}/releases/download/' + SelectedVersion + '/' + ReleaseAssetName;
      TmpZip := ExpandConstant('{tmp}\' + ReleaseAssetName);
      ExtractDir := ExpandConstant('{tmp}\version-bundle');

      // Download the release zip via PowerShell (Invoke-WebRequest, pre-installed on Windows).
      psCmd := '-NoProfile -ExecutionPolicy Bypass -Command "' +
        '$ProgressPreference = ''SilentlyContinue'';' +
        'try {' +
        '  Invoke-WebRequest -Uri ''' + DownloadUrl + ''' -OutFile ''' + TmpZip + ''' -UseBasicParsing -ErrorAction Stop' +
        '} catch { exit 1 }"';

      if Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
              psCmd, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0) then
      begin
        // Extract using in-box tar.exe (Windows 10 1803+).
        ForceDirectories(ExtractDir);
        if Exec(ExpandConstant('{sys}\tar.exe'),
                '-xf "' + TmpZip + '" -C "' + ExtractDir + '"',
                '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0) then
        begin
          // Re-prune the Python import roots before copying the selected version's files
          // so modules deleted between the bundled (latest) version and the chosen version
          // do not linger and shadow the selected code. This mirrors what [InstallDelete]
          // already did for the initial bundle lay-down.
          DelTree(AppDir + '\horde_worker_regen', True, True, True);
          DelTree(AppDir + '\worker_bootstrap', True, True, True);
          // Copy extracted files over the install, overwriting the bundled version.
          Exec(ExpandConstant('{sys}\xcopy.exe'),
               '"' + ExtractDir + '\*" "' + AppDir + '" /E /I /Y /Q /H',
               '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
        end;
      end;
      // Best-effort: if the download or extraction fails, the bundled (latest) files remain
      // in place and the install proceeds. The failure is silent because the first launch will
      // self-update to the latest version anyway, and a failed explicit version choice is not
      // worse than the bundled default.
    end;
  end;
end;
