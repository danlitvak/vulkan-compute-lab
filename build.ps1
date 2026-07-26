# Configures and builds the lab.
#
# CMake, Ninja and the MSVC toolchain all ship inside VS Build Tools but none of
# them are on PATH by default on this machine, so this script locates the VS
# install, imports the vcvars64 environment, and then builds.
#
#   .\build.ps1                     # debug build (validation layers on)
#   .\build.ps1 -Config Release     # optimised, no validation
#   .\build.ps1 -Run                # build, then run
#   .\build.ps1 -Run -Shader shaders\warp.comp
#
# ASCII only on purpose: Windows PowerShell 5.1 reads unmarked .ps1 files as
# ANSI, so a stray UTF-8 character here becomes a parse error.

[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')]
    [string]$Config = 'Debug',
    [switch]$Run,
    [string]$Shader
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$buildDir = Join-Path $root "build\$Config"

# --- MSVC environment ---------------------------------------------------------

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Is Visual Studio or Build Tools installed?" }

$vcvars = & $vswhere -latest -products * -find 'VC\Auxiliary\Build\vcvars64.bat' | Select-Object -First 1
if (-not $vcvars) { throw "vcvars64.bat not found. Install the 'Desktop development with C++' workload." }

Write-Host "[build] using $vcvars" -ForegroundColor DarkGray
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:\$($matches[1])" -Value $matches[2] }
}

# --- CMake + Ninja ------------------------------------------------------------

function Resolve-Tool([string]$name, [string[]]$fallbacks) {
    $found = Get-Command $name -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    foreach ($candidate in $fallbacks) {
        $hit = Get-Item $candidate -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    throw "$name not found. Add the 'C++ CMake tools for Windows' component."
}

$vsRoot = & $vswhere -latest -products * -property installationPath | Select-Object -First 1
$cmakeExe = Resolve-Tool 'cmake' @("$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
$ninjaExe = Resolve-Tool 'ninja' @("$vsRoot\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe")

# The SDK installer sets VULKAN_SDK machine-wide, but shells started before the
# install (and the cmd environment imported above) will not have picked it up.
if (-not $env:VULKAN_SDK) {
    $machineSdk = [Environment]::GetEnvironmentVariable('VULKAN_SDK', 'Machine')
    if (-not $machineSdk) {
        $machineSdk = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
    }
    if ($machineSdk) { $env:VULKAN_SDK = $machineSdk }
}
if (-not $env:VULKAN_SDK) { throw "VULKAN_SDK is not set. Install the LunarG Vulkan SDK." }
Write-Host "[build] Vulkan SDK: $env:VULKAN_SDK" -ForegroundColor DarkGray

# The -D arguments must be quoted: PowerShell does not expand a variable inside
# a bare token that starts with '-', so an unquoted -DFOO=$Config reaches CMake
# as the literal string.
& $cmakeExe -S $root -B $buildDir -G Ninja "-DCMAKE_BUILD_TYPE=$Config" "-DCMAKE_MAKE_PROGRAM=$ninjaExe"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

& $cmakeExe --build $buildDir
if ($LASTEXITCODE -ne 0) { throw "build failed" }

$exe = Join-Path $buildDir 'lab.exe'
Write-Host "[build] ok: $exe" -ForegroundColor Green

if ($Run) {
    if ($Shader) { & $exe (Resolve-Path $Shader) } else { & $exe }
}
