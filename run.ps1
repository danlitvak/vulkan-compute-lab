# Build and run a shader by (partial) name.
#
#   .\run grav              galaxy.comp, matched through its alias
#   .\run mandel            mandelbrot.comp, matched as a prefix
#   .\run                   list what is available
#   .\run grav -Release     optimised build, no validation layers
#   .\run grav -Clean       wipe the build directory first
#   .\run grav --particles 65536      anything unrecognised goes to lab.exe
#
# Names resolve against each shader's filename and any //!alias line in its
# header, trying exact, then prefix, then substring, then subsequence. An
# ambiguous name lists the candidates instead of guessing.
#
# ASCII only on purpose: Windows PowerShell 5.1 reads unmarked .ps1 files as
# ANSI, so a stray UTF-8 character here becomes a parse error.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Name,

    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')]
    [string]$Config = 'Debug',

    [switch]$Clean,
    [switch]$List,

    # Everything else is forwarded to lab.exe untouched.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# --- catalogue ----------------------------------------------------------------

function Get-Shaders {
    Get-ChildItem (Join-Path $root 'shaders') -Filter *.comp -File | ForEach-Object {
        $aliases = @()
        $simulation = $false
        foreach ($line in (Get-Content $_.FullName -TotalCount 40)) {
            $trimmed = $line.TrimStart()
            if ($trimmed -match '^//!alias\s+(.+)$') {
                $aliases += ($matches[1] -split '[\s,]+' | Where-Object { $_ })
            }
            if ($trimmed -match '^//!nbody\s+\d+') { $simulation = $true }
        }
        [pscustomobject]@{
            Name       = $_.BaseName
            Path       = $_.FullName
            Aliases    = $aliases
            Simulation = $simulation
        }
    }
}

function Show-Shaders($shaders) {
    Write-Host ""
    Write-Host "shaders in $($root)\shaders:" -ForegroundColor Cyan
    foreach ($s in $shaders) {
        $tag = if ($s.Simulation) { "[simulation]" } else { "" }
        $also = if ($s.Aliases.Count) { "aka " + ($s.Aliases -join ', ') } else { "" }
        Write-Host ("  {0,-13} {1,-13} {2}" -f $s.Name, $tag, $also)
    }
    Write-Host ""
    # Reachable both as .\run from the project and as vklab from the profile
    # function, so the hint names both rather than guessing which one was used.
    Write-Host "usage: .\run <name>   or   vklab <name>      e.g. $($shaders[0].Name)" -ForegroundColor DarkGray
}

# Exact, then prefix, then substring, then subsequence ("mbrot" -> mandelbrot).
function Resolve-Shader($shaders, [string]$query) {
    $q = $query.ToLowerInvariant()

    $keys = { param($s) @($s.Name) + $s.Aliases | ForEach-Object { $_.ToLowerInvariant() } }

    $exact = $shaders | Where-Object { (& $keys $_) -contains $q }
    if ($exact) { return @($exact) }

    $prefix = $shaders | Where-Object { (& $keys $_) | Where-Object { $_.StartsWith($q) } }
    if ($prefix) { return @($prefix) }

    $substring = $shaders | Where-Object { (& $keys $_) | Where-Object { $_.Contains($q) } }
    if ($substring) { return @($substring) }

    $subsequence = $shaders | Where-Object {
        foreach ($key in (& $keys $_)) {
            $i = 0
            foreach ($ch in $q.ToCharArray()) {
                $i = $key.IndexOf($ch, $i)
                if ($i -lt 0) { break }
                $i++
            }
            if ($i -ge 0) { return $true }
        }
        return $false
    }
    return @($subsequence)
}

# --- resolve ------------------------------------------------------------------

$shaders = @(Get-Shaders)
if ($shaders.Count -eq 0) { throw "no .comp files found in $root\shaders" }

if ($List -or -not $Name) {
    Show-Shaders $shaders
    exit 0
}

$matched = @(Resolve-Shader $shaders $Name)
if ($matched.Count -eq 0) {
    Write-Host "no shader matches '$Name'." -ForegroundColor Red
    Show-Shaders $shaders
    exit 1
}
if ($matched.Count -gt 1) {
    Write-Host "'$Name' is ambiguous: $(($matched.Name) -join ', ')" -ForegroundColor Red
    exit 1
}

$shader = $matched[0]

# --- build --------------------------------------------------------------------

$buildDir = Join-Path $root "build\$Config"
if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "[run] cleaning $buildDir" -ForegroundColor DarkGray
    Remove-Item $buildDir -Recurse -Force
}

& (Join-Path $root 'build.ps1') -Config $Config
if ($LASTEXITCODE -ne 0) { throw "build failed" }

# --- run ----------------------------------------------------------------------

$exe = Join-Path $buildDir 'lab.exe'
if (-not (Test-Path $exe)) { throw "built, but $exe is missing" }

Write-Host "[run] $($shader.Name)" -ForegroundColor Green
$labArgs = @($shader.Path) + @($Rest | Where-Object { $_ })

# Back to Continue before handing over. PowerShell 5.1 wraps a native program's
# stderr in ErrorRecords, so with ErrorActionPreference = Stop the validation
# layer's warnings would abort the script on a perfectly normal launch.
$ErrorActionPreference = 'Continue'
& $exe @labArgs
exit $LASTEXITCODE
