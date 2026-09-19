<#
.SYNOPSIS
    One entry point for the whole system on Windows.

.DESCRIPTION
    Every command sets PYTHONPATH and MLFLOW_TRACKING_URI itself, so there is nothing to
    remember and nothing to get wrong. Setting `set PYTHONPATH=src` in PowerShell silently
    does nothing and the next command dies with ModuleNotFoundError -- that class of
    problem is what this script exists to remove.

.EXAMPLE
    .\run.ps1 setup         # venv + dependencies (once, ~10 min)
    .\run.ps1 all           # ingest -> train -> serve, end to end
    .\run.ps1 mlflow        # tracking UI on http://127.0.0.1:5000
    .\run.ps1 serve         # ranking endpoint on http://127.0.0.1:8088
    .\run.ps1 status        # what is running, what is registered
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'ingest', 'train', 'evaluate', 'mlflow', 'serve', 'predict',
                 'loadtest', 'test', 'rollback', 'status', 'all', 'stop', 'help')]
    [string]$Command = 'help',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Py = Join-Path $Root '.venv\Scripts\python.exe'
$TrackingUri = 'http://127.0.0.1:5000'
$ServePort = 8088

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    $msg" -ForegroundColor Yellow }

function Assert-Venv {
    if (-not (Test-Path $Py)) {
        throw "No virtualenv found. Run:  .\run.ps1 setup"
    }
}

function Use-Env {
    $env:PYTHONPATH = Join-Path $Root 'src'
    if (-not $env:MLFLOW_TRACKING_URI) { $env:MLFLOW_TRACKING_URI = $TrackingUri }
    # Leftovers from a Databricks session would silently redirect a local run.
    Remove-Item Env:\MLFLOW_REGISTRY_URI -ErrorAction SilentlyContinue
    Remove-Item Env:\DATABRICKS_CONFIG_PROFILE -ErrorAction SilentlyContinue
}

function Test-Endpoint($url, $timeoutSec = 3) {
    try {
        $null = Invoke-WebRequest -Uri $url -TimeoutSec $timeoutSec -UseBasicParsing
        return $true
    } catch { return $false }
}

function Wait-Until($url, $label, $maxSeconds) {
    Write-Host "    waiting for $label " -NoNewline
    for ($i = 0; $i -lt $maxSeconds; $i += 3) {
        if (Test-Endpoint $url) { Write-Host ""; Write-Ok "$label ready"; return $true }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 3
    }
    Write-Host ""
    Write-Warn "$label did not come up within ${maxSeconds}s"
    return $false
}

switch ($Command) {

    'setup' {
        Write-Step 'Creating virtualenv (Python 3.12 required)'
        # 3.12 is not a preference: the researcher's bl_models_train.py uses PEP 701
        # f-strings, which are a SyntaxError on 3.11 and earlier.
        $py312 = @(
            "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
            "C:\Program Files\Python312\python.exe"
        ) | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $py312) {
            try { $py312 = (py -3.12 -c "import sys; print(sys.executable)" 2>$null) } catch {}
        }
        if (-not $py312) {
            throw "Python 3.12 not found. Install it from python.org, then re-run. The researcher's script requires >= 3.12 (PEP 701 f-strings)."
        }
        Write-Ok "using $py312"
        & $py312 -m venv (Join-Path $Root '.venv')
        Write-Step 'Installing dependencies (~10 min; torch is the long pole)'
        & $Py -m pip install --upgrade pip --quiet
        & $Py -m pip install -r (Join-Path $Root 'requirements-torch.txt')
        & $Py -m pip install -r (Join-Path $Root 'requirements.txt')
        Write-Ok 'setup complete'
        Write-Host "`nNext:  .\run.ps1 all" -ForegroundColor Cyan
    }

    'ingest' {
        Assert-Venv; Use-Env
        Write-Step 'CSV -> Delta table'
        & $Py -m bl_ranker.data.ingest @Rest
    }

    'train' {
        Assert-Venv; Use-Env
        Write-Step 'Production training (trains on all data, registers @champion)'
        Write-Warn 'takes ~2-3 min; no progress output until it finishes'
        & $Py -m bl_ranker.training.run --mode production @Rest
    }

    'evaluate' {
        Assert-Venv; Use-Env
        Write-Step 'Train/test evaluation (writes no artifacts, by design)'
        Write-Warn 'takes ~8 min; 80% of that is TabPFN scoring the held-out week'
        & $Py -m bl_ranker.training.run --mode train_test @Rest
    }

    'mlflow' {
        Assert-Venv
        if (Test-Endpoint "$TrackingUri/health") { Write-Ok "already running at $TrackingUri"; break }
        Write-Step "Starting MLflow at $TrackingUri"
        $store = Join-Path $Root 'mlflow_local'
        New-Item -ItemType Directory -Force -Path (Join-Path $store 'artifacts') | Out-Null
        Start-Process -FilePath $Py -WindowStyle Minimized -ArgumentList @(
            '-m','mlflow','server','--host','127.0.0.1','--port','5000',
            '--backend-store-uri',"sqlite:///$($store -replace '\\','/')/mlflow.db",
            '--artifacts-destination',"$($store -replace '\\','/')/artifacts",'--serve-artifacts'
        )
        Wait-Until "$TrackingUri/health" 'MLflow' 90 | Out-Null
        Write-Host "    open $TrackingUri" -ForegroundColor Cyan
    }

    'serve' {
        Assert-Venv; Use-Env
        if (Test-Endpoint "http://127.0.0.1:$ServePort/health") {
            Write-Ok "already running at http://127.0.0.1:$ServePort"; break
        }
        Write-Step "Starting ranking endpoint on port $ServePort"
        Write-Warn 'cold start is ~110s: loading @champion and fitting the TabPFN context'
        $env:TORCH_NUM_THREADS = '6'
        Start-Process -FilePath $Py -WindowStyle Minimized -ArgumentList @(
            '-m','uvicorn','bl_ranker.serving.app:app',
            '--host','127.0.0.1','--port',"$ServePort",'--no-access-log'
        )
        if (Wait-Until "http://127.0.0.1:$ServePort/health" 'endpoint' 240) {
            Write-Host "    docs: http://127.0.0.1:$ServePort/docs" -ForegroundColor Cyan
        }
    }

    'predict' {
        Assert-Venv
        if (-not (Test-Endpoint "http://127.0.0.1:$ServePort/health")) {
            throw "Endpoint is not running. Start it with:  .\run.ps1 serve"
        }
        Write-Step 'Ranking the example user'
        $body = Get-Content (Join-Path $Root 'examples\user.json') -Raw
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$ServePort/rank" -Method Post `
                               -ContentType 'application/json' -Body $body
        $r.ranking.PSObject.Properties |
            Sort-Object { $_.Value.rank } |
            ForEach-Object {
                "{0,2}. {1,-26} `${2,8:N2}" -f $_.Value.rank, $_.Name, $_.Value.expected_payout
            }
        Write-Host "`n    model version $($r.model_version) | $($r.latency_ms) ms" -ForegroundColor Cyan
    }

    'loadtest' {
        Assert-Venv; Use-Env
        if (-not (Test-Endpoint "http://127.0.0.1:$ServePort/health")) {
            throw "Endpoint is not running. Start it with:  .\run.ps1 serve"
        }
        Write-Step 'Load simulation'
        $args = if ($Rest) { $Rest } else { @('--requests','20','--concurrency','1','2','4','8') }
        & $Py (Join-Path $Root 'loadtest\simulate.py') --url "http://127.0.0.1:$ServePort/rank" @args
    }

    'test' {
        Assert-Venv; Use-Env
        Write-Step 'Test suite'
        & $Py -m pytest -q @Rest
    }

    'rollback' {
        Assert-Venv; Use-Env
        & $Py -m bl_ranker.rollback @(if ($Rest) { $Rest } else { '--list' })
    }

    'status' {
        Assert-Venv; Use-Env
        Write-Step 'Status'
        $mlflowUp = Test-Endpoint "$TrackingUri/health"
        $serveUp = Test-Endpoint "http://127.0.0.1:$ServePort/health"
        Write-Host ("    MLflow    : {0}  {1}" -f $(if ($mlflowUp) {'UP  '} else {'down'}), $TrackingUri)
        Write-Host ("    Endpoint  : {0}  http://127.0.0.1:{1}" -f $(if ($serveUp) {'UP  '} else {'down'}), $ServePort)
        $delta = Join-Path $Root 'data\delta\bl_sessions'
        Write-Host ("    Delta table: {0}" -f $(if (Test-Path $delta) {'present'} else {'not ingested'}))
        if ($mlflowUp) { & $Py -m bl_ranker.rollback --list }
    }

    'all' {
        & $PSCommandPath mlflow
        & $PSCommandPath ingest
        & $PSCommandPath train
        & $PSCommandPath serve
        & $PSCommandPath predict
        Write-Host "`nEverything is up." -ForegroundColor Green
        Write-Host "  MLflow   $TrackingUri"
        Write-Host "  Endpoint http://127.0.0.1:$ServePort/docs"
    }

    'stop' {
        Write-Step 'Stopping MLflow and the endpoint'
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $_.CommandLine -match 'mlflow server|uvicorn bl_ranker' } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        Write-Ok 'stopped'
    }

    default {
        Write-Host @"
BL Brand Ranking

  .\run.ps1 setup       create the venv and install dependencies (once)
  .\run.ps1 all         ingest -> train -> serve -> one prediction

  .\run.ps1 mlflow      start the tracking UI      ($TrackingUri)
  .\run.ps1 ingest      load the CSV into Delta
  .\run.ps1 train       production run, registers @champion   (~2-3 min)
  .\run.ps1 evaluate    train/test accuracy run               (~8 min)
  .\run.ps1 serve       start the ranking endpoint            (~110s cold start)
  .\run.ps1 predict     rank the example user
  .\run.ps1 loadtest    p50/p95/p99 against the endpoint
  .\run.ps1 test        run the test suite
  .\run.ps1 rollback    --list | --to-previous | --to-version N
  .\run.ps1 status      what is running and what is registered
  .\run.ps1 stop        stop MLflow and the endpoint
"@
    }
}
