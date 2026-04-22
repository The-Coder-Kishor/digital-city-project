param(
  [string]$StartId = '000001',
  [string]$EndId = '999999'
)

$ErrorActionPreference = 'Stop'

function Test-TenderId {
  param([string]$Value)
  return $Value -match '^[0-9]{6}$'
}

function Get-MaxParallel {
  if ($env:MAX_PARALLEL) {
    if ($env:MAX_PARALLEL -notmatch '^[1-9][0-9]*$') {
      throw "Invalid MAX_PARALLEL: $env:MAX_PARALLEL (expected a positive integer)."
    }
    return [int]$env:MAX_PARALLEL
  }

  try {
    return [Environment]::ProcessorCount
  } catch {
    return 4
  }
}

if ($StartId -in @('-h', '--help')) {
  Write-Host 'Usage:'
  Write-Host '  scripts/run_all_tender_ids.bat [START_ID] [END_ID] [extra CLI args...]'
  Write-Host ''
  Write-Host 'Examples:'
  Write-Host '  scripts/run_all_tender_ids.bat'
  Write-Host '  scripts/run_all_tender_ids.bat 000001 000500'
  Write-Host '  scripts/run_all_tender_ids.bat 000001 000100 --skip-award'
  exit 0
}

if (-not (Test-TenderId $StartId)) {
  throw "Invalid START_ID: $StartId (expected exactly 6 digits)."
}

if (-not (Test-TenderId $EndId)) {
  throw "Invalid END_ID: $EndId (expected exactly 6 digits)."
}

$startNum = [int]$StartId
$endNum = [int]$EndId
if ($startNum -gt $endNum) {
  throw 'START_ID must be <= END_ID.'
}

$workDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outputRoot = if ($env:OUTPUT_ROOT) { $env:OUTPUT_ROOT } else { Join-Path $workDir 'downloads\batch_runs' }
$logDir = if ($env:LOG_DIR) { $env:LOG_DIR } else { Join-Path $outputRoot 'logs' }
$rateLimitSeconds = if ($env:RATE_LIMIT_SECONDS) { $env:RATE_LIMIT_SECONDS } else { '2.0' }
$maxParallel = Get-MaxParallel
$extraArgs = $args

New-Item -ItemType Directory -Force -Path $outputRoot, $logDir | Out-Null

$summaryFile = Join-Path $logDir 'summary.txt'
$successFile = Join-Path $logDir 'success_ids.txt'
$failedFile = Join-Path $logDir 'failed_or_missing_ids.txt'
Set-Content -Path $summaryFile -Value ''
Set-Content -Path $successFile -Value ''
Set-Content -Path $failedFile -Value ''

$queue = [System.Collections.Generic.Queue[string]]::new()
for ($n = $startNum; $n -le $endNum; $n++) {
  $queue.Enqueue($n.ToString('D6'))
}

$jobs = @()
$total = 0
$success = 0
$failed = 0

function Start-TenderJob {
  param(
    [string]$TenderId,
    [string]$WorkDir,
    [string]$OutputRoot,
    [string]$RateLimitSeconds,
    [string[]]$ExtraArgs
  )

  $scriptBlock = {
    param($tid, $wd, $od, $rl, $ea)

    $runDir = Join-Path $od $tid
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $runLog = Join-Path $runDir 'run.log'
    $cli = Join-Path $wd 'scripts\telangana_tender_cli.py'

    $cmdArgs = @(
      'run', $cli,
      '--tender-id', $tid,
      '--write-json',
      '--download',
      '--unzip',
      '--output-dir', $runDir,
      '--rate-limit-seconds', $rl
    ) + $ea

    $output = & uv @cmdArgs 2>&1
    $output | Out-File -FilePath $runLog -Encoding utf8

    [pscustomobject]@{
      TenderId = $tid
      Success  = ($LASTEXITCODE -eq 0)
      RunLog   = $runLog
    }
  }

  Start-Job -ScriptBlock $scriptBlock -ArgumentList $TenderId, $WorkDir, $OutputRoot, $RateLimitSeconds, $ExtraArgs
}

Write-Host ("Starting batch run from {0} to {1}" -f $StartId, $EndId)
Add-Content -Path $summaryFile -Value ("Starting batch run from {0} to {1}" -f $StartId, $EndId)
Write-Host ("Output root: {0}" -f $outputRoot)
Add-Content -Path $summaryFile -Value ("Output root: {0}" -f $outputRoot)
Write-Host ("Rate limit: {0}s" -f $rateLimitSeconds)
Add-Content -Path $summaryFile -Value ("Rate limit: {0}s" -f $rateLimitSeconds)
Write-Host ("Parallel workers: {0}" -f $maxParallel)
Add-Content -Path $summaryFile -Value ("Parallel workers: {0}" -f $maxParallel)

while ($queue.Count -gt 0 -or $jobs.Count -gt 0) {
  while ($queue.Count -gt 0 -and $jobs.Count -lt $maxParallel) {
    $tenderId = $queue.Dequeue()
    $total++
    Write-Host ("[{0}] Queued tender {1}" -f $total, $tenderId)
    $jobs += Start-TenderJob -TenderId $tenderId -WorkDir $workDir -OutputRoot $outputRoot -RateLimitSeconds $rateLimitSeconds -ExtraArgs $extraArgs
  }

  if ($jobs.Count -gt 0) {
    $finished = Wait-Job -Job $jobs -Any
    $result = Receive-Job -Job $finished
    Remove-Job -Job $finished | Out-Null
    $jobs = @($jobs | Where-Object { $_.Id -ne $finished.Id })

    foreach ($item in @($result)) {
      if ($item.Success) {
        $success++
        Add-Content -Path $successFile -Value $item.TenderId
      } else {
        $failed++
        Add-Content -Path $failedFile -Value $item.TenderId
        Write-Host ("  -> skipped (missing tender or request failure). See {0}" -f $item.RunLog)
      }
    }
  }
}

Get-Job -ErrorAction SilentlyContinue | Remove-Job -Force -ErrorAction SilentlyContinue | Out-Null

Write-Host ''
Write-Host 'Completed batch run'
Write-Host ("Total processed: {0}" -f $total)
Write-Host ("Success: {0}" -f $success)
Write-Host ("Failed or missing: {0}" -f $failed)
Write-Host ("Success IDs file: {0}" -f $successFile)
Write-Host ("Failed/missing IDs file: {0}" -f $failedFile)
