#!/usr/bin/env pwsh
# Download MOT17 train set (~1.5GB) from MOTChallenge
$OutDir = "data/raw/mot17/train"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$ZipPath = "data/raw/mot17/MOT17.zip"
if (-not (Test-Path $ZipPath)) {
    Write-Host "Downloading MOT17..."
    $url = "https://motchallenge.net/data/1/2/MOT17.zip"
    try {
        Invoke-WebRequest -Uri $url -OutFile $ZipPath -UseBasicParsing
    } catch {
        Write-Host "Direct download failed. Manual: https://motchallenge.net/data/MOT17/"
        Write-Host "Place MOT17-02..13 folders under $OutDir"
        exit 1
    }
}

if (Test-Path $ZipPath) {
    Expand-Archive -Path $ZipPath -DestinationPath "data/raw/mot17" -Force
    Write-Host "Extracted MOT17 to data/raw/mot17"
}

$seqs = Get-ChildItem -Path $OutDir -Directory -Filter "MOT17-*" -ErrorAction SilentlyContinue
Write-Host "Sequences found: $($seqs.Count)"
