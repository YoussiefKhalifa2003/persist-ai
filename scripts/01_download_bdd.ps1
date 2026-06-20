#!/usr/bin/env pwsh
# BDD100K download helper — register at https://bdd-data.berkeley.edu/
param(
    [switch]$Verify
)

$LabelsDir = "data/raw/bdd100k/labels/box_track_20"
$VideosDir = "data/raw/bdd100k/videos/val"

if ($Verify) {
    $labelCount = (Get-ChildItem -Path $LabelsDir -Filter *.json -Recurse -ErrorAction SilentlyContinue).Count
    $videoCount = (Get-ChildItem -Path $VideosDir -Filter *.mp4 -Recurse -ErrorAction SilentlyContinue).Count
    Write-Host "Labels: $labelCount | Videos: $videoCount"
    if ($labelCount -ge 10 -and $videoCount -ge 10) {
        Write-Host "OK: minimal BDD subset present"
        exit 0
    }
    Write-Host "WARN: download labels/box_track_20 and val videos from BDD100K portal"
    exit 1
}

Write-Host @"
BDD100K manual download steps:
1. Register at https://bdd-data.berkeley.edu/portal.html
2. Download labels/box_track_20 to $LabelsDir
3. Download 100k val videos to $VideosDir
4. Re-run: .\scripts\01_download_bdd.ps1 -Verify
"@
New-Item -ItemType Directory -Force -Path $LabelsDir, $VideosDir | Out-Null
