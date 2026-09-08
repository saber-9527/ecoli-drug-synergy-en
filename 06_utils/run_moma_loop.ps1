# MOMA batch runner — loops until all combos done, fresh memory each batch
$pyExe = "python"
$script = "01_flux_simulation/run_moma_5154.py"
$outDir = "../data/continuous_bounded_moma_all"
$geneDir = "../data/gene_combo-2"

while ($true) {
    $done = (Get-ChildItem $outDir -Filter "*_FLUX.csv" -ErrorAction SilentlyContinue).Count
    $total = (Get-ChildItem $geneDir -Filter "*.csv" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "__" }).Count
    Write-Host "========== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =========="
    Write-Host "Done: $done / $total"

    if ($done -ge $total) {
        Write-Host "ALL DONE!"
        break
    }

    & $pyExe $script
    Start-Sleep -Seconds 3
}
