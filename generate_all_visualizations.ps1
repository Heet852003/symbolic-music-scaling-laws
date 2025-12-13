# Generate all visualizations for the project report
# PowerShell script for Windows (alternative to the bash script)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Generating All Visualizations" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Configuration
$DATA_DIR = "data/tokenized/splits"
$VOCAB_PATH = "data/tokenized/vocab.json"
$TRANSFORMER_DIR = "outputs/models/transformer"
$RNN_DIR = "outputs/models/rnn"
$OUTPUT_DIR = "outputs/plots"

# Create output directory
New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null
Write-Host "`nOutput directory: $OUTPUT_DIR"

# Check for required directories/files
Write-Host "`nChecking for required files/directories..."
$requiredPaths = @{
    "Data directory" = $DATA_DIR
    "Vocabulary file" = $VOCAB_PATH
    "Transformer models" = $TRANSFORMER_DIR
    "RNN models" = $RNN_DIR
}

$missingPaths = @()
foreach ($name in $requiredPaths.Keys) {
    $path = $requiredPaths[$name]
    if (Test-Path $path) {
        Write-Host "  [OK] $name found: $path" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] $name not found: $path" -ForegroundColor Yellow
        $missingPaths += $name
    }
}

if ($missingPaths.Count -gt 0) {
    Write-Host "`n  Note: Some paths are missing. The script will continue," -ForegroundColor Yellow
    Write-Host "        but plots requiring these may fail." -ForegroundColor Yellow
    Write-Host "  Missing: $($missingPaths -join ', ')" -ForegroundColor Yellow
}

$successCount = 0
$totalCount = 4

# 1. Data Statistics
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "1. Generating data statistics visualizations..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
try {
    python src/evaluation/plot_data_statistics.py `
        --data_dir $DATA_DIR `
        --vocab_path $VOCAB_PATH `
        --output_dir $OUTPUT_DIR
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[OK] Successfully completed: Data statistics" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "`n[ERROR] Failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "`n[ERROR] Failed to run data statistics script" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# 2. Scaling Laws
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "2. Generating scaling laws plots..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
try {
    python src/evaluation/scaling_analysis.py `
        --transformer_dir $TRANSFORMER_DIR `
        --rnn_dir $RNN_DIR `
        --output_dir $OUTPUT_DIR
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[OK] Successfully completed: Scaling laws" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "`n[ERROR] Failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "`n[ERROR] Failed to run scaling analysis script" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# 3. Training Curves
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "3. Generating training curves..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
try {
    python src/evaluation/plot_training_curves.py `
        --transformer_dir $TRANSFORMER_DIR `
        --rnn_dir $RNN_DIR `
        --output_dir $OUTPUT_DIR
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[OK] Successfully completed: Training curves" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "`n[ERROR] Failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "`n[ERROR] Failed to run training curves script" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# 4. Computational Efficiency
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "4. Generating computational efficiency plots..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
try {
    python src/evaluation/plot_computational_efficiency.py `
        --transformer_dir $TRANSFORMER_DIR `
        --rnn_dir $RNN_DIR `
        --output_dir $OUTPUT_DIR
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[OK] Successfully completed: Computational efficiency" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "`n[ERROR] Failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "`n[ERROR] Failed to run computational efficiency script" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}

# Summary
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "Visualization Generation Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Successfully completed: $successCount/$totalCount"
Write-Host "Results saved to: $OUTPUT_DIR"

# List generated files
if (Test-Path $OUTPUT_DIR) {
    $pngFiles = Get-ChildItem -Path $OUTPUT_DIR -Filter "*.png" -ErrorAction SilentlyContinue
    $csvFiles = Get-ChildItem -Path $OUTPUT_DIR -Filter "*.csv" -ErrorAction SilentlyContinue
    $jsonFiles = Get-ChildItem -Path $OUTPUT_DIR -Filter "*.json" -ErrorAction SilentlyContinue
    
    if ($pngFiles) {
        Write-Host "`nGenerated PNG files ($($pngFiles.Count)):"
        foreach ($f in $pngFiles) {
            $sizeKB = [math]::Round($f.Length / 1KB, 1)
            Write-Host "  - $($f.Name) ($sizeKB KB)"
        }
    }
    
    if ($csvFiles) {
        Write-Host "`nGenerated CSV files ($($csvFiles.Count)):"
        foreach ($f in $csvFiles) {
            $sizeKB = [math]::Round($f.Length / 1KB, 1)
            Write-Host "  - $($f.Name) ($sizeKB KB)"
        }
    }
    
    if ($jsonFiles) {
        Write-Host "`nGenerated JSON files ($($jsonFiles.Count)):"
        foreach ($f in $jsonFiles) {
            $sizeKB = [math]::Round($f.Length / 1KB, 1)
            Write-Host "  - $($f.Name) ($sizeKB KB)"
        }
    }
    
    if (-not ($pngFiles -or $csvFiles -or $jsonFiles)) {
        Write-Host "`nNo output files found in the output directory."
    }
}

if ($successCount -eq $totalCount) {
    Write-Host "`n[SUCCESS] All visualizations generated successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n[WARNING] Some visualizations failed ($($totalCount - $successCount) failed)" -ForegroundColor Yellow
    exit 1
}
