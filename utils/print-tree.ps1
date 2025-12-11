<#
.SYNOPSIS
Affiche une arborescence propre du projet (style tree) avec emojis et indentation.
#>

param (
    [string]$Path = ".",
    [int]$Depth = 10
)

# Filtres pour ignorer le bruit
$excludePattern = '\\\.venv\\|\\__pycache__\\|\\\.pytest_cache\\|\\data\\raw\\|\\data\\interim\\'

# Fonction récursive pour afficher les éléments sous forme d’arbre
function Show-Tree($folder, $prefix = "") {
    $items = Get-ChildItem -Path $folder | Where-Object {
        $_.FullName -notmatch $excludePattern -and
        ($_.Extension -match '\.py$|\.csv$|\.txt$|\.parquet$|\.yaml$|\.md$' -or $_.PSIsContainer)
    }

    $count = $items.Count
    for ($i = 0; $i -lt $count; $i++) {
        $item = $items[$i]
        $isLast = ($i -eq $count - 1)
        $branch = if ($isLast) { "└── " } else { "├── " }
        $newPrefix = if ($isLast) { "$prefix    " } else { "$prefix│   " }

        if ($item.PSIsContainer) {
            Write-Output "$prefix$branch📁 $($item.Name)"
            if ($Depth -gt 1) {
                Show-Tree $item.FullName $newPrefix
            }
        }
        else {
            Write-Output "$prefix$branch📄 $($item.Name)"
        }
    }
}

# Lancer depuis la racine donnée
Write-Output "📂 Arborescence de $Path"
Show-Tree (Resolve-Path $Path)
