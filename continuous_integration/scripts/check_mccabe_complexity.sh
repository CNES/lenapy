#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: You must specify a McCabe threshold and a directory to analyze."
    echo "Usage: $0 <threshold> <directory> [file1.py file2.py ...]"
    exit 1
fi

threshold=$1
directory=$2
shift 2

declare -A ignore_map
for ignore_file in "$@"; do
    ignore_map["$ignore_file"]=1
done

if [ ! -d "$directory" ]; then
    echo "Error: The directory '$directory' does not exist."
    exit 1
fi

all_files_ok=true
while IFS= read -r file; do
    rel_path="${file#$directory/}"  # chemin relatif

    # Vérifie si le fichier est dans la liste à ignorer
    if [[ ${ignore_map["$rel_path"]} ]] || [[ ${ignore_map["$file"]} ]]; then
        echo "Skipping $file (ignored)"
        continue
    fi

    echo "Analyzing $file ..."
    output=$(python -m mccabe --min "$threshold" "$file")

    if [ -n "$output" ]; then
        echo "Error: McCabe complexity too high in $file"
        echo "$output"
        all_files_ok=false
    fi
done < <(find "$directory" -name "*.py")

if $all_files_ok; then
    echo "✅ All files have McCabe scores less than or equal to $threshold. ✅"
else
    echo "❌ Some files have a complexity higher than $threshold ❌"
    exit 1
fi

exit 0
