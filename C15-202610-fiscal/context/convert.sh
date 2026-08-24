#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Uso: $0 <archivo.docx> <salida.md>"
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

pandoc "$INPUT" \
    -t gfm \
    --wrap=none \
    --extract-media="." \
    -o "$OUTPUT"

echo "✓ Convertido: $INPUT → $OUTPUT"
