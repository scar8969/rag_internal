#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d_%H%M%S)

echo "Creating backup..."

mkdir -p "$BACKUP_DIR"

if [ -f "./data/app.db" ]; then
    cp "./data/app.db" "$BACKUP_DIR/app_$DATE.db"
    echo "  Database backed up: app_$DATE.db"
fi

if [ -d "./data/chroma" ]; then
    cp -r "./data/chroma" "$BACKUP_DIR/chroma_$DATE"
    echo "  ChromaDB backed up: chroma_$DATE"
fi

echo "Backup complete: $DATE"

KEEP=30
if [ -d "$BACKUP_DIR" ]; then
    cd "$BACKUP_DIR"
    BACKUP_COUNT=$(ls -1 *.db 2>/dev/null | wc -l)
    if [ "$BACKUP_COUNT" -gt "$KEEP" ]; then
        echo "Cleaning old backups (keeping last $KEEP)..."
        ls -t *.db 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm
        ls -dt chroma_* 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -rf
    fi
fi

echo "Done!"
