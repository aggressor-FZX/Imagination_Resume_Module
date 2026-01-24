#!/bin/bash
# Script to purge .env files from Git history using git filter-branch
# CAUTION: This rewrites history. Backup your repo before running.

set -e

# Backup current branch
BACKUP_BRANCH="backup-before-env-purge-$(date +%Y%m%d%H%M%S)"
git branch "$BACKUP_BRANCH"
echo "Created backup branch: $BACKUP_BRANCH"

# List of files to purge
FILES_TO_PURGE=".env .env.local .env.production .env.test .env.development"

echo "Purging files from history: $FILES_TO_PURGE"

# Use git filter-branch to remove files from all commits
# Note: --index-filter is faster than --tree-filter
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch $FILES_TO_PURGE" \
  --prune-empty --tag-name-filter cat -- --all

echo "Purge complete."
echo "To finalize, you must force push: git push origin --force --all"
echo "To clean up the local repo: rm -rf .git/refs/original/ && git reflog expire --expire=now --all && git gc --prune=now --aggressive"
