#!/bin/bash

echo "Removing all .DS_Store files"

# Delete all .DS_Store files
# Delete all .DS_Store files
find . -type f -name '*.DS_Store' -delete
find . -type f -name '*._.DS_Store' -delete
