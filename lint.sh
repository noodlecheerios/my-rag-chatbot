#!/bin/bash

# Code linting script
echo "🔍 Running flake8 linting..."
flake8 backend/

if [ $? -eq 0 ]; then
    echo "✅ No linting issues found!"
else
    echo "❌ Linting issues found. Please fix before committing."
    exit 1
fi