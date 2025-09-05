#!/bin/bash

# Type checking script
echo "📝 Running mypy type checking..."
mypy backend/

if [ $? -eq 0 ]; then
    echo "✅ Type checking passed!"
else
    echo "❌ Type checking failed. Please fix type issues."
    exit 1
fi