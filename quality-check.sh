#!/bin/bash

# Complete quality check script
echo "🚀 Running complete code quality checks..."
echo ""

# Format code
echo "🎨 Formatting code..."
black backend/ --check --diff
if [ $? -ne 0 ]; then
    echo "❌ Code formatting issues found. Run './format.sh' to fix."
    exit 1
fi

# Check import sorting
echo "📂 Checking import sorting..."
isort backend/ --check-only --diff
if [ $? -ne 0 ]; then
    echo "❌ Import sorting issues found. Run './format.sh' to fix."
    exit 1
fi

# Run linting
echo "🔍 Running linting..."
flake8 backend/
if [ $? -ne 0 ]; then
    echo "❌ Linting issues found. Please fix before committing."
    exit 1
fi

# Run type checking
echo "📝 Running type checking..."
mypy backend/
if [ $? -ne 0 ]; then
    echo "❌ Type checking failed. Please fix type issues."
    exit 1
fi

echo ""
echo "✅ All quality checks passed!"