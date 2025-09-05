#!/bin/bash

# Code formatting script
echo "🎨 Formatting code with black..."
black backend/

echo "📂 Sorting imports with isort..."
isort backend/

echo "✅ Code formatting complete!"