#!/bin/bash

# Deployment script for Kent AI Agent to Fly.io

set -e

echo "🚀 Deploying Kent AI Agent to Fly.io..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first: https://fly.io/docs/getting-started/installing-flyctl/"
    exit 1
fi

# Check if logged in to Fly.io
if ! flyctl auth whoami &> /dev/null; then
    echo "❌ Not logged in to Fly.io. Please run: flyctl auth login"
    exit 1
fi

# Build frontend
echo "📦 Building frontend..."
cd frontend
npm install
npm run build
cd ..

# Check if app exists, if not create it
if ! flyctl apps list | grep -q "kent-ai-agent"; then
    echo "🆕 Creating new Fly.io app..."
    flyctl apps create kent-ai-agent --region iad
fi

# Create volume if it doesn't exist
if ! flyctl volumes list | grep -q "kent_data"; then
    echo "💾 Creating persistent volume for data..."
    flyctl volumes create kent_data --region iad --size 1
fi

# Set environment variables
echo "🔧 Setting environment variables..."
if [ -f ".env" ]; then
    echo "📄 Found .env file, setting variables from it..."
    while IFS='=' read -r key value; do
        if [[ ! $key =~ ^#.* ]] && [[ $key ]]; then
            flyctl secrets set "$key=$value"
        fi
    done < .env
else
    echo "⚠️  No .env file found. Make sure to set GOOGLE_API_KEY:"
    echo "   flyctl secrets set GOOGLE_API_KEY=your_api_key_here"
fi

# Deploy
echo "🛫 Deploying to Fly.io..."
flyctl deploy

echo "✅ Deployment complete!"
echo "🌐 Your app should be available at: https://kent-ai-agent.fly.dev"
echo "📊 Check status: flyctl status"
echo "📝 View logs: flyctl logs"