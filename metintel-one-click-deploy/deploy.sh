#!/bin/bash

# MetIntel.ai One-Click Deployment Script
echo "🚀 MetIntel.ai Enhanced AI Agent Deployment"
echo "==========================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Login to Railway
echo "🔐 Please login to Railway..."
railway login

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

# Get the deployment URL
echo "🌐 Getting deployment URL..."
RAILWAY_URL=$(railway status --json | jq -r '.deployments[0].url')

echo ""
echo "✅ Deployment Complete!"
echo "======================="
echo ""
echo "🌐 Your Railway URL: $RAILWAY_URL"
echo "📋 Next steps:"
echo "   1. Add environment variables in Railway dashboard"
echo "   2. Configure api.metintel.ai in cPanel"
echo "   3. Test your deployment"
echo ""
echo "🔗 Railway Dashboard: https://railway.app/dashboard"
echo "📖 Full instructions: See README.md"
echo ""

