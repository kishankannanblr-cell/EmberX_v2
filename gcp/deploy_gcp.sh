#!/usr/bin/env bash
# Automated Google Cloud Deployment Script for Project GuardianX (Bash / Cloud Shell)
set -e

PROJECT_ID=${1:-"your-gcp-project-id"}
REGION=${2:-"us-central1"}
REPO_NAME="guardianx-repo"

echo "======================================================"
echo "🚀 GuardianX: Google Cloud Automated Deployment Pipeline"
echo "======================================================"

# 1. Set Project
echo "📌 Setting active GCP Project: $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"

# 2. Enable Required GCP APIs
echo "🔌 Enabling GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    firestore.googleapis.com \
    storage.googleapis.com

# 3. Create Artifact Registry if needed
echo "📦 Ensuring Artifact Registry repository '$REPO_NAME' exists in $REGION..."
gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --description="GuardianX Docker Repository" || true

# 4. Build and Push Dashboard Container
DASHBOARD_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/guardianx-dashboard:latest"
echo "🔨 Building Dashboard Container: $DASHBOARD_IMAGE..."
gcloud builds submit --tag "$DASHBOARD_IMAGE" -f gcp/cloud_run/Dockerfile .

# 5. Deploy Dashboard to Cloud Run (Scale to Zero)
echo "🌐 Deploying GuardianX Tactical Dashboard to Cloud Run..."
gcloud run deploy guardianx-dashboard \
    --image="$DASHBOARD_IMAGE" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=5 \
    --port=8080

# 6. Build and Push Vertex AI Prediction Container
VERTEX_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/emberx-vertex-predictor:latest"
echo "🤖 Building Vertex AI Prediction Container: $VERTEX_IMAGE..."
gcloud builds submit --tag "$VERTEX_IMAGE" -f gcp/vertex_serving/Dockerfile .

# 7. Upload Model to Vertex AI Model Registry
echo "📋 Uploading Model to Vertex AI Model Registry..."
gcloud ai models upload \
    --region="$REGION" \
    --display-name="emberx-v2-sahi" \
    --container-image-uri="$VERTEX_IMAGE" \
    --container-health-route="/health" \
    --container-predict-route="/predict" \
    --container-ports=8080

echo ""
echo "✅ Deployment pipeline execution complete!"
echo "Dashboard live URL:"
gcloud run services describe guardianx-dashboard --region="$REGION" --format='value(status.url)'
