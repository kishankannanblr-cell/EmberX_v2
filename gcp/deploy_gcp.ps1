# gcp/deploy_gcp.ps1
# Automated Google Cloud Deployment Script for Project GuardianX (PowerShell)

param (
    [string]$ProjectId = "your-gcp-project-id",
    [string]$Region = "us-central1",
    [string]$RepoName = "guardianx-repo"
)

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "🚀 GuardianX: Google Cloud Automated Deployment Pipeline" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# 1. Set Project
Write-Host "📌 Setting active GCP Project: $ProjectId..." -ForegroundColor Yellow
gcloud config set project $ProjectId

# 2. Enable Required GCP APIs
Write-Host "🔌 Enabling GCP APIs (Cloud Run, Vertex AI, Artifact Registry, Firestore)..." -ForegroundColor Yellow
gcloud services enable `
    run.googleapis.com `
    aiplatform.googleapis.com `
    artifactregistry.googleapis.com `
    cloudbuild.googleapis.com `
    firestore.googleapis.com `
    storage.googleapis.com

# 3. Create Artifact Registry Repository if not exists
Write-Host "📦 Ensuring Artifact Registry repository '$RepoName' exists in $Region..." -ForegroundColor Yellow
gcloud artifacts repositories create $RepoName `
    --repository-format=docker `
    --location=$Region `
    --description="GuardianX Docker Repository" 2>$null

# 4. Build and Push Dashboard Container to Artifact Registry
$DASHBOARD_IMAGE = "$Region-docker.pkg.dev/$ProjectId/$RepoName/guardianx-dashboard:latest"
Write-Host "🔨 Building and pushing Dashboard container: $DASHBOARD_IMAGE..." -ForegroundColor Yellow
gcloud builds submit --tag $DASHBOARD_IMAGE -f gcp/cloud_run/Dockerfile .

# 5. Deploy Tactical Dashboard to Google Cloud Run (Scale to Zero)
Write-Host "🌐 Deploying GuardianX Tactical Dashboard to Google Cloud Run..." -ForegroundColor Yellow
gcloud run deploy guardianx-dashboard `
    --image=$DASHBOARD_IMAGE `
    --region=$Region `
    --platform=managed `
    --allow-unauthenticated `
    --memory=4Gi `
    --cpu=2 `
    --min-instances=0 `
    --max-instances=5 `
    --port=8080

# 6. Build and Push Vertex AI Prediction Container
$VERTEX_IMAGE = "$Region-docker.pkg.dev/$ProjectId/$RepoName/emberx-vertex-predictor:latest"
Write-Host "🤖 Building Vertex AI Prediction Container: $VERTEX_IMAGE..." -ForegroundColor Yellow
gcloud builds submit --tag $VERTEX_IMAGE -f gcp/vertex_serving/Dockerfile .

# 7. Upload Model to Vertex AI Model Registry
Write-Host "📋 Uploading Model to Vertex AI Model Registry..." -ForegroundColor Yellow
gcloud ai models upload `
    --region=$Region `
    --display-name="emberx-v2-sahi" `
    --container-image-uri=$VERTEX_IMAGE `
    --container-health-route="/health" `
    --container-predict-route="/predict" `
    --container-ports=8080

Write-Host "`n✅ Deployment Complete!" -ForegroundColor Green
Write-Host "Run the following to check your Cloud Run Dashboard URL:" -ForegroundColor Cyan
Write-Host "gcloud run services describe guardianx-dashboard --region=$Region --format='value(status.url)'" -ForegroundColor White
