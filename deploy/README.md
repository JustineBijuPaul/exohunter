# ExoHunter Deployment Guide

This directory contains deployment configurations and guides for deploying the ExoHunter application to various cloud platforms.

## Quick Start with Docker Compose

### Prerequisites
- Docker (v20.10+)
- Docker Compose (v2.0+)
- Git

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd exohunter
   ```

2. **Start all services:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Database: localhost:5432

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop services:**
   ```bash
   docker-compose down
   ```

### Production Setup

For production deployment with Nginx reverse proxy:
```bash
docker-compose --profile production up -d
```

## Cloud Platform Deployments

### 🚀 Heroku Deployment

#### Prerequisites
- Heroku CLI installed
- Heroku account

#### Steps

1. **Create Heroku apps:**
   ```bash
   # Create API app
   heroku create exohunter-api
   
   # Create frontend app (optional, can use static hosting)
   heroku create exohunter-frontend
   ```

2. **Add PostgreSQL addon:**
   ```bash
   heroku addons:create heroku-postgresql:mini -a exohunter-api
   ```

3. **Set environment variables:**
   ```bash
   heroku config:set DATABASE_AVAILABLE=true -a exohunter-api
   heroku config:set PYTHONPATH=/app -a exohunter-api
   ```

4. **Create `heroku.yml` in project root:**
   ```yaml
   build:
     docker:
       api: Dockerfile
   run:
     api: uvicorn web.api.main:app --host 0.0.0.0 --port $PORT
   ```

5. **Deploy:**
   ```bash
   # Set stack to container
   heroku stack:set container -a exohunter-api
   
   # Deploy
   git push heroku main
   ```

6. **Scale and monitor:**
   ```bash
   heroku ps:scale api=1 -a exohunter-api
   heroku logs --tail -a exohunter-api
   ```

#### Heroku Configuration Files

Create these files in your project root:

**heroku.yml:**
```yaml
build:
  docker:
    api: Dockerfile
run:
  api: uvicorn web.api.main:app --host 0.0.0.0 --port $PORT
release:
  api: python -c "print('Database migrations would run here')"
```

**Procfile (alternative to heroku.yml):**
```
web: uvicorn web.api.main:app --host 0.0.0.0 --port $PORT
```

### ☁️ Google Cloud Platform (GCP) Deployment

#### Option 1: Cloud Run (Recommended)

1. **Setup and authentication:**
   ```bash
   # Install gcloud CLI
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Enable required APIs:**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable sql-component.googleapis.com
   ```

3. **Create Cloud SQL instance:**
   ```bash
   gcloud sql instances create exohunter-db \
     --database-version=POSTGRES_14 \
     --tier=db-f1-micro \
     --region=us-central1
   
   gcloud sql databases create exohunter --instance=exohunter-db
   gcloud sql users create exohunter --instance=exohunter-db --password=YOUR_PASSWORD
   ```

4. **Build and deploy to Cloud Run:**
   ```bash
   # Build container
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/exohunter-api
   
   # Deploy to Cloud Run
   gcloud run deploy exohunter-api \
     --image gcr.io/YOUR_PROJECT_ID/exohunter-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars DATABASE_AVAILABLE=true \
     --add-cloudsql-instances YOUR_PROJECT_ID:us-central1:exohunter-db
   ```

#### Option 2: Google Kubernetes Engine (GKE)

1. **Create GKE cluster:**
   ```bash
   gcloud container clusters create exohunter-cluster \
     --zone us-central1-a \
     --num-nodes 3 \
     --enable-autoscaling \
     --min-nodes 1 \
     --max-nodes 5
   ```

2. **Deploy using Kubernetes manifests:**
   ```bash
   kubectl apply -f deploy/k8s/
   ```

See `deploy/k8s/` directory for Kubernetes manifests.

### 🌐 AWS ECS Deployment

#### Prerequisites
- AWS CLI configured
- ECS CLI installed

#### Steps

1. **Create ECS cluster:**
   ```bash
   aws ecs create-cluster --cluster-name exohunter-cluster
   ```

2. **Create ECR repositories:**
   ```bash
   aws ecr create-repository --repository-name exohunter-api
   aws ecr create-repository --repository-name exohunter-frontend
   ```

3. **Push images to ECR:**
   ```bash
   # Get login token
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
   
   # Tag and push API image
   docker build -t exohunter-api .
   docker tag exohunter-api:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/exohunter-api:latest
   docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/exohunter-api:latest
   ```

4. **Create RDS instance:**
   ```bash
   aws rds create-db-instance \
     --db-instance-identifier exohunter-db \
     --db-instance-class db.t3.micro \
     --engine postgres \
     --master-username exohunter \
     --master-user-password YOUR_PASSWORD \
     --allocated-storage 20
   ```

5. **Create ECS task definition:**
   ```bash
   aws ecs register-task-definition --cli-input-json file://deploy/aws/task-definition.json
   ```

6. **Create ECS service:**
   ```bash
   aws ecs create-service \
     --cluster exohunter-cluster \
     --service-name exohunter-api-service \
     --task-definition exohunter-api:1 \
     --desired-count 2
   ```

#### AWS Configuration Files

See `deploy/aws/` directory for:
- `task-definition.json` - ECS task definition
- `service.json` - ECS service configuration
- `cloudformation.yml` - Infrastructure as Code

### 🔧 Environment Variables

Create a `.env` file for local development:

```bash
# Database
DATABASE_URL=postgresql://exohunter:exohunter_password@localhost:5432/exohunter
DATABASE_AVAILABLE=true

# API Configuration
API_VERSION=1.0.0
DEBUG=false

# Frontend
REACT_APP_API_URL=http://localhost:8000

# Optional: Redis
REDIS_URL=redis://localhost:6379

# Optional: ML Model paths
MODEL_PATH=/app/models
```

### 📊 Monitoring and Logging

#### Health Checks
- API Health: `GET /health`
- Frontend Health: `GET /health`
- Database Health: Built into docker-compose

#### Logging
```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f api
docker-compose logs -f frontend
docker-compose logs -f db
```

#### Monitoring
- Use `docker stats` for resource monitoring
- Consider adding Prometheus + Grafana for production monitoring
- Application Performance Monitoring (APM) tools like DataDog, New Relic

### 🔒 Security Considerations

1. **Environment Variables:**
   - Never commit secrets to version control
   - Use cloud provider secret management services
   - Rotate passwords regularly

2. **Database Security:**
   - Use SSL connections in production
   - Implement proper backup strategies
   - Regular security updates

3. **API Security:**
   - Implement rate limiting
   - Use HTTPS in production
   - Validate all inputs

4. **Container Security:**
   - Use non-root users in containers
   - Regularly update base images
   - Scan images for vulnerabilities

### 🚨 Troubleshooting

#### Common Issues

1. **Port conflicts:**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   ```

2. **Database connection errors:**
   ```bash
   # Check database logs
   docker-compose logs db
   
   # Test connection
   docker-compose exec db psql -U exohunter -d exohunter
   ```

3. **Frontend build issues:**
   ```bash
   # Rebuild frontend
   docker-compose build frontend --no-cache
   ```

4. **API import errors:**
   ```bash
   # Check Python path and dependencies
   docker-compose exec api python -c "import sys; print(sys.path)"
   ```

### 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [React Deployment Guide](https://create-react-app.dev/docs/deployment/)
- [Heroku Container Registry](https://devcenter.heroku.com/articles/container-registry-and-runtime)
- [AWS ECS Guide](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/)
- [GCP Cloud Run Documentation](https://cloud.google.com/run/docs)

## Support

For deployment issues, please check:
1. Application logs
2. Database connectivity
3. Environment variable configuration
4. Network connectivity between services

If you need additional help, please create an issue in the repository with:
- Deployment platform
- Error messages
- Configuration details (without secrets)
