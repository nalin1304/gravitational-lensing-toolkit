# 🐳 Docker Quick Setup Script
# Run this to set up Docker configuration step-by-step

Write-Host "`n" -NoNewline
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "     🐳 DOCKER SETUP - Gravitational Lensing Toolkit" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Function to prompt for input
function Get-UserInput {
    param(
        [string]$Prompt,
        [string]$Default = ""
    )
    if ($Default) {
        $input = Read-Host "$Prompt [$Default]"
        if ([string]::IsNullOrWhiteSpace($input)) {
            return $Default
        }
    } else {
        $input = Read-Host $Prompt
    }
    return $input
}

# Check if Docker is installed
Write-Host "📋 Step 1: Checking Docker installation..." -ForegroundColor Yellow
Write-Host ""

try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Docker installed: $dockerVersion" -ForegroundColor Green
    } else {
        throw "Docker not found"
    }
} catch {
    Write-Host "  ❌ Docker not found!" -ForegroundColor Red
    Write-Host "  Please install Docker Desktop from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

try {
    $composeVersion = docker-compose --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Docker Compose installed: $composeVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "  ⚠️  Docker Compose not found (may be integrated in Docker)" -ForegroundColor Yellow
}

Write-Host ""

# Check if Docker is running
Write-Host "📋 Step 2: Checking Docker daemon..." -ForegroundColor Yellow
Write-Host ""

try {
    docker ps >$null 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Docker daemon is running" -ForegroundColor Green
    } else {
        throw "Docker not running"
    }
} catch {
    Write-Host "  ❌ Docker daemon is not running!" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host ""

# Docker Hub credentials
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "📋 Step 3: Docker Hub Configuration" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You need a Docker Hub account to push images." -ForegroundColor White
Write-Host "Sign up at: https://hub.docker.com/signup" -ForegroundColor Cyan
Write-Host ""

$setupDockerHub = Read-Host "Do you have a Docker Hub account? (y/n)"

if ($setupDockerHub -eq 'y') {
    Write-Host ""
    $dockerUsername = Get-UserInput "Enter your Docker Hub username"
    $dockerToken = Get-UserInput "Enter your Docker Hub token/password" "skip"
    
    if ($dockerToken -ne "skip") {
        Write-Host ""
        Write-Host "  Testing Docker Hub login..." -ForegroundColor Yellow
        
        # Test login
        $dockerToken | docker login --username $dockerUsername --password-stdin 2>$null
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✅ Docker Hub login successful!" -ForegroundColor Green
            
            # Save to .env
            $envPath = ".env"
            $envContent = ""
            
            if (Test-Path $envPath) {
                $envContent = Get-Content $envPath -Raw
            }
            
            # Update or add Docker credentials
            if ($envContent -match "DOCKER_USERNAME=") {
                $envContent = $envContent -replace "DOCKER_USERNAME=.*", "DOCKER_USERNAME=$dockerUsername"
            } else {
                $envContent += "`nDOCKER_USERNAME=$dockerUsername"
            }
            
            if ($envContent -match "DOCKER_PASSWORD=") {
                $envContent = $envContent -replace "DOCKER_PASSWORD=.*", "DOCKER_PASSWORD=$dockerToken"
            } else {
                $envContent += "`nDOCKER_PASSWORD=$dockerToken"
            }
            
            $envContent | Out-File -FilePath $envPath -Encoding UTF8
            Write-Host "  ✅ Credentials saved to .env" -ForegroundColor Green
        } else {
            Write-Host "  ❌ Login failed. Check your credentials." -ForegroundColor Red
        }
    }
} else {
    Write-Host ""
    Write-Host "  ⚠️  Skipping Docker Hub setup" -ForegroundColor Yellow
    Write-Host "  Note: You'll need this for CI/CD and image distribution" -ForegroundColor Yellow
}

Write-Host ""

# Database password
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "📋 Step 4: Database Password" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Set a secure password for PostgreSQL database" -ForegroundColor White
Write-Host ""

$dbPassword = Get-UserInput "Enter database password" "lensing_dev_password_123"

# Save to .env
$envPath = ".env"
$envContent = ""

if (Test-Path $envPath) {
    $envContent = Get-Content $envPath -Raw
}

if ($envContent -match "DB_PASSWORD=") {
    $envContent = $envContent -replace "DB_PASSWORD=.*", "DB_PASSWORD=$dbPassword"
} else {
    $envContent += "`nDB_PASSWORD=$dbPassword"
}

if ($envContent -match "DATABASE_URL=") {
    $envContent = $envContent -replace "DATABASE_URL=.*", "DATABASE_URL=postgresql://lensing:$dbPassword@localhost:5432/lensing_db"
} else {
    $envContent += "`nDATABASE_URL=postgresql://lensing:$dbPassword@localhost:5432/lensing_db"
}

$envContent | Out-File -FilePath $envPath -Encoding UTF8
Write-Host "  ✅ Database password saved to .env" -ForegroundColor Green

Write-Host ""

# Build images
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "📋 Step 5: Build Docker Images" -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

$buildImages = Read-Host "Do you want to build Docker images now? (y/n)"

if ($buildImages -eq 'y') {
    Write-Host ""
    Write-Host "  Building images (this may take 5-10 minutes)..." -ForegroundColor Yellow
    Write-Host ""
    
    docker-compose build
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "  ✅ Images built successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "  ❌ Build failed. Check the output above for errors." -ForegroundColor Red
    }
}

Write-Host ""

# Summary
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "✅ SETUP COMPLETE" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host ""
Write-Host "  1. Start all services:" -ForegroundColor Cyan
Write-Host "     docker-compose up -d" -ForegroundColor White
Write-Host ""
Write-Host "  2. View logs:" -ForegroundColor Cyan
Write-Host "     docker-compose logs -f" -ForegroundColor White
Write-Host ""
Write-Host "  3. Access applications:" -ForegroundColor Cyan
Write-Host "     - Streamlit:  http://localhost:8501" -ForegroundColor White
Write-Host "     - API:        http://localhost:8000" -ForegroundColor White
Write-Host "     - API Docs:   http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "  4. Stop services:" -ForegroundColor Cyan
Write-Host "     docker-compose down" -ForegroundColor White
Write-Host ""
Write-Host "For more info, see: DOCKER_SETUP.md" -ForegroundColor Yellow
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
