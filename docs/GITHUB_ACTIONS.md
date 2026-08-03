# GitHub Actions Continuous Integration (CI) Documentation (`.github/workflows/ci.yml`)

Automated Continuous Integration pipeline verifying code quality, unit test execution, React production compilation, and Docker image builds on every commit and pull request.

---

## 1. CI Workflow Architecture

```
                  ┌─────────────────────────────────────┐
                  │    Git Push / Pull Request Event    │
                  └──────────────────┬──────────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                         │                         │
           ▼                         ▼                         ▼
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│   Job 1: Python    │    │  Job 2: Frontend   │    │   Job 3: Docker    │
│    pytest Suite    │    │  Vite React Build  │    │ Build Validation   │
└──────────┬─────────┘    └──────────┬─────────┘    └──────────┬─────────┘
           │                         │                         │
           ▼                         ▼                         ▼
  Python Coverage XML      Frontend Dist Artifact    Verified Docker Images
```

---

## 2. Pipeline Stage Specifications

### Job 1: `python-test` (Python Test Suite & MARL Validation)
- **Environment**: `ubuntu-latest`, Python `3.11`
- **Dependency Cache**: `pip` cache key tied to `backend/requirements.txt`
- **Execution**:
  - Installs requirements and test dependencies (`pytest`, `pytest-cov`).
  - Executes unit and integration test suite across `backend/app`, `marl`, and `simulator`.
- **Artifact Output**: `python-coverage-report` (`coverage.xml` preserved for 7 days).

### Job 2: `frontend-build` (React Frontend Build & Type Check)
- **Environment**: `ubuntu-latest`, Node.js `20`
- **Dependency Cache**: `npm` cache key tied to `frontend/package.json`
- **Execution**:
  - Installs Node packages (`npm install`).
  - Compiles TypeScript and Vite production bundle (`npm run build`).
- **Artifact Output**: `frontend-dist-build` (`frontend/dist` compiled static bundle preserved for 7 days).

### Job 3: `docker-verify` (Docker Image & Compose Verification)
- **Environment**: `ubuntu-latest`, Docker Buildx `v3`
- **Execution**:
  - Compiles `backend/Dockerfile` image (`warehouse-backend:ci`).
  - Compiles `frontend/Dockerfile` image (`warehouse-frontend:ci`).
  - Compiles `nginx/Dockerfile` image (`warehouse-proxy:ci`).
  - Validates `docker compose config` syntax.

---

## 3. Zero-Secret Architecture Guarantee

> [!IMPORTANT]
> This pipeline is strictly **Continuous Integration (CI)**.
> - **No Cloud Deployment**: Does not interact with Google Cloud, AWS, or Azure.
> - **No Registry Pushes**: Images are verified locally in the runner and discarded.
> - **Zero Required Secrets**: Executes cleanly on any public or private fork without needing GitHub Secrets or environment tokens.

---

## 4. Rerunning Failed Pipeline Jobs

If a job fails due to network transient timeouts or temporary issues:
1. Navigate to the **Actions** tab in the GitHub repository.
2. Select the failed workflow run.
3. Click **Re-run jobs** $\rightarrow$ **Re-run failed jobs**.
