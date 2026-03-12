# Implementation Plan: API Interface for Alpha Gen

## Overview

Add a FastAPI-based REST API interface to Alpha Gen, exposing all CLI commands (research, gather, opportunities, news, analyze) as API endpoints with OIDC client authentication (pass-through JWT validation), optional background task execution for long-running operations, and enhanced OpenAPI documentation.

## Global Context

### Current Architecture
- **Clean separation**: `src/alpha_gen/core/` contains all business logic (interface-agnostic)
- **CLI interface**: `src/alpha_gen/cli/` wraps core logic with Typer commands
- **Async-first**: All agents and data sources are async
- **Standardized responses**: All agents return `{"status": "success"|"error", "error": str|None, ...}`
- **Configuration**: Pydantic models loaded via `get_config()` singleton from environment

### Key Agents to Expose
1. **ResearchAgent** - Deep company analysis (10-60s)
2. **GatherAgent** - Data collection & storage (5-30s per ticker)
3. **OpportunitiesAgent** - Market opportunity analysis (15-45s)
4. **NewsAgent** - News sentiment analysis (10-30s)
5. **Analyze** - Quick analysis combining research + news (10-40s)

### Existing Patterns to Follow
- Pydantic models for all data structures
- Structlog for structured logging
- Async/await throughout
- Error handling with status/error pattern
- Configuration via environment variables

## Architecture Decisions

### API Framework: FastAPI
- Native async support matches codebase
- Built-in Pydantic integration (already using Pydantic)
- Auto-generated OpenAPI documentation
- Dependency injection system for auth/config

### Authentication: OIDC Client (Pass-Through)
- External OIDC provider handles authentication (Zitadel, Auth0, Keycloak, etc.)
- API validates incoming JWTs against provider's JWKS endpoint
- No local user credentials or password handling
- Minimal user tracking: provider user ID extracted from token
- Configurable for any OIDC-compliant provider

### Background Tasks: FastAPI BackgroundTasks + In-Memory Task Store
- Simple in-memory task registry (no external dependencies)
- Task ID returned immediately for async requests
- Status endpoint for polling task progress
- Optional: upgrade to Redis/Celery later if needed

### Project Structure
```
src/alpha_gen/
├── api/                    # NEW: API interface (sibling to cli/)
│   ├── __init__.py
│   ├── app.py              # FastAPI app factory
│   ├── dependencies.py     # Auth, config dependencies
│   ├── auth/               # Authentication module
│   │   ├── __init__.py
│   │   ├── oidc.py         # OIDC client, JWKS validation
│   │   └── user.py         # Minimal user tracking
│   ├── routes/             # API route modules
│   │   ├── __init__.py
│   │   ├── research.py     # Research endpoint
│   │   ├── gather.py       # Gather endpoint
│   │   ├── opportunities.py # Opportunities endpoint
│   │   ├── news.py         # News endpoint
│   │   ├── analyze.py      # Analyze endpoint
│   │   └── tasks.py        # Task status endpoint
│   ├── models/             # Pydantic request/response models
│   │   ├── __init__.py
│   │   ├── requests.py     # API request models
│   │   ├── responses.py    # API response models
│   │   └── tasks.py        # Task status models
│   └── tasks/              # Background task management
│       ├── __init__.py
│       ├── registry.py     # In-memory task registry
│       └── executor.py     # Task execution wrapper
├── core/                   # Existing: unchanged
├── cli/                    # Existing: add 'api' command
└── main.py                 # Existing: unchanged
```

## Phase Versioning Strategy

### Git Tag

All work happens on the current branch. Each completed phase is tagged as a checkpoint.

**Tag naming convention**: `phase/[phase-number]-[kebab-case-name]`
- Example: `phase/1-setup-api-structure`, `phase/2-authentication-system`

**Benefits**:
- Minimal git overhead — no branch management
- Linear commit history
- Easy to roll back to any phase checkpoint

**For phases with dependencies**: Phases are inherently sequential on the same branch; no merging needed.

## Assumptions

1. **OIDC Provider**: External provider (Zitadel, Auth0, Keycloak, etc.) handles all authentication
2. **Pass-through tokens**: Client obtains JWT from OIDC provider, sends to API
3. **Minimal user tracking**: Only store provider user ID (sub claim) for reference
4. **In-memory task store**: Simple task registry. Will lose tasks on restart (acceptable for MVP).
5. **No rate limiting in MVP**: Can be added as middleware later.
6. **No HTTPS in MVP**: Assume reverse proxy (nginx/traefik) handles TLS termination.

---

## Phases

### Phase 1: Setup API Structure & Dependencies

**Objective**: Add FastAPI dependencies and create the api/ directory structure with base configuration.

**Complexity**: low  
**Estimated Time**: 20 min

**Prerequisites**: None

**Context for this Phase**:
- Project uses `uv` for dependency management (`just sync`, `just sync-dev`)
- Dependencies are defined in `pyproject.toml` under `[project.dependencies]` and `[project.optional-dependencies]`
- The `src/alpha_gen/` directory contains `core/` and `cli/` as siblings
- Configuration is loaded via `get_config()` from `alpha_gen.core.config`
- All imports from core should use `from alpha_gen.core import ...`

**Files**:
| File | Action | Purpose |
|------|--------|---------|
| `pyproject.toml` | modify | Add FastAPI, uvicorn, python-jose, passlib, python-multipart dependencies |
| `src/alpha_gen/api/__init__.py` | create | Package init, export app factory |
| `src/alpha_gen/api/app.py` | create | FastAPI app factory with basic config |
| `src/alpha_gen/api/dependencies.py` | create | Config dependency injection |
| `src/alpha_gen/api/models/__init__.py` | create | Models package init |
| `src/alpha_gen/api/models/responses.py` | create | Base response models (HealthResponse, ErrorResponse) |
| `src/alpha_gen/api/routes/__init__.py` | create | Routes package init |
| `src/alpha_gen/api/routes/health.py` | create | Health check endpoint |

**Implementation Steps**:

1. Add dependencies to `pyproject.toml`:
   ```toml
   [project.dependencies]
   # ... existing dependencies ...
   "fastapi>=0.115.0",
   "uvicorn[standard]>=0.32.0",
   "python-jose[cryptography]>=3.3.0",  # JWT handling
   "passlib[bcrypt]>=1.7.4",            # Password hashing
   "python-multipart>=0.0.12",          # Form data for OAuth2
   ```

2. Run `just sync` to install new dependencies.

3. Create `src/alpha_gen/api/__init__.py`:
   - Export `create_app()` function from `app.py`

4. Create `src/alpha_gen/api/app.py`:
   - Define `create_app()` factory function
   - Create FastAPI instance with title, version, description
   - Include health router
   - Configure CORS middleware (allow all origins for MVP)
   - Add exception handlers for generic errors

5. Create `src/alpha_gen/api/dependencies.py`:
   - Define `get_config()` dependency that returns `AppConfig`
   - Define `get_vector_store()` dependency

6. Create `src/alpha_gen/api/models/responses.py`:
   - `HealthResponse`: status, version, config_status
   - `ErrorResponse`: detail, error_code (optional)

7. Create `src/alpha_gen/api/routes/health.py`:
   - `GET /health` - returns HealthResponse
   - Check config is loaded, vector store is accessible

**Verification**:
- [ ] `just sync` completes without errors
- [ ] `just lint` passes
- [ ] `just type-check` passes
- [ ] Can import `from alpha_gen.api import create_app`
- [ ] `create_app()` returns a FastAPI instance

**Completion Gate**:
> This phase is NOT complete until the user has reviewed the work and explicitly confirmed it is done. Do not proceed to dependent phases or mark this phase as finished without user approval.

**Outputs**:
- FastAPI app factory ready for route registration
- Base models for API responses
- Health endpoint for basic verification

**After completing this phase**:
```bash
# Commit your changes on the current branch
git add .
git commit -m "feat(api): setup FastAPI structure and dependencies"

# Tag the completion of this phase
git tag -a phase/1-setup-api-structure -m "Phase 1: Setup API Structure & Dependencies complete"

# Push commit and tag to remote (optional)
git push && git push --tags
```

---

### Phase 2: OIDC Client Authentication

**Objective**: Implement OIDC client authentication that validates JWTs from external provider (Zitadel or any OIDC-compliant provider).

**Complexity**: medium  
**Estimated Time**: 35 min

**Prerequisites**: Phase 1 complete (API structure exists)

**Context for this Phase**:
- OIDC providers expose a `.well-known/openid-configuration` discovery endpoint
- JWKS (JSON Web Key Set) endpoint provides public keys for JWT verification
- JWT validation requires: signature verification, issuer check, audience check, expiry check
- Use `python-jose` or `pyjwt` for JWT handling
- Use `httpx` for async OIDC discovery
- Configuration already has `AppConfig` pattern to follow

**Files**:
| File | Action | Purpose |
|------|--------|---------|
| `src/alpha_gen/core/config/settings.py` | modify | Add OIDCConfig with provider settings |
| `src/alpha_gen/api/auth/__init__.py` | create | Auth package init |
| `src/alpha_gen/api/auth/oidc.py` | create | OIDC client, JWKS fetching, JWT validation |
| `src/alpha_gen/api/auth/user.py` | create | Minimal user tracking (provider ID) |
| `src/alpha_gen/api/dependencies.py` | modify | Add get_current_user dependency |
| `src/alpha_gen/api/models/responses.py` | modify | Add UserResponse |
| `src/alpha_gen/api/routes/auth.py` | create | Auth info endpoint (optional) |
| `src/alpha_gen/api/app.py` | modify | Include auth router, fetch OIDC config on startup |

**Implementation Steps**:

1. Add to `src/alpha_gen/core/config/settings.py`:
   ```python
   class OIDCConfig(BaseModel):
       """OIDC provider configuration for JWT validation (pass-through auth).
       
       No client_secret needed - we only validate incoming JWTs using the provider's
       public keys (JWKS). Client secret is only required for token exchange flows
       (authorization code, client credentials, refresh), which we don't use.
       """
       issuer: str  # e.g., "https://your-zitadel.domain"
       client_id: str  # Your application's client ID (for audience validation)
       discovery_url: str | None = None  # Defaults to issuer/.well-known/openid-configuration
       audience: str | None = None  # Optional: override audience validation
       algorithms: list[str] = ["RS256"]
       jwks_cache_ttl_seconds: int = 3600  # 1 hour default for JWKS caching
       
       @property
       def effective_discovery_url(self) -> str:
           return self.discovery_url or f"{self.issuer.rstrip('/')}/.well-known/openid-configuration"
       
       model_config = {"frozen": True}
   
   # Add to AppConfig:
   oidc: OIDCConfig
   ```

2. Add `httpx` to `pyproject.toml` dependencies (for async OIDC discovery).

3. Create `src/alpha_gen/api/auth/oidc.py`:
   ```python
   # Key components:
   # - OIDCConfig: Store discovered endpoints and JWKS
   # - fetch_oidc_config(discovery_url): Get provider metadata
   # - fetch_jwks(jwks_uri): Get public keys
   # - verify_jwt(token, config, jwks): Validate signature, issuer, audience, expiry
   # - extract_user_info(token_payload): Get sub, email, name from token
   ```

4. Create `src/alpha_gen/api/auth/user.py`:
   ```python
   @dataclass(frozen=True)
   class UserInfo:
       """Minimal user info extracted from JWT."""
       sub: str  # Provider user ID (subject claim)
       email: str | None = None
       name: str | None = None
       roles: list[str] | None = None
       raw_payload: dict[str, Any] | None = None  # Full token claims
   ```

5. Update `src/alpha_gen/api/dependencies.py`:
   ```python
   # Add dependency:
   async def get_current_user(
       authorization: str = Header(..., alias="Authorization"),
       oidc_config: OIDCConfig = Depends(get_oidc_config),
       jwks: dict = Depends(get_jwks),
   ) -> UserInfo:
       """Validate JWT from Authorization header and extract user info."""
       # Extract Bearer token
       # Verify JWT signature and claims
       # Return UserInfo
   ```

6. Create `src/alpha_gen/api/routes/auth.py` (optional):
   - `GET /api/v1/auth/info` - Returns OIDC provider info (issuer, client_id)
   - `GET /api/v1/auth/me` - Returns current user info from token

7. Update `src/alpha_gen/api/app.py`:
   - On startup: fetch OIDC config and JWKS from provider
   - Cache JWKS in app state
   - Include auth router

**Verification**:
- [ ] `just sync` completes (httpx added)
- [ ] `just lint` passes
- [ ] `just type-check` passes
- [ ] OIDC discovery fetches provider config successfully
- [ ] JWKS endpoint returns public keys
- [ ] Valid JWT from provider is accepted
- [ ] Invalid/expired JWT is rejected with 401
- [ ] Token with wrong issuer is rejected
- [ ] `get_current_user` dependency extracts user info from valid token

**Completion Gate**:
> This phase is NOT complete until the user has reviewed the work and explicitly confirmed it is done. Do not proceed to dependent phases or mark this phase as finished without user approval.

**Outputs**:
- OIDC client that validates JWTs from external provider
- `get_current_user` dependency for protected routes
- Minimal user info extraction from tokens

**After completing this phase**:
```bash
# Commit your changes on the current branch
git add .
git commit -m "feat(api): add OIDC client authentication"

# Tag the completion of this phase
git tag -a phase/2-oidc-auth -m "Phase 2: OIDC Client Authentication complete"

# Push commit and tag to remote (optional)
git push && git push --tags
```

---

### Phase 3: Research & Gather API Endpoints

**Objective**: Implement Research and Gather API endpoints with synchronous execution and JWT protection.

**Complexity**: medium  
**Estimated Time**: 40 min

**Prerequisites**: Phase 2 complete (authentication working)

**Context for this Phase**:
- ResearchAgent is in `alpha_gen.core.agents.research`
- GatherAgent is in `alpha_gen.core.agents.gather`
- Both agents return `dict[str, Any]` with `status`, `error`, and result fields
- ResearchAgent.run() takes `{"ticker": str, "skip_gather": bool}`
- GatherAgent has `gather_multiple_tickers(tickers: list[str])`
- All agents are async
- Use `get_current_user` dependency for protection

**Files**:
| File | Action | Purpose |
|------|--------|---------|
| `src/alpha_gen/api/models/requests.py` | modify | Add ResearchRequest, GatherRequest |
| `src/alpha_gen/api/models/responses.py` | modify | Add ResearchResponse, GatherResponse, GatherMultipleResponse |
| `src/alpha_gen/api/routes/research.py` | create | Research endpoint |
| `src/alpha_gen/api/routes/gather.py` | create | Gather endpoint |
| `src/alpha_gen/api/app.py` | modify | Include research and gather routers |

**Implementation Steps**:

1. Add to `src/alpha_gen/api/models/requests.py`:
   ```python
   class ResearchRequest(BaseModel):
       ticker: str = Field(..., min_length=1, max_length=10, pattern=r"^[A-Z]+$")
       skip_gather: bool = False
   
   class GatherRequest(BaseModel):
       tickers: list[str] = Field(..., min_length=1, max_length=10)
       
       @field_validator("tickers")
       @classmethod
       def validate_tickers(cls, v: list[str]) -> list[str]:
           return [t.upper().strip() for t in v]
   ```

2. Add to `src/alpha_gen/api/models/responses.py`:
   ```python
   class ResearchResponse(BaseModel):
       status: Literal["success", "error"]
       ticker: str
       analysis: str | None = None
       error: str | None = None
       duration_ms: float | None = None
       latest_quarter: str | None = None
       latest_news_time: str | None = None
       context: dict[str, Any] | None = None
   
   class GatherResponse(BaseModel):
       status: Literal["success", "error"]
       ticker: str
       docs_added: int | None = None
       news_articles_stored: int | None = None
       metrics_stored: int | None = None
       indicators_stored: int | None = None
       latest_quarter: str | None = None
       duration_ms: float | None = None
       error: str | None = None
   
   class GatherMultipleResponse(BaseModel):
       status: Literal["success", "partial", "error"]
       total_tickers: int
       successful: int
       failed: int
       results: list[GatherResponse]
       errors: list[dict[str, str]] | None = None
   ```

3. Create `src/alpha_gen/api/routes/research.py`:
   - `POST /api/v1/research` - protected endpoint
   - Accept `ResearchRequest` body
   - Import and instantiate `ResearchAgent`
   - Call `agent.run({"ticker": ..., "skip_gather": ...})`
   - Map result to `ResearchResponse`
   - Add OpenAPI description, examples, error responses

4. Create `src/alpha_gen/api/routes/gather.py`:
   - `POST /api/v1/gather` - protected endpoint
   - Accept `GatherRequest` body
   - Import `gather_multiple_tickers` from gather agent
   - Call with tickers list
   - Map result to `GatherMultipleResponse`
   - Add OpenAPI description, examples, error responses

5. Update `src/alpha_gen/api/app.py`:
   - Include research router with prefix `/api/v1`
   - Include gather router with prefix `/api/v1`

**Verification**:
- [ ] `just lint` passes
- [ ] `just type-check` passes
- [ ] `POST /api/v1/research` with valid token returns analysis
- [ ] `POST /api/v1/research` without token returns 401
- [ ] `POST /api/v1/gather` with valid token stores data
- [ ] Invalid ticker returns appropriate error response
- [ ] OpenAPI docs show request/response schemas

**Completion Gate**:
> This phase is NOT complete until the user has reviewed the work and explicitly confirmed it is done. Do not proceed to dependent phases or mark this phase as finished without user approval.

**Outputs**:
- Working Research API endpoint
- Working Gather API endpoint
- Both endpoints protected by JWT authentication

**After completing this phase**:
```bash
# Commit your changes on the current branch
git add .
git commit -m "feat(api): add research and gather endpoints"

# Tag the completion of this phase
git tag -a phase/3-research-gather-endpoints -m "Phase 3: Research & Gather Endpoints complete"

# Push commit and tag to remote (optional)
git push && git push --tags
```

---

### Phase 4: Opportunities, News & Analyze Endpoints

**Objective**: Implement remaining API endpoints for opportunities, news, and analyze commands.

**Complexity**: medium  
**Estimated Time**: 35 min

**Prerequisites**: Phase 3 complete (research/gather pattern established)

**Context for this Phase**:
- OpportunitiesAgent is in `alpha_gen.core.agents.opportunities`
- NewsAgent is in `alpha_gen.core.agents.news`
- Analyze is a CLI command that combines research + news (in `alpha_gen.cli.commands.analyze`)
- All follow same pattern as Phase 3
- OpportunitiesAgent.run() takes `{"limit": int}`
- NewsAgent.run() takes `{}` (no params)
- Analyze takes ticker and optional news flag

**Files**:
| File | Action | Purpose |
|------|--------|---------|
| `src/alpha_gen/api/models/requests.py` | modify | Add OpportunitiesRequest, AnalyzeRequest |
| `src/alpha_gen/api/models/responses.py` | modify | Add OpportunitiesResponse, NewsResponse, AnalyzeResponse |
| `src/alpha_gen/api/routes/opportunities.py` | create | Opportunities endpoint |
| `src/alpha_gen/api/routes/news.py` | create | News endpoint |
| `src/alpha_gen/api/routes/analyze.py` | create | Analyze endpoint |
| `src/alpha_gen/api/app.py` | modify | Include new routers |

**Implementation Steps**:

1. Add to `src/alpha_gen/api/models/requests.py`:
   ```python
   class OpportunitiesRequest(BaseModel):
       limit: int = Field(default=25, ge=1, le=100)
   
   class AnalyzeRequest(BaseModel):
       ticker: str = Field(..., min_length=1, max_length=10, pattern=r"^[A-Z]+$")
       include_news: bool = False
   ```

2. Add to `src/alpha_gen/api/models/responses.py`:
   ```python
   class OpportunitiesResponse(BaseModel):
       status: Literal["success", "error"]
       losers_count: int | None = None
       analysis: str | None = None
       duration_ms: float | None = None
       error: str | None = None
   
   class NewsResponse(BaseModel):
       status: Literal["success", "error"]
       analysis: str | None = None
       articles_analyzed: int | None = None
       duration_ms: float | None = None
       error: str | None = None
   
   class AnalyzeResponse(BaseModel):
       status: Literal["success", "error"]
       ticker: str
       analysis: str | None = None
       news_analysis: str | None = None
       duration_ms: float | None = None
       error: str | None = None
   ```

3. Create `src/alpha_gen/api/routes/opportunities.py`:
   - `POST /api/v1/opportunities` - protected endpoint
   - Accept `OpportunitiesRequest` body
   - Import and instantiate `OpportunitiesAgent`
   - Call `agent.run({"limit": ...})`
   - Map result to `OpportunitiesResponse`
   - Add OpenAPI docs with examples

4. Create `src/alpha_gen/api/routes/news.py`:
   - `POST /api/v1/news` - protected endpoint
   - No request body needed (or empty object)
   - Import and instantiate `NewsAgent`
   - Call `agent.run({})`
   - Map result to `NewsResponse`
   - Add OpenAPI docs

5. Create `src/alpha_gen/api/routes/analyze.py`:
   - `POST /api/v1/analyze` - protected endpoint
   - Accept `AnalyzeRequest` body
   - Run ResearchAgent (abbreviated output)
   - Optionally run NewsAgent if `include_news=True`
   - Combine results into `AnalyzeResponse`
   - Add OpenAPI docs

6. Update `src/alpha_gen/api/app.py`:
   - Include opportunities, news, analyze routers with prefix `/api/v1`

**Verification**:
- [ ] `just lint` passes
- [ ] `just type-check` passes
- [ ] `POST /api/v1/opportunities` returns analysis
- [ ] `POST /api/v1/news` returns news analysis
- [ ] `POST /api/v1/analyze` returns combined analysis
- [ ] All endpoints require authentication
- [ ] OpenAPI docs complete for all endpoints

**Completion Gate**:
> This phase is NOT complete until the user has reviewed the work and explicitly confirmed it is done. Do not proceed to dependent phases or mark this phase as finished without user approval.

**Outputs**:
- All 5 CLI commands exposed as API endpoints
- Consistent request/response patterns
- Complete OpenAPI documentation

**After completing this phase**:
```bash
# Commit your changes on the current branch
git add .
git commit -m "feat(api): add opportunities, news, and analyze endpoints"

# Tag the completion of this phase
git tag -a phase/4-remaining-endpoints -m "Phase 4: Opportunities, News & Analyze Endpoints complete"

# Push commit and tag to remote (optional)
git push && git push --tags
```

---

### Phase 5: Background Task System

**Objective**: Add optional async execution for long-running operations with task status polling.

**Complexity**: medium  
**Estimated Time**: 40 min

**Prerequisites**: Phase 4 complete (all sync endpoints working)

**Context for this Phase**:
- Research and Gather can take 30-60+ seconds
- FastAPI has `BackgroundTasks` for simple async execution
- Need in-memory task registry to track task status
- Tasks should be queryable by ID
- Each endpoint should accept `async: bool` parameter

**Files**:
| File | Action | Purpose |
|------|--------|---------|
| `src/alpha_gen/api/tasks/__init__.py` | create | Tasks package init |
| `src/alpha_gen/api/tasks/registry.py` | create | In-memory task registry |
| `src/alpha_gen/api/tasks/executor.py` | create | Task execution wrapper |
| `src/alpha_gen/api/models/tasks.py` | create | Task status models |
| `src/alpha_gen/api/routes/tasks.py` | create | Task status endpoint |
| `src/alpha_gen/api/routes/research.py` | modify | Add async option |
| `src/alpha_gen/api/routes/gather.py` | modify | Add async option |
| `src/alpha_gen/api/routes/opportunities.py` | modify | Add async option |
| `src/alpha_gen/api/routes/news.py` | modify | Add async option |
| `src/alpha_gen/api/routes/analyze.py` | modify | Add async option |
| `src/alpha_gen/api/app.py` | modify | Include tasks router |

**Implementation Steps**:

1. Create `src/alpha_gen/api/models/tasks.py`:
   ```python
   class TaskStatus(str, Enum):
       PENDING = "pending"
       RUNNING = "running"
       COMPLETED = "completed"
       FAILED = "failed"
   
   class TaskInfo(BaseModel):
       task_id: str
       status: TaskStatus
       created_at: datetime
       started_at: datetime | None = None
       completed_at: datetime | None = None
       result: dict[str, Any] | None = None
       error: str | None = None
       progress: int | None = None  # 0-100
   
   class TaskResponse(BaseModel):
       task_id: str
       status: TaskStatus
       message: str
   ```

2. Create `src/alpha_gen/api/tasks/registry.py`:
   - `TaskRegistry` class with in-memory dict
   - `register_task(task_id, task_info)` - store task
   - `get_task(task_id)` - retrieve task info
   - `update_task(task_id, **kwargs)` - update fields
   - `list_tasks(user_id)` - list user's tasks (optional)
   - Use `asyncio.Lock` for thread safety

3. Create `src/alpha_gen/api/tasks/executor.py`:
   - `execute_task(task_id, coro, registry)` - wrapper function
   - Updates registry with status changes
   - Catches exceptions and stores error
   - Returns task_id immediately

4. Create `src/alpha_gen/api/routes/tasks.py`:
   - `GET /api/v1/tasks/{task_id}` - get task status
   - `GET /api/v1/tasks` - list all tasks for current user
   - `DELETE /api/v1/tasks/{task_id}` - cancel/delete task

5. Update all endpoint files to support async:
   - Add `async_mode: bool = False` to request models
   - If `async_mode=True`:
     - Generate task_id (UUID)
     - Register task as PENDING
     - Use `BackgroundTasks` to run agent
     - Return `TaskResponse` immediately
   - If `async_mode=False`:
     - Run synchronously as before
     - Return full response

6. Update `src/alpha_gen/api/app.py`:
   - Include tasks router
   - Initialize task registry on startup

**Verification**:
- [ ] `just lint` passes
- [ ] `just type-check` passes
- [ ] Sync requests work as before
- [ ] Async request returns task_id immediately
- [ ] Task status endpoint shows progress
- [ ] Completed task shows result
- [ ] Failed task shows error
- [ ] Can list all tasks

**Completion Gate**:
> This phase is NOT complete until the user has reviewed the work and explicitly confirmed it is done. Do not proceed to dependent phases or mark this phase as finished without user approval.

**Outputs**:
- Optional async execution for all endpoints
- Task status polling endpoint
- In-memory task registry

**After completing this phase**:
```bash
# Commit your changes on the current branch
git add .
git commit -m "feat(api): add background task system with async execution"

# Tag the completion of this phase
git tag -a phase/5-background-tasks -m "Phase 5: Background Task System complete"

# Push commit and tag to remote (optional)
git push && git push --tags
```

---

### Phase 6: API Test Suite

**Objective**: Create comprehensive test suite for all API endpoints.

**Complexity**: medium  
**Estimated Time**: 45 min

**Prerequisites**: Phase 5 complete (all features implemented)

**Context for this Phase**:
- Project uses pytest with `pytest-asyncio`
- Tests are in `tests/` directory
- Use `TestClient` from FastAPI for sync tests
- Use `AsyncClient` from httpx for async tests
- Mock external dependencies (Alpha Vantage, LLM)
- Follow existing test patterns in `tests/`

**Files**:
| File | Action | Purpose |
|------|--------|---------|
| `tests/api/__init__.py` | create | API tests package |
| `tests/api/conftest.py` | create | Fixtures for API tests |
| `tests/api/test_auth.py` | create | Auth endpoint tests |
| `tests/api/test_research.py` | create | Research endpoint tests |
| `tests/api/test_gather.py` | create | Gather endpoint tests |
| `tests/api/test_opportunities.py` | create | Opportunities endpoint tests |
| `tests/api/test_news.py` | create | News endpoint tests |
| `tests/api/test_analyze.py` | create | Analyze endpoint tests |
| `tests/api/test_tasks.py` | create | Task status endpoint tests |

**Implementation Steps**:

1. Create `tests/api/conftest.py`:
   - `client` fixture: TestClient with test app
   - `auth_headers` fixture: Generate mock JWT with test claims, return headers
   - `mock_oidc_provider` fixture: Mock JWKS endpoint responses
   - `mock_alpha_vantage` fixture: Mock API responses
   - `mock_llm` fixture: Mock LLM responses
   - `test_app` fixture: Create app with test OIDC config

2. Create `tests/api/test_auth.py`:
   - `test_valid_jwt_accepted` - valid token from OIDC provider
   - `test_invalid_signature_rejected` - tampered token
   - `test_expired_token_rejected` - past expiry
   - `test_wrong_issuer_rejected` - issuer mismatch
   - `test_missing_token_rejected` - no Authorization header
   - `test_malformed_auth_header` - invalid Bearer format
   - `test_get_current_user` - extracts user info from valid token

3. Create `tests/api/test_research.py`:
   - `test_research_sync_success` - valid request, sync mode
   - `test_research_sync_invalid_ticker` - validation error
   - `test_research_unauthorized` - no token
   - `test_research_async_mode` - returns task_id
   - `test_research_skip_gather` - skip_gather flag works

4. Create `tests/api/test_gather.py`:
   - `test_gather_single_ticker` - one ticker
   - `test_gather_multiple_tickers` - multiple tickers
   - `test_gather_invalid_ticker` - validation error
   - `test_gather_async_mode` - returns task_id
   - `test_gather_unauthorized` - no token

5. Create `tests/api/test_opportunities.py`:
   - `test_opportunities_default_limit` - default limit
   - `test_opportunities_custom_limit` - custom limit
   - `test_opportunities_invalid_limit` - out of range
   - `test_opportunities_async_mode` - returns task_id

6. Create `tests/api/test_news.py`:
   - `test_news_success` - returns analysis
   - `test_news_async_mode` - returns task_id
   - `test_news_unauthorized` - no token

7. Create `tests/api/test_analyze.py`:
   - `test_analyze_ticker_only` - without news
   - `test_analyze_with_news` - include_news=True
   - `test_analyze_invalid_ticker` - validation error
   - `test_analyze_async_mode` - returns task_id

8. Create `tests/api/test_tasks.py`:
   - `test_get_task_status_pending` - task in progress
   - `test_get_task_status_completed` - finished task
   - `test_get_task_status_failed` - failed task
   - `test_get_nonexistent_task` - 404 error
   - `test_list_tasks` - list all user tasks
   - `test_delete_task` - remove task

**Verification**:
- [ ] `just test` passes all tests
- [ ] `just test-coverage` shows 80%+ coverage for api/ module
- [ ] All endpoints have at least one test
- [ ] Auth tests cover all flows
- [ ] Task tests cover all states

**Completion Gate**:
> This phase is NOT complete until the user has reviewed the work and explicitly confirmed it is done. Do not proceed to dependent phases or mark this phase as finished without user approval.

**Outputs**:
- Comprehensive test suite for API
- Fixtures for future API tests
- High test coverage

**After completing this phase**:
```bash
# Commit your changes on the current branch
git add .
git commit -m "test(api): add comprehensive API test suite"

# Tag the completion of this phase
git tag -a phase/6-api-tests -m "Phase 6: API Test Suite complete"

# Push commit and tag to remote (optional)
git push && git push --tags
```

---

### Phase 7: CLI Server Command

**Objective**: Add `alpha-gen api` CLI command to start the API server.

**Complexity**: low  
**Estimated Time**: 20 min

**Prerequisites**: Phase 6 complete (API fully tested)

**Context for this Phase**:
- CLI uses Typer (in `src/alpha_gen/cli/`)
- Commands are in `src/alpha_gen/cli/commands/`
- Main CLI is in `src/alpha_gen/cli/main.py`
- Use uvicorn to run FastAPI
- Support host, port, reload options

**Files**:
| File | Action | Purpose |
|------|--------|---------|
| `src/alpha_gen/cli/commands/api.py` | create | API server command |
| `src/alpha_gen/cli/main.py` | modify | Register api command |
| `justfile` | modify | Add api-related recipes |
| `pyproject.toml` | modify | Add api entry point (optional) |

**Implementation Steps**:

1. Create `src/alpha_gen/cli/commands/api.py`:
   ```python
   import typer
   import uvicorn
   from alpha_gen.api import create_app
   
   @typer_app.command("api")
   def run_api(
       host: str = typer.Option("0.0.0.0", "--host", "-h"),
       port: int = typer.Option(8000, "--port", "-p"),
       reload: bool = typer.Option(False, "--reload", "-r"),
       workers: int = typer.Option(1, "--workers", "-w"),
   ) -> None:
       """Start the Alpha Gen API server."""
       if reload:
           uvicorn.run("alpha_gen.api:app", host=host, port=port, reload=True)
       else:
           app = create_app()
           uvicorn.run(app, host=host, port=port, workers=workers)
   ```

2. Update `src/alpha_gen/cli/main.py`:
   - Import api command from `commands.api`
   - Add to typer app

3. Update `justfile`:
   ```just
   # API server
   api:  # Start API server (default host/port)
       uv run alpha-gen api
   
   api-dev:  # Start API server with reload
       uv run alpha-gen api --reload
   
   api-prod host port workers:  # Start API server for production
       uv run alpha-gen api --host {{host}} --port {{port}} --workers {{workers}}
   ```

4. Optional: Add script entry point in `pyproject.toml`:
   ```toml
   [project.scripts]
   alpha-gen-api = "alpha_gen.api:run_server"
   ```

**Verification**:
- [ ] `just lint` passes
- [ ] `just type-check` passes
- [ ] `alpha-gen api --help` shows command
- [ ] `alpha-gen api` starts server on port 8000
- [ ] `alpha-gen api --port 3000` starts on port 3000
- [ ] `alpha-gen api --reload` enables auto-reload
- [ ] Server responds to `GET /health`
- [ ] OpenAPI docs accessible at `/docs`

**Completion Gate**:
> This phase is NOT complete until the user has reviewed the work and explicitly confirmed it is done. Do not proceed to dependent phases or mark this phase as finished without user approval.

**Outputs**:
- CLI command to start API server
- Justfile recipes for API management
- Server runs and serves all endpoints

**After completing this phase**:
```bash
# Commit your changes on the current branch
git add .
git commit -m "feat(cli): add api server command"

# Tag the completion of this phase
git tag -a phase/7-cli-server -m "Phase 7: CLI Server Command complete"

# Push commit and tag to remote (optional)
git push && git push --tags
```

---

## Phase Dependencies

```
Phase 1 (setup)
    └── Phase 2 (auth) ─────────────────────────────┐
            └── Phase 3 (research/gather)           │
                    └── Phase 4 (opps/news/analyze) │
                            └── Phase 5 (tasks)    │
                                    └── Phase 6 (tests)
                                            └── Phase 7 (cli)
```

All phases are sequential. Each phase builds on the previous one.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| OIDC provider unavailable | High | Cache JWKS, add health check, graceful degradation |
| JWKS key rotation | Medium | Cache with TTL, refresh on key validation failure |
| Token clock skew | Low | Add small leeway to expiry validation |
| Task registry memory | Low | Document limitation, suggest Redis upgrade path |
| Long-running requests timeout | Medium | Document timeout settings, recommend async mode |
| Alpha Vantage rate limits | Low | Already handled in core, document for API users |

## Questions for User

1. **CORS**: For MVP, allow all origins. Want to restrict to specific domains?
2. **API versioning**: Using `/api/v1/` prefix. Is this acceptable?
3. **JWKS caching**: Default TTL of 1 hour for cached JWKS. Want different value?
4. **Token audience validation**: Should API validate `aud` claim, or just `iss`?
