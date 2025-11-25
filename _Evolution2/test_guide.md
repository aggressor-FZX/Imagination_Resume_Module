# Evolution 2 Test Guide

## Overview

Evolution 2 introduces major enhancements to the Imaginator system including parallel LLM processing, Redis caching, HTTP connection pooling, and synthesis-based resume generation. This guide provides instructions for running and understanding the comprehensive integration test suite.

## 📋 Test Files Created

### Integration Test Suite
- **`tests/test_evolution2_integration.py`** - End-to-end synthesis pipeline validation
- **`tests/test_parallel_llm_calls.py`** - Async parallelization testing
- **`tests/test_redis_cache_integration.py`** - Distributed caching tests
- **`tests/test_performance_benchmarks.py`** - Performance regression benchmarks
- **`test/run_evolution2_integration.py`** - Executable test runner script

## 🚀 Quick Start

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-asyncio psutil

# Install project in development mode
pip install -e .

# Start Redis (optional, skip tests if not available)
redis-server
```

### Run All Tests
```bash
cd /home/skystarved/Render_Dockers/Imaginator

# Run integration tests
python -m pytest tests/test_evolution2_integration.py -v --tb=short

# Run parallel processing tests
python -m pytest tests/test_parallel_llm_calls.py -v --tb=short

# Run cache tests (requires Redis)
python -m pytest tests/test_redis_cache_integration.py -v --tb=short

# Run performance benchmarks
python -m pytest tests/test_performance_benchmarks.py -v --tb=short

# Run executable test runner
python test/run_evolution2_integration.py
```

### Skip Redis Tests
If Redis is not available, run with marker:
```bash
pytest tests/ -v -m "not redis"
```

## 🧪 Test Details

### 1. End-to-End Synthesis Integration (`test_evolution2_integration.py`)

**Purpose**: Validate the complete Evolution 2 pipeline from analysis through synthesis.

**Key Tests**:
- `test_end_to_end_synthesis_integration` - Full pipeline with realistic data
- `test_parallel_llm_calls_integration` - Async parallel generation + criticism
- `test_http_pool_metrics_exposure` - Connection pooling metrics validation
- `test_realistic_workload_capacity` - 5 concurrent users simulation
- `test_synthesis_gap_propagation` - Gap analysis integration verification

**Data Used**:
- Senior Software Engineer resume (8+ years experience)
- Backend Engineer job description (AWS, microservices focused)
- Structured skills JSON (Python, AWS, leadership)
- Domain insights JSON (tech industry terms)

### 2. Parallel LLM Calls (`test_parallel_llm_calls.py`)

**Purpose**: Test asyncio.gather optimization for independent LLM operations.

**Key Tests**:
- `test_generation_criticism_parallel` - Concurrent generation and criticism
- `test_fifty_concurrent_operations` - 50 parallel LLM calls load test
- `test_thread_safety_metrics` - RUN_METRICS thread safety validation

**Performance Expectations**:
- Sequential execution: ~2.0s (4 API calls × 500ms avg)
- Parallel execution: ~0.5s (max API call time only)
- Expected improvement: 4x+ speedup

### 3. Redis Cache Integration (`test_redis_cache_integration.py`)

**Purpose**: Validate distributed caching with Redis for high-availability deployments.

**Key Tests**:
- `test_redis_cache_hit_miss` - Cache performance validation
- `test_ttl_expiration` - Time-based cache invalidation
- `test_graceful_degradation` - Fallback to in-memory cache
- `test_high_load_cache_operations` - 50 concurrent cache operations

**Cache Configuration**:
- TTL: 24 hours for successful analyses, 1 hour for low-confidence
- Key schema: SHA256 hash of normalized (resume_text + job_ad)
- Memory limit: 1GB, LRU eviction

### 4. Performance Benchmarks (`test_performance_benchmarks.py`)

**Purpose**: Regression testing to ensure performance doesn't degrade.

**Key Tests**:
- `test_parallel_vs_sequential_benchmark` - Statistical comparison (10 runs)
- `test_concurrent_capacity_by_load` - 1, 5, 10, 20 concurrent users
- `test_cache_performance_improvement` - 5x+ expected improvement on hits

**Benchmark Metrics**:
- Mean response time reduction: 75%+ with caching
- 95th percentile latency: <200ms with caching vs <800ms without
- Memory usage: <100MB increase with Redis enabled

## 🔍 Test Data & Fixtures

### Realistic Test Data
All tests use professional-grade, realistic data rather than simplistic unit test data:

**Resume Example** (Senior Software Engineer):
- 8 years experience, leadership roles
- AWS, Kubernetes, microservices skills
- Quantified achievements (40% improvement, 500K+ users)

**Job Description Example** (Backend Engineer):
- 5+ years Python/backend required
- AWS/GCP, microservices, leadership preferred
- Code reviews, mentoring responsibilities

### Structured Data Integration
- **Skills JSON**: Confidence-scored technical/soft skills
- **Domain Insights**: Industry-specific terminology and competencies
- **Analysis Results**: Gap analysis with seniority assessment

## 📊 Interpreting Results

### Successful Test Indicators
✅ **Synthesis Integration**: Final text includes gap-bridging statements
✅ **Parallel Performance**: 4x+ speedup for generation + criticism
✅ **Cache Efficiency**: 5x+ response time reduction on cache hits
✅ **Async Capacity**: <200ms for 5 concurrent realistic requests

### Expected Performance Numbers
```
Sequential Analysis:       1.8-2.5s (high variance due to LLMs)
Parallel Analysis:         0.4-0.8s (max single API call time)
Cached Repeat Analysis:    0.02-0.05s (JSON parsing + cache lookup)
5 Concurrent Users:        0.8-1.2s total (async gains)
Redis Overhead:            2-5ms per cache operation
```

### Common Issues & Solutions

#### No Redis Available
**Error**: `ConnectionError: Redis not running`
**Solution**: Tests skip automatically with `-m "not redis"`

#### Slow LLM Mocks
**Error**: Tests timeout after 30s
**Solution**: Mock responses simulate 100-500ms latency

#### Memory Pressure
**Error**: `MemoryError` during high-load tests
**Solution**: Increase system RAM or reduce concurrent operations

## 🚀 Production Readiness Checklist

To deploy Evolution 2 features:

- [ ] Run full integration test suite in staging
- [ ] Validate cache hit rates > 60% in production traffic
- [ ] Monitor parallel execution for <15% failure rate
- [ ] Confirm HTTP pool connections reused > 90%
- [ ] Test synthesis quality with human review of 20+ samples

## 📈 Monitoring & Alerting

### Key Metrics to Track
```
Synthetic metrics:
- run_metrics.stages.synthesis.duration_ms
- run_metrics.http_pool.connections_reused
- Cache hit rate: cache_hits / (cache_hits + cache_misses)
- Parallel speedup: sequential_time / parallel_time

Operational metrics:
- Analysis request rate (requests/second)
- P95 latency across all stages
- API error rate per provider
- Redis memory usage and eviction rate
```

### Health Check Endpoints
- `/health` - Overall system health
- `/keys/health` - LLM provider readiness
- Metrics exposed via `run_metrics` in response

## 🔧 Development Workflow

### Adding New Tests
1. Place in `tests/` directory following naming convention
2. Use `pytest-asyncio` for async tests
3. Include realistic data fixtures
4. Add performance assertions where relevant

### Running Individual Tests
```bash
# Run specific test function
pytest tests/test_evolution2_integration.py::TestEvolution2Integration::test_end_to_end_synthesis_integration -v

# Run specific test file
pytest tests/test_parallel_llm_calls.py -v

# Debug with breakpoint
pytest tests/test_evolution2_integration.py -v --pdb
```

### Continuous Integration
Tests run automatically on pushes to main branch with:
- `ruff` linting validation
- `mypy` type checking
- Full test suite execution (< 60s total)
- Performance regression detection

---

**Evolution 2 Test Suite**: Comprehensive validation of parallel processing, caching, synthesis, and production readiness. Tests designed with realistic workloads for accurate performance modeling.