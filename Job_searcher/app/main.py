from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.v1.endpoints.simple_ats import router as simple_ats_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Job Search API", lifespan=lifespan)

app.include_router(simple_ats_router, prefix="/api/v1", tags=["ats"])

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "Job Search API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)