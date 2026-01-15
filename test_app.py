#!/usr/bin/env python3
"""
Test server for the refactored Imaginator package
"""
from fastapi import FastAPI, HTTPException, Header
from imaginator.orchestrator import run_full_funnel_pipeline
from imaginator.microservices import call_loader_process_text_only, call_fastsvm_process_resume, call_hermes_extract
import json
import asyncio
import uvicorn

app = FastAPI(title='Imaginator Test Server')

@app.post('/analyze')
async def analyze_resume(resume_text: str, job_ad: str, x_api_key: str = Header(...)):
    if x_api_key != '05c2765ea794c6e15374f2a63ac35da8e0e665444f6232225a3d4abfe5238c45':
        raise HTTPException(status_code=401, detail='Invalid API key')
    
    try:
        # Step 1: Get structured data from Hermes and FastSVM
        hermes_data = await call_hermes_extract(resume_text)
        svm_data = await call_fastsvm_process_resume(resume_text)
        
        # Step 2: Run the complete funnel pipeline
        result = await run_full_funnel_pipeline(
            resume_text=resume_text,
            job_ad=job_ad,
            hermes_data=hermes_data,
            svm_data=svm_data
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)