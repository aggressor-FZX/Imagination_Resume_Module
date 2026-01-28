"""
Career Progression Integration Snippet for Imaginator
Add this code to imaginator_new_integration.py after the researcher stage completes
"""

import os
import logging
from typing import Optional, Dict, Any
from career_progression_enricher import CareerProgressionEnricher
import requests

logger = logging.getLogger(__name__)


def enrich_domain_insights_with_career_data(
    domain_insights: Dict[str, Any],
    job_title: Optional[str],
    job_ad: str
) -> Dict[str, Any]:
    """
    Enrich domain insights with career progression data from O*NET
    
    Args:
        domain_insights: Existing domain insights dict
        job_title: Extracted job title (if available)
        job_ad: Full job ad text
        
    Returns:
        Enhanced domain_insights with career_ladder, seniority, and skill data
    """
    if not job_title:
        # Try to extract from job ad
        job_title = _extract_job_title_simple(job_ad)
    
    if not job_title:
        logger.warning("[CAREER_PROGRESSION] No job title available, skipping enrichment")
        return domain_insights
    
    try:
        logger.info(f"[CAREER_PROGRESSION] Enriching for: {job_title}")
        
        # Get O*NET API credentials
        onet_auth = os.getenv("ONET_API_AUTH", "Y29naXRvbWV0cmljOjI4Mzd5cGQ=")
        
        # Search O*NET for matching occupation
        onet_url = "https://services.onetcenter.org/ws/online/search"
        onet_headers = {
            "Authorization": f"Basic {onet_auth}",
            "Accept": "application/json"
        }
        
        response = requests.get(
            onet_url,
            headers=onet_headers,
            params={"keyword": job_title, "end": 1},
            timeout=10
        )
        
        if response.status_code != 200:
            logger.warning(f"[CAREER_PROGRESSION] O*NET search failed: {response.status_code}")
            return domain_insights
        
        search_data = response.json()
        if not search_data.get("occupation"):
            logger.info(f"[CAREER_PROGRESSION] No O*NET match for: {job_title}")
            return domain_insights
        
        onet_code = search_data["occupation"][0]["code"]
        onet_title = search_data["occupation"][0]["title"]
        logger.info(f"[CAREER_PROGRESSION] Matched O*NET {onet_code}: {onet_title}")
        
        # Initialize enricher and get career data
        enricher = CareerProgressionEnricher()
        career_data = enricher.get_full_career_insights(
            job_title=job_title,
            onet_code=onet_code,
            location="United States"
        )
        
        # Merge career data into domain_insights
        domain_insights["career_ladder"] = career_data.get("career_ladder", [])
        domain_insights["seniority_info"] = career_data.get("seniority", {})
        domain_insights["onet_code"] = onet_code
        domain_insights["onet_title"] = onet_title
        
        # Add top skills if not already present
        if not domain_insights.get("top_skills"):
            domain_insights["top_skills"] = [
                s["name"] for s in career_data.get("core_skills", [])[:10]
            ]
        
        # Add skill impact analysis
        domain_insights["skill_impact"] = career_data.get("skill_impact", {})
        
        # Enhance insights list with career progression info
        if isinstance(domain_insights.get("insights"), list):
            career_insight = f"Career progression: {career_data['seniority']['level']} role requiring {career_data['seniority']['experience_required']} experience and {career_data['seniority']['typical_education']} education"
            domain_insights["insights"].append(career_insight)
        
        logger.info(f"[CAREER_PROGRESSION] Successfully enriched domain insights")
        
    except Exception as e:
        logger.error(f"[CAREER_PROGRESSION] Enrichment failed: {e}", exc_info=True)
    
    return domain_insights


def _extract_job_title_simple(job_ad: str) -> Optional[str]:
    """
    Simple fallback job title extraction from job ad
    """
    import re
    
    # Look for common patterns in first 500 chars
    text = job_ad[:500]
    
    # Pattern 1: "Position: Software Engineer" or "Job Title: ..."
    patterns = [
        r"(?:position|job title|role):\s*([A-Z][A-Za-z\s]+(?:Engineer|Developer|Scientist|Analyst|Manager|Designer|Architect))",
        r"^([A-Z][A-Za-z\s]+(?:Engineer|Developer|Scientist|Analyst|Manager|Designer|Architect))",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            if 5 < len(title) < 50:
                return title
    
    # Fallback: Look for common job keywords
    keywords = {
        "software engineer": "Software Engineer",
        "data scientist": "Data Scientist",
        "product manager": "Product Manager",
        "machine learning": "Machine Learning Engineer",
        "devops": "DevOps Engineer",
        "full stack": "Full Stack Developer",
    }
    
    text_lower = text.lower()
    for keyword, title in keywords.items():
        if keyword in text_lower:
            return title
    
    return None


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================
"""
In imaginator_new_integration.py, around line 166-180 where domain_insights is built:

    # ... existing domain_insights construction ...
    
    # NEW: Enrich with career progression data
    domain_insights = enrich_domain_insights_with_career_data(
        domain_insights=domain_insights,
        job_title=extracted_job_title,  # From earlier in the function
        job_ad=job_ad
    )
    
    # Continue with existing code...
    output = {
        "final_written_section_markdown": final_markdown,
        # ...
        "domain_insights": domain_insights,  # Now enriched!
        # ...
    }
"""
