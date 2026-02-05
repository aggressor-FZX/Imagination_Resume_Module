#!/usr/bin/env python3
"""
Career Progression & Transitional Advice Enricher
Combines O*NET + Data USA APIs to provide:
- Localized wage trends (YoY growth)
- Workforce size & growth patterns
- Career progression paths (education-based)
- Seniority levels (O*NET Job Zones)
- Skill-to-salary mapping
"""

import os
import re
import json
import requests
import httpx
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time

# API Configuration
ONET_API_BASE = "https://services.onetcenter.org/ws"
ONET_API_AUTH = os.getenv("ONET_API_AUTH", "Y29naXRvbWV0cmljOjI4Mzd5cGQ=")
DATA_USA_BASE = os.getenv("DATA_USA_BASE", "https://api.datausa.io/tesseract/data.jsonrecords")
DATA_USA_CUBE_WAGE = os.getenv("DATA_USA_CUBE_WAGE", "acs_ygo_occupation_for_median_earnings_5")
DATA_USA_CUBE_WORKFORCE = os.getenv("DATA_USA_CUBE_WORKFORCE", "acs_ygso_gender_by_occupation_c_5")
DATA_USA_CUBE_GROWTH = os.getenv("DATA_USA_CUBE_GROWTH", "bls_growth_occupation")
DATA_USA_CUBE_EDU = os.getenv("DATA_USA_CUBE_EDU")
DATA_USA_FORCE = os.getenv("DATA_USA_FORCE")
DATA_USA_GEO_LEVEL = os.getenv("DATA_USA_GEO_LEVEL", "MSA")
DATA_USA_NATION_KEY = os.getenv("DATA_USA_NATION_KEY", "01000US")

_DATAUSA_CUBE_CACHE: Dict[str, Dict[str, Any]] = {}
_DATAUSA_MEMBERS_CACHE: Dict[tuple, List[Dict[str, Any]]] = {}


class CareerProgressionEnricher:
    """
    Enriches domain insights with career progression data from O*NET + Data USA
    """
    
    def __init__(self):
        self.onet_headers = {
            "Authorization": f"Basic {ONET_API_AUTH}",
            "Accept": "application/json"
        }
    
    def get_full_career_insights(
        self,
        job_title: str,
        onet_code: str,
        location: str = "United States",
        city_geo_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive career progression insights
        
        Args:
            job_title: Target job title
            onet_code: O*NET occupation code
            location: City/state name (for display)
            city_geo_id: Data USA geo ID (e.g., "16000US0644000" for LA)
                        If None, uses national data
        
        Returns:
            Rich career progression data including:
            - Local wage trends & YoY growth
            - Workforce size & demand
            - Career ladder (entry → senior → lead)
            - Education-based progression
            - Skill upgrade paths
        """
        
        insights = {
            "job_title": job_title,
            "onet_code": onet_code,
            "location": location,
            "last_updated": datetime.now().isoformat()
        }
        
        # 1. O*NET: Get seniority level & experience requirements
        print(f"📊 Fetching O*NET data for {onet_code}...")
        onet_data = self._get_onet_summary(onet_code)
        insights["seniority"] = self._parse_job_zone(onet_data)
        
        # 2. O*NET: Get skills for progression mapping
        onet_skills = self._get_onet_skills(onet_code)
        insights["core_skills"] = onet_skills[:10]  # Top 10
        
        # 3. Data USA: Get workforce trends (national or local)
        print(f"💰 Fetching Data USA workforce trends...")
        workforce_data = self._get_workforce_trends(onet_code, city_geo_id, job_title=job_title)
        insights["workforce"] = workforce_data
        
        # 4. Data USA: Get education-based salary progression
        print(f"🎓 Fetching education-based salary data...")
        education_data = self._get_education_progression(onet_code, city_geo_id)
        insights["education_progression"] = education_data
        
        # 5. Build career ladder (synthesized)
        print(f"🪜 Building career progression ladder...")
        insights["career_ladder"] = self._build_career_ladder(
            job_title, 
            insights["seniority"],
            education_data
        )
        
        # 6. Calculate skill-to-salary impact (comparative analysis)
        print(f"📈 Analyzing skill impact on salary...")
        insights["skill_impact"] = self._analyze_skill_impact(
            onet_code, 
            onet_skills,
            workforce_data
        )
        
        return insights
    
    def _get_onet_summary(self, onet_code: str) -> Dict[str, Any]:
        """Get O*NET occupation summary (includes Job Zone)"""
        try:
            url = f"{ONET_API_BASE}/online/occupations/{onet_code}/summary"
            response = requests.get(url, headers=self.onet_headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️  O*NET summary failed: {e}")
            return {}
    
    def _get_onet_skills(self, onet_code: str) -> List[Dict[str, Any]]:
        """Get O*NET skills with importance ratings"""
        try:
            url = f"{ONET_API_BASE}/online/occupations/{onet_code}/details/skills"
            response = requests.get(url, headers=self.onet_headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "element" in data:
                skills = []
                for skill in data["element"]:
                    importance = self._extract_score(skill, "IM")
                    skills.append({
                        "name": skill.get("name", ""),
                        "description": skill.get("description", ""),
                        "importance": importance
                    })
                # Sort by importance
                skills.sort(key=lambda x: x["importance"], reverse=True)
                return skills
        except Exception as e:
            print(f"⚠️  O*NET skills failed: {e}")
        return []
    
    def _extract_score(self, element: Dict, scale_id: str = "IM") -> float:
        """
        Extract importance/level score from O*NET element
        
        Handles multiple response formats:
        - {"score": {"scale": "Importance", "value": 75}}  (skills endpoint)
        - {"score": [{"scale_id": "IM", "value": 75}, ...]}  (other endpoints)
        """
        if "score" not in element:
            return 0.0
            
        scores = element["score"]
        
        # Format 1: Single score object with "scale" and "value"
        if isinstance(scores, dict):
            if "value" in scores:
                return float(scores.get("value", 0))
            if scores.get("scale_id") == scale_id:
                return float(scores.get("value", 0))
        
        # Format 2: Array of score objects
        if isinstance(scores, list):
            for score in scores:
                if isinstance(score, dict):
                    # Check for scale_id match
                    if score.get("scale_id") == scale_id:
                        return float(score.get("value", 0))
                    # Check for scale match (Importance = IM)
                    scale_name = score.get("scale", "")
                    if (scale_id == "IM" and "importance" in scale_name.lower()) or \
                       (scale_id == "LV" and "level" in scale_name.lower()):
                        return float(score.get("value", 0))
        
        return 0.0
    
    def _parse_job_zone(self, onet_data: Dict) -> Dict[str, Any]:
        """
        Parse O*NET Job Zone into actionable seniority info
        
        Job Zones:
        1: Little or no preparation (< 1 year)
        2: Some preparation (1-2 years)
        3: Medium preparation (2-4 years)
        4: High preparation (4-10 years)
        5: Extensive preparation (10+ years, often PhD)
        """
        job_zone_map = {
            "1": {"level": "Entry", "experience": "< 1 year", "education": "High School"},
            "2": {"level": "Junior", "experience": "1-2 years", "education": "Some College"},
            "3": {"level": "Mid-Level", "experience": "2-4 years", "education": "Bachelor's"},
            "4": {"level": "Senior", "experience": "4-10 years", "education": "Bachelor's+"},
            "5": {"level": "Expert/Principal", "experience": "10+ years", "education": "Master's/PhD"}
        }
        
        # Extract job zone from O*NET data
        job_zone = onet_data.get("job_zone", {}).get("code", "3")
        zone_info = job_zone_map.get(str(job_zone), job_zone_map["3"])
        
        return {
            "job_zone": job_zone,
            "level": zone_info["level"],
            "experience_required": zone_info["experience"],
            "typical_education": zone_info["education"],
            "description": onet_data.get("job_zone", {}).get("name", "")
        }

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()

    def _get_cube_schema(self, cube: str) -> Dict[str, Any]:
        if cube in _DATAUSA_CUBE_CACHE:
            return _DATAUSA_CUBE_CACHE[cube]
        url = f"https://api.datausa.io/tesseract/cubes/{cube}"
        response = httpx.get(url, headers={"User-Agent": "Cogitometric-Career-Insights/1.0"}, timeout=20)
        response.raise_for_status()
        data = response.json()
        _DATAUSA_CUBE_CACHE[cube] = data
        return data

    def _get_level_names(self, cube: str, dimension_name: str) -> List[str]:
        schema = self._get_cube_schema(cube)
        for dim in schema.get("dimensions", []):
            if dim.get("name") == dimension_name:
                levels = []
                for hierarchy in dim.get("hierarchies", []):
                    for level in hierarchy.get("levels", []):
                        level_name = level.get("name")
                        if level_name and level_name not in levels:
                            levels.append(level_name)
                return levels
        return []

    def _get_members(self, cube: str, level: str) -> List[Dict[str, Any]]:
        cache_key = (cube, level)
        if cache_key in _DATAUSA_MEMBERS_CACHE:
            return _DATAUSA_MEMBERS_CACHE[cache_key]
        url = f"https://api.datausa.io/tesseract/members?cube={cube}&level={level}"
        response = httpx.get(url, headers={"User-Agent": "Cogitometric-Career-Insights/1.0"}, timeout=25)
        response.raise_for_status()
        data = response.json()
        members = data.get("members") or []
        _DATAUSA_MEMBERS_CACHE[cache_key] = members
        return members

    def _find_member_key(self, cube: str, level: str, target: str) -> Optional[str]:
        if not target:
            return None
        target_norm = self._normalize_text(target)
        members = self._get_members(cube, level)
        best_key = None
        best_score = 0
        for member in members:
            caption = member.get("caption") or ""
            caption_norm = self._normalize_text(caption)
            if not caption_norm:
                continue
            if target_norm in caption_norm:
                score = len(target_norm) / max(len(caption_norm), 1)
            else:
                target_tokens = set(target_norm.split())
                caption_tokens = set(caption_norm.split())
                overlap = len(target_tokens & caption_tokens)
                score = overlap / max(len(target_tokens), 1)
            if score > best_score:
                best_score = score
                best_key = member.get("key")
        return str(best_key) if best_key is not None else None

    def _pick_measure(self, cube: str, preferred: List[str], contains: Optional[str] = None) -> Optional[str]:
        schema = self._get_cube_schema(cube)
        measures = [m.get("name") for m in schema.get("measures", []) if m.get("name")]
        for name in preferred:
            if name in measures:
                return name
        if contains:
            for name in measures:
                if contains.lower() in name.lower():
                    return name
        return measures[0] if measures else None

    def _query_tesseract(
        self,
        cube: str,
        drilldowns: str,
        measures: str,
        include: Optional[str] = None,
        time_param: Optional[str] = None,
        limit: Optional[str] = None,
    ) -> Dict[str, Any]:
        params = {
            "cube": cube,
            "drilldowns": drilldowns,
            "measures": measures,
        }
        if include:
            params["include"] = include
        if time_param:
            params["time"] = time_param
        if limit:
            params["limit"] = limit
        if DATA_USA_FORCE:
            params["force"] = DATA_USA_FORCE
        response = httpx.get(DATA_USA_BASE, params=params, headers={"User-Agent": "Cogitometric-Career-Insights/1.0"}, timeout=25)
        response.raise_for_status()
        return response.json()

    def _query_datausa_with_fallback(
        self,
        soc_code: str,
        geo_id: Optional[str],
        params: Dict[str, Any],
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Query Data USA with correct filter keys.

        Data USA filters use the exact column names (e.g., Occupation, ID Occupation,
        Geography, ID Geography). We try common variants to avoid 404s/empty data.
        """
        url = DATA_USA_BASE
        headers = {
            "User-Agent": "Cogitometric-Career-Insights/1.0 (careers@cogitometric.org)"
        }

        drilldowns = str(params.get("drilldowns", ""))
        if "Detailed Occupation" in drilldowns:
            occ_keys = ["Detailed Occupation", "ID Detailed Occupation"]
        elif "SOC" in drilldowns:
            occ_keys = ["SOC", "ID SOC"]
        else:
            occ_keys = ["Occupation", "ID Occupation"]

        geo_keys = ["Geography", "ID Geography"]

        filter_sets = []
        for occ_key in occ_keys:
            if geo_id:
                for geo_key in geo_keys:
                    filter_sets.append({occ_key: soc_code, geo_key: geo_id})
            else:
                filter_sets.append({occ_key: soc_code})

        # Legacy fallbacks
        filter_sets.append({"occupation": soc_code, "geo": geo_id} if geo_id else {"occupation": soc_code})

        last_error = None
        for filters in filter_sets:
            request_params = {**params}
            if DATA_USA_CUBE:
                request_params["cube"] = DATA_USA_CUBE
            if DATA_USA_FORCE:
                request_params["force"] = DATA_USA_FORCE

            # Convert year=latest to time=Year.latest for Tesseract
            if request_params.get("year") == "latest" and "time" not in request_params:
                request_params.pop("year", None)
                request_params["time"] = "Year.latest"

            # Prefer include filters for Tesseract
            include_pairs = [f"{k}:{v}" for k, v in filters.items() if v]
            if include_pairs:
                request_params["include"] = ";".join(include_pairs)
            try:
                response = httpx.get(url, params=request_params, headers=headers, timeout=15)
                response.raise_for_status()
                if "application/json" not in response.headers.get("content-type", ""):
                    raise ValueError("Data USA returned non-JSON content")
                data = response.json()
                if data.get("data"):
                    return data
            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error
        return {}
    
    def _get_workforce_trends(
        self,
        onet_code: str,
        geo_id: Optional[str] = None,
        job_title: str = "",
    ) -> Dict[str, Any]:
        """
        Get workforce size & wage trends from Data USA
        
        Returns YoY growth by comparing latest 2 years (uses drilldowns/measures)
        Also includes time-series data for trend analysis (5 years)
        """
        try:
            soc_code = onet_code.replace("-", "").replace(".", "")[:6]

            occ_key_wage = self._find_member_key(
                DATA_USA_CUBE_WAGE,
                "Occupation",
                job_title or soc_code,
            )
            occ_key_workforce = self._find_member_key(
                DATA_USA_CUBE_WORKFORCE,
                "Occupation",
                job_title or soc_code,
            )
            occ_key_growth = self._find_member_key(
                DATA_USA_CUBE_GROWTH,
                "Occupation",
                job_title or soc_code,
            )

            include_geo = f"{DATA_USA_GEO_LEVEL}:{geo_id}" if geo_id else f"Nation:{DATA_USA_NATION_KEY}"
            include_wage_parts = [include_geo, f"Occupation:{occ_key_wage}" if occ_key_wage else ""]
            include_workforce_parts = [include_geo, f"Occupation:{occ_key_workforce}" if occ_key_workforce else ""]
            include_growth_parts = [f"Occupation:{occ_key_growth}" if occ_key_growth else ""]

            include_wage = ";".join([p for p in include_wage_parts if p])
            include_workforce = ";".join([p for p in include_workforce_parts if p])
            include_growth = ";".join([p for p in include_growth_parts if p])

            wage_measure = self._pick_measure(
                DATA_USA_CUBE_WAGE,
                [
                    "Median Earings by Occupation: Occupation",
                    "Median Earnings by Occupation: Occupation",
                ],
                contains="Median"
            )
            workforce_measure = self._pick_measure(
                DATA_USA_CUBE_WORKFORCE,
                ["Workforce by Occupation and Gender"],
                contains="Workforce"
            )
            growth_measures = [
                m for m in [
                    "Occupation Employment Change Percent",
                    "Occupation Employment",
                    "Occupation Employment Openings",
                ] if self._pick_measure(DATA_USA_CUBE_GROWTH, [m], contains=m)
            ]

            wage_rows = []
            workforce_rows = []
            growth_rows = []

            if wage_measure:
                wage_data = self._query_tesseract(
                    cube=DATA_USA_CUBE_WAGE,
                    drilldowns="Year",
                    measures=wage_measure,
                    include=include_wage or None,
                    time_param="Year.trailing.3",
                )
                wage_rows = wage_data.get("data", [])

            if workforce_measure:
                workforce_data = self._query_tesseract(
                    cube=DATA_USA_CUBE_WORKFORCE,
                    drilldowns="Year",
                    measures=workforce_measure,
                    include=include_workforce or None,
                    time_param="Year.trailing.3",
                )
                workforce_rows = workforce_data.get("data", [])

            if growth_measures:
                growth_data = self._query_tesseract(
                    cube=DATA_USA_CUBE_GROWTH,
                    drilldowns="Year",
                    measures=",".join(growth_measures),
                    include=include_growth or None,
                    time_param="Year.latest",
                )
                growth_rows = growth_data.get("data", [])

            if not wage_rows and not workforce_rows and not growth_rows:
                return self._empty_workforce_data()

            def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return sorted(rows, key=lambda x: int(x.get("Year", 0)), reverse=True)

            wage_rows = _sort_rows(wage_rows)
            workforce_rows = _sort_rows(workforce_rows)

            latest_wage = wage_rows[0].get(wage_measure) if wage_rows and wage_measure else None
            prev_wage = wage_rows[1].get(wage_measure) if len(wage_rows) > 1 and wage_measure else None

            latest_workforce = workforce_rows[0].get(workforce_measure) if workforce_rows and workforce_measure else None
            prev_workforce = workforce_rows[1].get(workforce_measure) if len(workforce_rows) > 1 and workforce_measure else None

            wage_growth = (
                ((latest_wage - prev_wage) / prev_wage * 100)
                if latest_wage is not None and prev_wage
                else 0
            )
            workforce_growth = (
                ((latest_workforce - prev_workforce) / prev_workforce * 100)
                if latest_workforce is not None and prev_workforce
                else 0
            )

            bls_growth_pct = None
            if growth_rows and growth_measures:
                for measure in growth_measures:
                    if "Change Percent" in measure:
                        bls_growth_pct = growth_rows[0].get(measure)
                        break

            workforce_growth_final = bls_growth_pct if bls_growth_pct is not None else workforce_growth

            return {
                "workforce_size": int(latest_workforce) if latest_workforce is not None else None,
                "average_wage": int(latest_wage) if latest_wage is not None else None,
                "yoy_growth_workforce": round(workforce_growth_final, 1) if workforce_growth_final is not None else None,
                "yoy_growth_wage": round(wage_growth, 1) if wage_growth is not None else None,
                "workforce_delta_3yr": None,
                "year_latest": wage_rows[0].get("Year") if wage_rows else (workforce_rows[0].get("Year") if workforce_rows else "Unknown"),
                "year_previous": wage_rows[1].get("Year") if len(wage_rows) > 1 else (workforce_rows[1].get("Year") if len(workforce_rows) > 1 else "Unknown"),
                "time_series": (workforce_rows or wage_rows)[:5],
                "data_source": "Data USA (Tesseract: ACS + BLS)"
            }
        except Exception as e:
            print(f"⚠️  Data USA workforce trends failed: {e}")
            return self._empty_workforce_data()
    
    def _empty_workforce_data(self) -> Dict[str, Any]:
        """Return empty workforce data structure"""
        return {
            "workforce_size": None,
            "average_wage": None,
            "yoy_growth_workforce": None,
            "yoy_growth_wage": None,
            "workforce_delta_3yr": None,
            "time_series": [],
            "data_source": "Data unavailable"
        }
    
    def calculate_market_intel(
        self, 
        workforce_data: Dict[str, Any], 
        onet_summary: Dict[str, Any],
        job_title: str = "",
        location: str = ""
    ) -> Dict[str, Any]:
        """
        Calculate the 3 Market Insight Markers:
        A. Demand Score (Workforce Delta) - 3-year growth
        B. Shortage Indicator (Wage vs. Workforce growth)
        C. Bright Outlook Calibration (O*NET + Local trends)
        
        Args:
            workforce_data: Output from _get_workforce_trends()
            onet_summary: Output from _get_onet_summary()
            job_title: Job title for narrative generation
            location: Location name for narrative generation
            
        Returns:
            Market intel dict with status, labels, and narrative
        """
        # Extract workforce growth metrics
        workforce_delta_3yr = workforce_data.get("workforce_delta_3yr")
        workforce_growth = workforce_data.get("yoy_growth_workforce", 0) or 0
        wage_growth = workforce_data.get("yoy_growth_wage", 0) or 0
        
        # Use 3-year delta if available, otherwise fall back to YoY
        workforce_growth_pct = workforce_delta_3yr if workforce_delta_3yr is not None else workforce_growth
        
        # Check for Bright Outlook in O*NET summary
        has_bright_outlook = False
        bright_outlook_tags = []
        if onet_summary:
            summary_str = json.dumps(onet_summary).lower()
            # Look for Bright Outlook indicators
            bright_outlook_keywords = [
                "bright outlook",
                "rapid growth",
                "numerous openings",
                "green occupation",
                "projected growth"
            ]
            for keyword in bright_outlook_keywords:
                if keyword in summary_str:
                    has_bright_outlook = True
                    bright_outlook_tags.append(keyword)
            
            # Also check for explicit bright_outlook field
            if onet_summary.get("bright_outlook") or onet_summary.get("bright_outlook_category"):
                has_bright_outlook = True
        
        # A. Demand Score (Workforce Delta) - 3-year comparison
        if workforce_growth_pct > 10:
            demand_label = "High Demand"
        elif workforce_growth_pct > 0:
            demand_label = "Stable Market"
        else:
            demand_label = "Market Saturation"
        
        # B. Shortage Indicator (Wages outpace workforce growth)
        # If workforce is growing slowly but wages spike > 7% YoY, it's a shortage
        is_shortage = False
        if wage_growth > 7 and (workforce_growth < 5 or workforce_growth_pct < 5):
            # Wages growing faster than workforce suggests shortage
            if wage_growth > (workforce_growth * 1.5) or wage_growth > (workforce_growth_pct * 1.5):
                is_shortage = True
        
        # C. Bright Outlook Calibration
        if has_bright_outlook and workforce_growth_pct > 10:
            status_label = "Career Rocket Ship 🚀"
        elif has_bright_outlook and workforce_growth_pct < 0:
            status_label = "Regional Pivot Required"
        elif is_shortage:
            status_label = "Critical Shortage"
        elif has_bright_outlook:
            status_label = f"Bright Outlook - {demand_label}"
        else:
            status_label = demand_label
        
        # Generate narrative
        narrative = self._generate_market_narrative(
            job_title, location, status_label, demand_label, 
            workforce_growth_pct, wage_growth, has_bright_outlook, is_shortage
        )
        
        return {
            "status": status_label,
            "demand_label": demand_label,
            "is_shortage": is_shortage,
            "workforce_growth_pct": round(workforce_growth_pct, 1) if workforce_growth_pct is not None else 0,
            "wage_growth_pct": round(wage_growth, 1),
            "has_bright_outlook": has_bright_outlook,
            "bright_outlook_tags": bright_outlook_tags,
            "average_wage": workforce_data.get("average_wage"),
            "workforce_size": workforce_data.get("workforce_size"),
            "data_year": workforce_data.get("year_latest") or workforce_data.get("year", "Unknown"),
            "narrative": narrative
        }
    
    def _generate_market_narrative(
        self,
        job_title: str,
        location: str,
        status_label: str,
        demand_label: str,
        workforce_growth_pct: float,
        wage_growth_pct: float,
        has_bright_outlook: bool,
        is_shortage: bool
    ) -> str:
        """Generate human-readable market narrative"""
        location_display = location if location and location != "United States" else "your area"
        job_display = job_title if job_title else "this role"
        
        parts = []
        
        # Opening: Bright Outlook status
        if has_bright_outlook:
            parts.append(
                f"Your chosen pivot into {job_display} aligns with a Bright Outlook status nationally."
            )
        else:
            parts.append(f"Market analysis for {job_display} in {location_display}:")
        
        # Local trend analysis
        if workforce_growth_pct > 10:
            parts.append(
                f"Locally, in {location_display}, our analysis of Data USA trends shows "
                f"a {workforce_growth_pct:.1f}% workforce expansion over the past 3 years—"
                f"indicating strong market growth."
            )
        elif workforce_growth_pct > 0:
            parts.append(
                f"Locally, in {location_display}, workforce growth has been steady "
                f"({workforce_growth_pct:.1f}% over 3 years), suggesting a stable market."
            )
        elif workforce_growth_pct < 0:
            parts.append(
                f"Locally, in {location_display}, workforce has declined "
                f"({abs(workforce_growth_pct):.1f}% over 3 years), indicating market saturation."
            )
        
        # Shortage indicator
        if is_shortage:
            parts.append(
                f"Notably, wages are rising {wage_growth_pct:.1f}% year-over-year—"
                f"signaling that companies are fighting for talent with your specific skill architecture. "
                f"This suggests a critical shortage of qualified professionals in this metro area."
            )
        elif wage_growth_pct > 5:
            parts.append(
                f"Wage growth of {wage_growth_pct:.1f}% year-over-year indicates "
                f"healthy demand for skilled professionals."
            )
        
        # Combine parts
        return " ".join(parts) if parts else f"Market data for {job_display} in {location_display}."
    
    def _get_education_progression(
        self, 
        onet_code: str,
        geo_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get salary by education level from Data USA using drilldowns/measures.
        """
        if not DATA_USA_CUBE_EDU:
            return []
        try:
            soc_code = onet_code.replace("-", "").replace(".", "")[:6]
            param_variants = [
                {"drilldowns": "Education Level", "measures": "Average Wage"},
                {"drilldowns": "Education Level", "measures": "avg_wage"},
            ]

            data = {}
            last_error = None
            include_geo = f"{DATA_USA_GEO_LEVEL}:{geo_id}" if geo_id else f"Nation:{DATA_USA_NATION_KEY}"
            for base_params in param_variants:
                try:
                    data = self._query_tesseract(
                        cube=DATA_USA_CUBE_EDU,
                        drilldowns=base_params["drilldowns"],
                        measures=base_params["measures"],
                        include=include_geo,
                        time_param="Year.latest",
                    )
                    if data.get("data"):
                        break
                except Exception as e:
                    last_error = e
                    continue
            if not data.get("data") and last_error:
                raise last_error
            rows = data.get("data", [])
            if not rows:
                return []
            education_map = {
                "High School or GED": 1,
                "Some College": 2,
                "Associate's Degree": 3,
                "Bachelor's Degree": 4,
                "Master's Degree": 5,
                "Doctoral Degree": 6,
                "Professional Degree": 6
            }
            edu_wages = []
            for item in rows:
                edu_level = item.get("Education Level", "")
                wage_val = item.get("Average Wage", 0)
                try:
                    wage = int(float(wage_val)) if wage_val else 0
                except:
                    wage = 0
                if wage > 0 and edu_level:
                    edu_wages.append({
                        "education": edu_level,
                        "average_wage": wage,
                        "sort_order": education_map.get(edu_level, 0)
                    })
            edu_wages.sort(key=lambda x: x["sort_order"])
            for i in range(len(edu_wages)):
                if i > 0:
                    prev_wage = edu_wages[i-1]["average_wage"]
                    curr_wage = edu_wages[i]["average_wage"]
                    increase = curr_wage - prev_wage
                    increase_pct = (increase / prev_wage * 100) if prev_wage > 0 else 0
                    edu_wages[i]["increase_from_previous"] = {
                        "amount": increase,
                        "percentage": round(increase_pct, 1)
                    }
            return edu_wages
        except Exception as e:
            print(f"⚠️  Data USA education progression failed: {e}")
            return []
    
    def _build_career_ladder(
        self,
        job_title: str,
        seniority: Dict[str, Any],
        education_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build career progression ladder
        
        Combines:
        - O*NET job zone (experience levels)
        - Data USA education progression (degree impact on salary)
        """
        base_title_parts = job_title.replace("Senior", "").replace("Junior", "").replace("Lead", "").strip()
        
        ladder = [
            {
                "level": "Entry / Junior",
                "title": f"Junior {base_title_parts}",
                "experience": "0-2 years",
                "education": "Bachelor's Degree",
                "key_focus": "Learn fundamentals, build portfolio projects",
                "salary_range": self._get_edu_wage(education_data, "Bachelor's Degree")
            },
            {
                "level": "Mid-Level",
                "title": base_title_parts,
                "experience": "2-5 years",
                "education": "Bachelor's Degree",
                "key_focus": "Take ownership of features/projects, mentor juniors",
                "salary_range": self._get_edu_wage(education_data, "Bachelor's Degree", multiplier=1.2)
            },
            {
                "level": "Senior",
                "title": f"Senior {base_title_parts}",
                "experience": "5-10 years",
                "education": "Bachelor's or Master's",
                "key_focus": "Lead complex projects, drive technical decisions",
                "salary_range": self._get_edu_wage(education_data, "Master's Degree")
            },
            {
                "level": "Lead / Principal",
                "title": f"Lead/Principal {base_title_parts}",
                "experience": "10+ years",
                "education": "Master's Degree preferred",
                "key_focus": "Strategic planning, cross-team leadership, architecture",
                "salary_range": self._get_edu_wage(education_data, "Master's Degree", multiplier=1.3)
            }
        ]
        
        try:
            soc_code = onet_code.replace("-", "").replace(".", "")[:6]

            # Resolve member keys for occupation and geography
            occ_key_wage = self._find_member_key(
                DATA_USA_CUBE_WAGE,
                "Occupation",
                job_title or soc_code,
            )
            occ_key_workforce = self._find_member_key(
                DATA_USA_CUBE_WORKFORCE,
                "Occupation",
                job_title or soc_code,
            )
            occ_key_growth = self._find_member_key(
                DATA_USA_CUBE_GROWTH,
                "Occupation",
                job_title or soc_code,
            )

            include_geo = None
            if geo_id:
                include_geo = f"{DATA_USA_GEO_LEVEL}:{geo_id}"
            else:
                include_geo = f"Nation:{DATA_USA_NATION_KEY}"

            include_wage = ";".join([
                include_geo,
                f"Occupation:{occ_key_wage}" if occ_key_wage else "",
            ]).strip(";")
            include_workforce = ";".join([
                include_geo,
                f"Occupation:{occ_key_workforce}" if occ_key_workforce else "",
            ]).strip(";")
            include_growth = ";".join([
                f"Occupation:{occ_key_growth}" if occ_key_growth else "",
            ]).strip(";")

            wage_measure = self._pick_measure(
                DATA_USA_CUBE_WAGE,
                [
                    "Median Earings by Occupation: Occupation",
                    "Median Earnings by Occupation: Occupation",
                ],
                contains="Median"
            )
            workforce_measure = self._pick_measure(
                DATA_USA_CUBE_WORKFORCE,
                ["Workforce by Occupation and Gender"],
                contains="Workforce"
            )
            growth_measure = self._pick_measure(
                DATA_USA_CUBE_GROWTH,
                [
                    "Occupation Employment Change Percent",
                    "Occupation Employment",
                ],
                contains="Change Percent"
            )

            wage_rows = []
            workforce_rows = []
            growth_rows = []

            if wage_measure:
                wage_data = self._query_tesseract(
                    cube=DATA_USA_CUBE_WAGE,
                    drilldowns="Year",
                    measures=wage_measure,
                    include=include_wage or None,
                    time_param="Year.trailing.3",
                )
                wage_rows = wage_data.get("data", [])

            if workforce_measure:
                workforce_data = self._query_tesseract(
                    cube=DATA_USA_CUBE_WORKFORCE,
                    drilldowns="Year",
                    measures=workforce_measure,
                    include=include_workforce or None,
                    time_param="Year.trailing.3",
                )
                workforce_rows = workforce_data.get("data", [])

            if growth_measure:
                growth_data = self._query_tesseract(
                    cube=DATA_USA_CUBE_GROWTH,
                    drilldowns="Year",
                    measures=growth_measure,
                    include=include_growth or None,
                    time_param="Year.latest",
                )
                growth_rows = growth_data.get("data", [])

            if not wage_rows and not workforce_rows and not growth_rows:
                return self._empty_workforce_data()

            def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                return sorted(rows, key=lambda x: int(x.get("Year", 0)), reverse=True)

            wage_rows = _sort_rows(wage_rows)
            workforce_rows = _sort_rows(workforce_rows)

            latest_wage = None
            prev_wage = None
            if wage_measure and wage_rows:
                latest_wage = wage_rows[0].get(wage_measure)
                if len(wage_rows) >= 2:
                    prev_wage = wage_rows[1].get(wage_measure)

            latest_workforce = None
            prev_workforce = None
            if workforce_measure and workforce_rows:
                latest_workforce = workforce_rows[0].get(workforce_measure)
                if len(workforce_rows) >= 2:
                    prev_workforce = workforce_rows[1].get(workforce_measure)

            wage_growth = (
                ((latest_wage - prev_wage) / prev_wage * 100)
                if latest_wage is not None and prev_wage
                else 0
            )
            workforce_growth = (
                ((latest_workforce - prev_workforce) / prev_workforce * 100)
                if latest_workforce is not None and prev_workforce
                else 0
            )

            # Use BLS growth percent if available
            bls_growth_pct = None
            if growth_measure and growth_rows:
                bls_growth_pct = growth_rows[0].get(growth_measure)

            workforce_growth_final = bls_growth_pct if bls_growth_pct is not None else workforce_growth

            return {
                "workforce_size": int(latest_workforce) if latest_workforce is not None else None,
                "average_wage": int(latest_wage) if latest_wage is not None else None,
                "yoy_growth_workforce": round(workforce_growth_final, 1) if workforce_growth_final is not None else None,
                "yoy_growth_wage": round(wage_growth, 1) if wage_growth is not None else None,
                "workforce_delta_3yr": None,
                "year_latest": wage_rows[0].get("Year") if wage_rows else (workforce_rows[0].get("Year") if workforce_rows else "Unknown"),
                "year_previous": wage_rows[1].get("Year") if len(wage_rows) > 1 else (workforce_rows[1].get("Year") if len(workforce_rows) > 1 else "Unknown"),
                "time_series": (workforce_rows or wage_rows)[:5],
                "data_source": "Data USA (Tesseract: ACS + BLS)"
            }
        except Exception as e:
            print(f"⚠️  Data USA workforce trends failed: {e}")
            return self._empty_workforce_data()

    def _analyze_skill_impact(
        self,
        onet_code: str,
        onet_skills: List[Dict[str, Any]],
        workforce_data: Dict[str, Any],
        current_wage: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Analyze which high-importance skills influence wage outcomes.
        """
        core_skills = onet_skills or []
        inferred_wage = workforce_data.get("average_wage") if workforce_data else None
        current_wage = current_wage or inferred_wage

        high_value_skills = [
            skill for skill in core_skills[:5]
            if skill.get("importance", 0) >= 75
        ]

        return {
            "current_role_wage": current_wage,
            "high_value_skills": [s.get("name") for s in high_value_skills if s.get("name")],
            "recommendation": (
                f"Mastering {', '.join([s.get('name') for s in high_value_skills[:3] if s.get('name')])} "
                f"is critical for advancing to senior roles in this field."
            )
        }
    
    # =========================================================================
    # CAREER PIVOT ANALYSIS - Lateral Moves & Skill Gap Detection
    # =========================================================================
    
    def _get_related_occupations(self, onet_code: str) -> List[Dict[str, Any]]:
        """
        Get related occupations (lateral moves) from O*NET
        
        These are jobs sharing similar Knowledge, Skills, and Abilities (KSAs)
        Returns 10-20 occupations ranked by "Relatedness"
        
        Endpoint: /online/occupations/{code}/details/related_occupations
        """
        try:
            url = f"{ONET_API_BASE}/online/occupations/{onet_code}/details/related_occupations"
            response = requests.get(url, headers=self.onet_headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            related = []
            if "occupation" in data:
                for occ in data["occupation"]:
                    related.append({
                        "code": occ.get("code", ""),
                        "title": occ.get("title", ""),
                        "href": occ.get("href", ""),
                        "bright_outlook": occ.get("tags", {}).get("bright_outlook", False),
                        "green": occ.get("tags", {}).get("green", False)
                    })
            
            print(f"   Found {len(related)} related occupations")
            return related
            
        except Exception as e:
            print(f"⚠️  O*NET related occupations failed: {e}")
            return []
    
    def _get_occupation_technology(self, onet_code: str) -> List[Dict[str, Any]]:
        """
        Get tools & technology skills for an occupation from O*NET
        """
        try:
            url = f"{ONET_API_BASE}/online/occupations/{onet_code}/details/technology_skills"
            response = requests.get(url, headers=self.onet_headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            tech_skills = []
            if "category" in data:
                for category in data["category"]:
                    category_name = category.get("title", {}).get("name", "")
                    examples = category.get("example", [])
                    for ex in examples[:5]:  # Top 5 per category
                        tech_skills.append({
                            "category": category_name,
                            "name": ex.get("name", ""),
                            "hot_technology": ex.get("hot_technology", False)
                        })
            
            return tech_skills
            
        except Exception as e:
            print(f"⚠️  O*NET technology skills failed: {e}")
            return []
    
    def _get_occupation_knowledge(self, onet_code: str) -> List[Dict[str, Any]]:
        """
        Get knowledge areas for an occupation from O*NET
        """
        try:
            url = f"{ONET_API_BASE}/online/occupations/{onet_code}/details/knowledge"
            response = requests.get(url, headers=self.onet_headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            knowledge = []
            if "element" in data:
                for item in data["element"]:
                    importance = self._extract_score(item, "IM")
                    knowledge.append({
                        "name": item.get("name", ""),
                        "description": item.get("description", ""),
                        "importance": importance
                    })
                knowledge.sort(key=lambda x: x["importance"], reverse=True)
            
            return knowledge
            
        except Exception as e:
            print(f"⚠️  O*NET knowledge failed: {e}")
            return []
    
    def _calculate_skill_gap(
        self,
        resume_skills: List[str],
        target_skills: List[Dict[str, Any]],
        importance_threshold: float = 60.0
    ) -> Dict[str, Any]:
        """
        Calculate the skill gap between resume skills and target job skills
        
        Uses fuzzy matching to account for:
        - "Python" → "Programming" (tech implies skill)
        - "Problem Solving" → "Complex Problem Solving" (partial match)
        - "Critical Thinking" → "Critical Thinking" (exact match)
        
        Args:
            resume_skills: List of skill names from the user's resume
            target_skills: List of skill dicts from O*NET (with importance scores)
            importance_threshold: Only consider skills with importance >= this value
            
        Returns:
            Skill gap analysis with missing skills, match percentage, etc.
        """
        # Normalize resume skills for comparison
        resume_skills_lower = {s.lower().strip() for s in resume_skills}
        
        # Skill synonym/implication mapping
        # These programming languages/tools imply the "Programming" skill
        skill_synonyms = {
            "programming": ["python", "javascript", "java", "c++", "c#", "ruby", "go", "rust", 
                          "typescript", "sql", "php", "scala", "kotlin", "swift", "coding",
                          "software development", "coding", "development"],
            "complex problem solving": ["problem solving", "troubleshooting", "debugging", 
                                        "analytical thinking", "root cause analysis"],
            "critical thinking": ["analysis", "analytical", "evaluate", "assessment", "reasoning"],
            "active learning": ["learning", "self-taught", "continuous learning", "growth mindset"],
            "reading comprehension": ["documentation", "research", "reading", "comprehension"],
            "systems analysis": ["system design", "architecture", "systems thinking", "design patterns"],
            "systems evaluation": ["testing", "qa", "quality assurance", "evaluation", "assessment"],
            "judgment and decision making": ["decision making", "prioritization", "judgement"],
            "mathematics": ["math", "statistics", "data analysis", "algorithms", "calculus"],
            "writing": ["documentation", "technical writing", "communication", "reports"],
            "speaking": ["presentation", "communication", "public speaking", "meetings"],
            "active listening": ["collaboration", "teamwork", "communication", "listening"],
            "monitoring": ["monitoring", "observability", "logging", "alerting", "metrics"],
            "coordination": ["project management", "coordination", "collaboration", "teamwork"],
            "time management": ["agile", "scrum", "sprint", "deadlines", "planning"],
            "technology design": ["architecture", "system design", "design", "blueprints"],
            "quality control analysis": ["qa", "testing", "quality", "code review"],
            "operations analysis": ["devops", "operations", "sre", "infrastructure"],
            "science": ["research", "scientific", "methodology", "experiments"],
            "instructing": ["mentoring", "teaching", "training", "coaching", "onboarding"],
            "service orientation": ["customer service", "user experience", "ux", "support"],
            "social perceptiveness": ["empathy", "user research", "stakeholder management"],
            "persuasion": ["negotiation", "stakeholder management", "sales", "influence"],
            "negotiation": ["negotiation", "contract", "stakeholder", "compromise"],
            "management of personnel resources": ["management", "leadership", "team lead", "hiring"],
            "management of material resources": ["budgeting", "resource allocation", "procurement"],
            "management of financial resources": ["budget", "financial", "cost management"],
        }
        
        # Filter to high-importance skills only
        important_target_skills = [
            s for s in target_skills 
            if s.get("importance", 0) >= importance_threshold
        ]
        
        # Calculate matches and gaps
        matched_skills = []
        missing_skills = []
        
        for skill in important_target_skills:
            skill_name = skill.get("name", "")
            skill_lower = skill_name.lower()
            
            # Check for direct match (partial matching)
            is_match = any(
                skill_lower in rs or rs in skill_lower
                for rs in resume_skills_lower
            )
            
            # Check for synonym/implication match
            if not is_match:
                synonyms = skill_synonyms.get(skill_lower, [])
                is_match = any(
                    any(syn in rs or rs in syn for syn in synonyms)
                    for rs in resume_skills_lower
                )
            
            # Check for word overlap (at least 2 significant words match)
            if not is_match:
                skill_words = set(skill_lower.replace("-", " ").split())
                # Remove common words
                common_words = {"and", "or", "the", "a", "an", "of", "to", "in", "for", "with"}
                skill_words = skill_words - common_words
                
                for rs in resume_skills_lower:
                    rs_words = set(rs.replace("-", " ").split()) - common_words
                    overlap = skill_words & rs_words
                    if len(overlap) >= 1 and len(skill_words) <= 3:
                        is_match = True
                        break
                    elif len(overlap) >= 2:
                        is_match = True
                        break
            
            if is_match:
                matched_skills.append({
                    "name": skill_name,
                    "importance": skill.get("importance", 0)
                })
            else:
                missing_skills.append({
                    "name": skill_name,
                    "importance": skill.get("importance", 0),
                    "description": skill.get("description", "")
                })
        
        # Calculate match percentage
        total_important = len(important_target_skills)
        match_count = len(matched_skills)
        match_percentage = (match_count / total_important * 100) if total_important > 0 else 0
        
        # Sort missing skills by importance (highest first)
        missing_skills.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "match_percentage": round(match_percentage, 1),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills[:5],  # Top 5 gaps
            "total_required": total_important,
            "total_matched": match_count
        }
    
    def _calculate_tech_gap(
        self,
        resume_tech: List[str],
        target_tech: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate the technology/tools gap between resume and target job
        Uses fuzzy matching for tech names
        """
        resume_tech_lower = {t.lower().strip() for t in resume_tech}
        
        # Tech synonyms and variations
        tech_synonyms = {
            "python": ["python", "py", "django", "flask", "fastapi"],
            "javascript": ["javascript", "js", "node.js", "nodejs", "react", "angular", "vue"],
            "react": ["react", "reactjs", "react.js"],
            "node.js": ["node", "nodejs", "node.js", "express"],
            "postgresql": ["postgresql", "postgres", "psql"],
            "mongodb": ["mongodb", "mongo"],
            "docker": ["docker", "containers", "containerization"],
            "aws": ["aws", "amazon web services", "ec2", "s3", "lambda"],
            "git": ["git", "github", "gitlab", "version control"],
            "sql": ["sql", "mysql", "postgresql", "sqlite", "oracle", "database"],
            "java": ["java", "spring", "spring boot", "maven", "gradle"],
            "c++": ["c++", "cpp"],
            "c#": ["c#", "csharp", ".net", "dotnet"],
            "kubernetes": ["kubernetes", "k8s"],
            "linux": ["linux", "unix", "ubuntu", "centos", "bash"],
            "agile": ["agile", "scrum", "kanban", "sprint"],
        }
        
        matched_tech = []
        missing_tech = []
        hot_tech_missing = []
        
        for tech in target_tech:
            tech_name = tech.get("name", "")
            tech_lower = tech_name.lower()
            
            # Direct match
            is_match = any(
                tech_lower in rt or rt in tech_lower
                for rt in resume_tech_lower
            )
            
            # Check synonyms
            if not is_match:
                for base_tech, synonyms in tech_synonyms.items():
                    if any(syn in tech_lower for syn in synonyms):
                        # Target needs this tech family
                        if any(any(syn in rt for syn in synonyms) for rt in resume_tech_lower):
                            is_match = True
                            break
            
            # Check word overlap
            if not is_match:
                tech_words = set(tech_lower.replace("-", " ").replace(".", " ").split())
                for rt in resume_tech_lower:
                    rt_words = set(rt.replace("-", " ").replace(".", " ").split())
                    if tech_words & rt_words:
                        is_match = True
                        break
            
            if is_match:
                matched_tech.append(tech_name)
            else:
                missing_tech.append(tech_name)
                if tech.get("hot_technology", False):
                    hot_tech_missing.append(tech_name)
        
        return {
            "matched": matched_tech,
            "missing": missing_tech[:10],  # Top 10 gaps
            "hot_tech_missing": hot_tech_missing,  # Trending tech you're missing
            "total_target": len(target_tech),
            "total_matched": len(matched_tech)
        }
    
    def generate_career_pivot_analysis(
        self,
        current_onet_code: str,
        current_job_title: str,
        resume_skills: List[str],
        resume_tech: List[str],
        current_wage: Optional[int] = None,
        location: str = "United States",
        geo_id: Optional[str] = None,
        max_pivots: int = 5
    ) -> Dict[str, Any]:
        """
        Generate comprehensive career pivot analysis
        
        This is the main entry point for career coaching insights.
        
        Args:
            current_onet_code: User's current occupation O*NET code
            current_job_title: User's current job title
            resume_skills: List of skills extracted from user's resume
            resume_tech: List of technologies/tools from user's resume
            current_wage: User's current salary (optional, for comparison)
            location: Location for display
            geo_id: Data USA geo ID for local wage data
            max_pivots: Maximum number of pivot options to analyze
            
        Returns:
            Comprehensive pivot analysis with lateral moves, skill gaps, and insights
        """
        print(f"\n🔄 Generating Career Pivot Analysis for {current_job_title}...")
        
        analysis = {
            "current_role": {
                "title": current_job_title,
                "onet_code": current_onet_code,
                "wage": current_wage
            },
            "location": location,
            "pivot_opportunities": [],
            "generated_at": datetime.now().isoformat()
        }
        
        # Step 1: Get related occupations (lateral moves)
        print(f"🔍 Finding related occupations...")
        related_occupations = self._get_related_occupations(current_onet_code)
        
        if not related_occupations:
            print("   No related occupations found")
            return analysis
        
        # Step 2: Get current job's skills and seniority for baseline
        current_skills = self._get_onet_skills(current_onet_code)
        current_summary = self._get_onet_summary(current_onet_code)
        current_seniority = self._parse_job_zone(current_summary)
        
        analysis["current_role"]["skills"] = [s["name"] for s in current_skills[:10]]
        analysis["current_role"]["seniority"] = current_seniority
        
        # Step 3: Analyze each related occupation
        print(f"📊 Analyzing {min(len(related_occupations), max_pivots)} pivot opportunities...")
        
        for i, related in enumerate(related_occupations[:max_pivots]):
            related_code = related["code"]
            related_title = related["title"]
            
            print(f"   [{i+1}/{max_pivots}] Analyzing: {related_title}")
            
            try:
                # Get target job details
                target_skills = self._get_onet_skills(related_code)
                target_tech = self._get_occupation_technology(related_code)
                target_knowledge = self._get_occupation_knowledge(related_code)
                target_summary = self._get_onet_summary(related_code)
                target_seniority = self._parse_job_zone(target_summary)
                
                # Calculate skill gap
                skill_gap = self._calculate_skill_gap(
                    resume_skills=resume_skills,
                    target_skills=target_skills,
                    importance_threshold=60.0
                )
                
                # Calculate tech gap
                tech_gap = self._calculate_tech_gap(
                    resume_tech=resume_tech,
                    target_tech=target_tech
                )
                
                # Calculate knowledge gap
                knowledge_gap = self._calculate_skill_gap(
                    resume_skills=resume_skills,  # Knowledge often overlaps with skills
                    target_skills=target_knowledge,
                    importance_threshold=60.0
                )
                
                # Build pivot opportunity
                pivot = {
                    "title": related_title,
                    "onet_code": related_code,
                    "qualification_match": skill_gap["match_percentage"],
                    "seniority": target_seniority,
                    "skill_gap": {
                        "matched": len(skill_gap["matched_skills"]),
                        "missing": skill_gap["missing_skills"],
                        "match_percentage": skill_gap["match_percentage"]
                    },
                    "tech_gap": {
                        "matched": len(tech_gap["matched"]),
                        "missing": tech_gap["missing"][:5],
                        "hot_tech_missing": tech_gap["hot_tech_missing"][:3]
                    },
                    "knowledge_gap": {
                        "missing": knowledge_gap["missing_skills"][:3]
                    }
                }
                
                # Generate coaching insight for this pivot
                pivot["insight"] = self._generate_pivot_insight(
                    current_title=current_job_title,
                    target_title=related_title,
                    qualification_match=skill_gap["match_percentage"],
                    missing_skills=skill_gap["missing_skills"],
                    missing_tech=tech_gap["missing"][:3],
                    target_seniority=target_seniority
                )
                
                analysis["pivot_opportunities"].append(pivot)
                
            except Exception as e:
                print(f"   ⚠️  Failed to analyze {related_title}: {e}")
                continue
        
        # Sort pivots by qualification match (highest first)
        analysis["pivot_opportunities"].sort(
            key=lambda x: x["qualification_match"], 
            reverse=True
        )
        
        # Add summary insights
        analysis["summary"] = self._generate_pivot_summary(
            current_title=current_job_title,
            pivots=analysis["pivot_opportunities"]
        )
        
        return analysis
    
    def _generate_pivot_insight(
        self,
        current_title: str,
        target_title: str,
        qualification_match: float,
        missing_skills: List[Dict],
        missing_tech: List[str],
        target_seniority: Dict[str, Any]
    ) -> str:
        """
        Generate natural language coaching insight for a career pivot
        """
        # Qualification level description
        if qualification_match >= 80:
            qual_desc = "highly qualified"
            effort_desc = "minimal skill development"
        elif qualification_match >= 60:
            qual_desc = "well-qualified"
            effort_desc = "some focused upskilling"
        elif qualification_match >= 40:
            qual_desc = "partially qualified"
            effort_desc = "dedicated skill development"
        else:
            qual_desc = "exploring new territory"
            effort_desc = "significant investment in new skills"
        
        # Build insight
        insight_parts = []
        
        # Opening
        insight_parts.append(
            f"Based on your experience as a {current_title}, you are "
            f"{int(qualification_match)}% qualified for a {target_title} role ({qual_desc})."
        )
        
        # Skill gaps
        if missing_skills:
            skill_names = [s["name"] for s in missing_skills[:3]]
            insight_parts.append(
                f"Skill Gap: Focus on developing '{', '.join(skill_names)}' "
                f"to strengthen your candidacy."
            )
        
        # Tech gaps
        if missing_tech:
            insight_parts.append(
                f"Tech Gap: Adding '{', '.join(missing_tech)}' to your toolkit "
                f"would make you more competitive."
            )
        
        # Seniority note
        job_zone = target_seniority.get("job_zone", 3)
        if int(job_zone) >= 4:
            insight_parts.append(
                f"Seniority: This role typically requires {target_seniority.get('experience_required', '4+ years')} "
                f"of experience with {target_seniority.get('typical_education', 'a Bachelor\'s degree')}."
            )
        
        # Effort summary
        insight_parts.append(
            f"This pivot would require {effort_desc}."
        )
        
        return " ".join(insight_parts)
    
    def _generate_pivot_summary(
        self,
        current_title: str,
        pivots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate summary of all pivot opportunities
        """
        if not pivots:
            return {
                "headline": "No pivot opportunities found",
                "recommendations": []
            }
        
        # Find best match
        best_match = pivots[0] if pivots else None
        
        # Categorize pivots
        easy_pivots = [p for p in pivots if p["qualification_match"] >= 70]
        moderate_pivots = [p for p in pivots if 40 <= p["qualification_match"] < 70]
        stretch_pivots = [p for p in pivots if p["qualification_match"] < 40]
        
        recommendations = []
        
        if easy_pivots:
            recommendations.append({
                "category": "Easy Transitions",
                "description": "These roles closely match your current skills",
                "roles": [p["title"] for p in easy_pivots[:3]]
            })
        
        if moderate_pivots:
            recommendations.append({
                "category": "Growth Opportunities",
                "description": "These roles require some upskilling but are achievable",
                "roles": [p["title"] for p in moderate_pivots[:3]]
            })
        
        if stretch_pivots:
            recommendations.append({
                "category": "Stretch Goals",
                "description": "These roles represent significant career changes",
                "roles": [p["title"] for p in stretch_pivots[:2]]
            })
        
        return {
            "headline": f"Found {len(pivots)} career pivot opportunities for {current_title}",
            "best_match": {
                "title": best_match["title"] if best_match else None,
                "match": best_match["qualification_match"] if best_match else 0
            },
            "easy_count": len(easy_pivots),
            "moderate_count": len(moderate_pivots),
            "stretch_count": len(stretch_pivots),
            "recommendations": recommendations
        }


def format_career_insights_for_frontend(insights: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format career progression data for frontend display
    """
    workforce = insights.get("workforce", {})
    
    # Build headline stats
    headline = f"{insights['job_title']} in {insights['location']}"
    
    stats = []
    if workforce.get("average_wage"):
        wage = workforce["average_wage"]
        stats.append(f"${wage:,}/year average")
    
    if workforce.get("yoy_growth_wage"):
        growth = workforce["yoy_growth_wage"]
        direction = "📈" if growth > 0 else "📉"
        stats.append(f"{direction} {abs(growth):.1f}% YoY")
    
    if workforce.get("workforce_size"):
        size = workforce["workforce_size"]
        stats.append(f"{size:,} workers")
    
    return {
        "headline": headline,
        "stats": " • ".join(stats) if stats else "Data unavailable",
        
        "seniority": insights.get("seniority", {}),
        
        "career_ladder": insights.get("career_ladder", []),
        
        "education_progression": [
            {
                "level": edu["education"],
                "salary": f"${edu['average_wage']:,}",
                "increase": (
                    f"+${edu['increase_from_previous']['amount']:,} "
                    f"({edu['increase_from_previous']['percentage']:.1f}%)"
                ) if "increase_from_previous" in edu else "Base level"
            }
            for edu in insights.get("education_progression", [])
        ],
        
        "skill_impact": insights.get("skill_impact", {}),
        
        "top_skills": [s["name"] for s in insights.get("core_skills", [])[:10]],
        
        "data_freshness": insights.get("last_updated", "")
    }


def main():
    """Test the career progression enricher"""
    
    print("=" * 80)
    print("CAREER PROGRESSION ENRICHER - TEST")
    print("=" * 80)
    
    # Test with Software Developer role
    enricher = CareerProgressionEnricher()
    
    # First, search O*NET for the occupation code
    job_title = "Software Engineer"
    print(f"\n🔍 Searching O*NET for: {job_title}")
    
    try:
        search_url = f"{ONET_API_BASE}/online/search"
        response = requests.get(
            search_url,
            headers={"Authorization": f"Basic {ONET_API_AUTH}", "Accept": "application/json"},
            params={"keyword": job_title, "end": 1}
        )
        response.raise_for_status()
        search_data = response.json()
        
        if "occupation" in search_data and search_data["occupation"]:
            onet_code = search_data["occupation"][0]["code"]
            print(f"✅ Found O*NET code: {onet_code}")
        else:
            onet_code = "15-1252.00"  # Fallback: Software Developers
            print(f"⚠️  Using fallback O*NET code: {onet_code}")
    except Exception as e:
        print(f"❌ Search failed: {e}")
        onet_code = "15-1252.00"
    
    # Get full career insights
    print(f"\n📊 Generating career progression insights...")
    insights = enricher.get_full_career_insights(
        job_title=job_title,
        onet_code=onet_code,
        location="United States",
        city_geo_id=None  # Use national data (could use "16000US0644000" for LA)
    )
    
    # Format for frontend
    frontend_data = format_career_insights_for_frontend(insights)
    
    # Display results
    print("\n" + "=" * 80)
    print("📊 CAREER PROGRESSION INSIGHTS")
    print("=" * 80)
    
    print(f"\n{frontend_data['headline']}")
    print(f"{frontend_data['stats']}\n")
    
    print("🎯 Seniority Level:")
    seniority = frontend_data["seniority"]
    print(f"   Level: {seniority.get('level', 'Unknown')}")
    print(f"   Experience: {seniority.get('experience_required', 'Unknown')}")
    print(f"   Education: {seniority.get('typical_education', 'Unknown')}")
    
    print("\n🪜 Career Ladder:")
    for step in frontend_data["career_ladder"]:
        print(f"\n   {step['level']} - {step['title']}")
        print(f"   └─ Experience: {step['experience']}")
        print(f"   └─ Education: {step['education']}")
        print(f"   └─ Focus: {step['key_focus']}")
        print(f"   └─ Salary: {step['salary_range']}")
    
    if frontend_data["education_progression"]:
        print("\n🎓 Education Impact on Salary:")
        for edu in frontend_data["education_progression"]:
            print(f"   {edu['level']}: {edu['salary']} ({edu['increase']})")
    
    print("\n🔑 Top Skills for This Role:")
    for i, skill in enumerate(frontend_data["top_skills"][:5], 1):
        print(f"   {i}. {skill}")
    
    skill_impact = frontend_data["skill_impact"]
    if skill_impact:
        print(f"\n💡 Skill Impact Analysis:")
        wage = skill_impact.get('current_role_wage', 0) or 0
        print(f"   Current Role Avg: ${wage:,}")
        print(f"   High-Value Skills: {', '.join(skill_impact.get('high_value_skills', []))}")
        print(f"   Recommendation: {skill_impact.get('recommendation', '')}")
    
    print("\n" + "=" * 80)
    print("✅ Career Progression Test Complete!")
    print("=" * 80)
    
    # Save full JSON output
    output_file = "/home/skystarved/Render_Dockers/career_progression_output.json"
    with open(output_file, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"\n💾 Full output saved to: {output_file}")
    
    # =========================================================================
    # TEST CAREER PIVOT ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("🔄 CAREER PIVOT ANALYSIS - TEST")
    print("=" * 80)
    
    # Simulate resume skills (what the user has)
    sample_resume_skills = [
        "Python", "JavaScript", "SQL", "Git",
        "Problem Solving", "Critical Thinking", "Communication",
        "Data Analysis", "API Development", "Agile",
        "Project Management", "Team Collaboration"
    ]
    
    sample_resume_tech = [
        "Python", "JavaScript", "React", "Node.js",
        "PostgreSQL", "MongoDB", "Docker", "AWS",
        "Git", "Jira", "VS Code"
    ]
    
    print(f"\n📄 Sample Resume Skills: {', '.join(sample_resume_skills[:6])}...")
    print(f"🛠️  Sample Resume Tech: {', '.join(sample_resume_tech[:6])}...")
    
    # Generate pivot analysis
    print(f"\n🔄 Generating pivot analysis for: {job_title}")
    pivot_analysis = enricher.generate_career_pivot_analysis(
        current_onet_code=onet_code,
        current_job_title=job_title,
        resume_skills=sample_resume_skills,
        resume_tech=sample_resume_tech,
        current_wage=None,  # Would come from Data USA or user input
        location="United States",
        max_pivots=5
    )
    
    # Display pivot analysis results
    print("\n" + "-" * 80)
    print("📊 PIVOT ANALYSIS RESULTS")
    print("-" * 80)
    
    summary = pivot_analysis.get("summary", {})
    print(f"\n{summary.get('headline', 'No opportunities found')}")
    
    if summary.get("best_match"):
        best = summary["best_match"]
        print(f"\n🏆 Best Match: {best.get('title', 'N/A')} ({best.get('match', 0):.0f}% qualification)")
    
    print(f"\n📈 Opportunity Breakdown:")
    print(f"   Easy Transitions: {summary.get('easy_count', 0)}")
    print(f"   Growth Opportunities: {summary.get('moderate_count', 0)}")
    print(f"   Stretch Goals: {summary.get('stretch_count', 0)}")
    
    # Show top 3 pivot opportunities
    pivots = pivot_analysis.get("pivot_opportunities", [])[:3]
    if pivots:
        print(f"\n🔄 Top 3 Career Pivots:")
        for i, pivot in enumerate(pivots, 1):
            print(f"\n   [{i}] {pivot['title']}")
            print(f"       Qualification Match: {pivot['qualification_match']:.0f}%")
            print(f"       Seniority: {pivot['seniority'].get('level', 'N/A')} ({pivot['seniority'].get('experience_required', 'N/A')})")
            
            missing_skills = pivot.get("skill_gap", {}).get("missing", [])
            if missing_skills:
                skill_names = [s["name"] for s in missing_skills[:3]]
                print(f"       Skill Gaps: {', '.join(skill_names)}")
            
            missing_tech = pivot.get("tech_gap", {}).get("missing", [])
            if missing_tech:
                print(f"       Tech Gaps: {', '.join(missing_tech[:3])}")
            
            print(f"\n       💡 Insight: {pivot.get('insight', 'N/A')[:200]}...")
    
    # Show recommendations
    recommendations = summary.get("recommendations", [])
    if recommendations:
        print(f"\n📋 Recommendations by Category:")
        for rec in recommendations:
            print(f"\n   {rec['category']}:")
            print(f"   {rec['description']}")
            print(f"   Roles: {', '.join(rec['roles'])}")
    
    print("\n" + "=" * 80)
    print("✅ Career Pivot Analysis Complete!")
    print("=" * 80)
    
    # Save pivot analysis JSON
    pivot_output_file = "/home/skystarved/Render_Dockers/career_pivot_analysis_output.json"
    with open(pivot_output_file, "w") as f:
        json.dump(pivot_analysis, f, indent=2)
    print(f"\n💾 Pivot analysis saved to: {pivot_output_file}")


if __name__ == "__main__":
    main()
