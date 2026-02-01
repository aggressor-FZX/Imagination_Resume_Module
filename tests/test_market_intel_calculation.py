from career_progression_enricher import CareerProgressionEnricher


def test_calculate_market_intel_rocket_ship():
    enricher = CareerProgressionEnricher()
    workforce_data = {
        "workforce_delta_3yr": 12.5,
        "yoy_growth_workforce": 6.0,
        "yoy_growth_wage": 8.0,
        "average_wage": 145000,
        "workforce_size": 85000,
        "year_latest": "2023",
    }
    onet_summary = {"bright_outlook": True}

    result = enricher.calculate_market_intel(
        workforce_data=workforce_data,
        onet_summary=onet_summary,
        job_title="Software Engineer",
        location="San Francisco",
    )

    assert result["status"] == "Career Rocket Ship 🚀"
    assert result["demand_label"] == "High Demand"
    assert result["has_bright_outlook"] is True
    assert result["is_shortage"] is False


def test_calculate_market_intel_shortage():
    enricher = CareerProgressionEnricher()
    workforce_data = {
        "workforce_delta_3yr": 2.0,
        "yoy_growth_workforce": 2.0,
        "yoy_growth_wage": 10.0,
        "average_wage": 120000,
        "workforce_size": 50000,
        "year_latest": "2023",
    }
    onet_summary = {}

    result = enricher.calculate_market_intel(
        workforce_data=workforce_data,
        onet_summary=onet_summary,
        job_title="Data Scientist",
        location="Los Angeles",
    )

    assert result["is_shortage"] is True
    assert result["status"] == "Critical Shortage"
    assert result["demand_label"] in {"Stable Market", "High Demand"}
