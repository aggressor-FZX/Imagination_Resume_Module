"""
Simple test for Target Company Guard - Tests the regex pattern matching
"""

import re

def test_target_company_pattern():
    """Test the regex pattern used in the guard"""
    
    print("=" * 80)
    print("TARGET COMPANY GUARD - PATTERN MATCHING TEST")
    print("=" * 80)
    print()
    
    # Test cases
    test_cases = [
        {
            "name": "Standard format with bold",
            "markdown": "## **Senior Software Engineer** | **Armada**\n*2020 - Present*",
            "target": "Armada",
            "should_match": True
        },
        {
            "name": "Format without bold on company",
            "markdown": "## **Senior Software Engineer** | Armada\n*2020 - Present*",
            "target": "Armada",
            "should_match": True
        },
        {
            "name": "Case insensitive match",
            "markdown": "## **Senior Software Engineer** | **armada**\n*2020 - Present*",
            "target": "Armada",
            "should_match": True
        },
        {
            "name": "Company in bullet (should NOT match header pattern)",
            "markdown": "## **Engineer** | **Google**\n- Worked with Armada team",
            "target": "Armada",
            "should_match": False
        },
        {
            "name": "Different company (should NOT match)",
            "markdown": "## **Senior Software Engineer** | **Google**\n*2020 - Present*",
            "target": "Armada",
            "should_match": False
        }
    ]
    
    # The pattern from the implementation
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"  Markdown: {test['markdown'][:60]}...")
        print(f"  Target: {test['target']}")
        
        # Create the pattern (same as in star_editor.py)
        pattern = re.compile(rf'\|\s*\**{re.escape(test["target"])}\**', re.IGNORECASE)
        match = pattern.search(test['markdown'])
        
        matched = match is not None
        expected = test['should_match']
        
        if matched == expected:
            print(f"  ✅ PASSED: {'Matched' if matched else 'Did not match'} as expected")
            passed += 1
        else:
            print(f"  ❌ FAILED: {'Matched' if matched else 'Did not match'} but expected {'match' if expected else 'no match'}")
            if match:
                print(f"     Match found: '{match.group()}'")
            failed += 1
        print()
    
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    print()
    
    # Test the replacement logic
    print("REPLACEMENT LOGIC TEST")
    print("-" * 80)
    
    markdown_with_hallucination = "## **Senior Software Engineer** | **Armada**\n*2020 - Present | Seattle, WA*\n- Built systems"
    target_company = "Armada"
    actual_employer = "Washington State Data Exchange"
    
    print(f"Original (hallucinated):")
    print(f"  {markdown_with_hallucination}")
    print()
    
    pattern = re.compile(rf'\|\s*\**{re.escape(target_company)}\**', re.IGNORECASE)
    replacement = f"| **{actual_employer}**"
    fixed_markdown = pattern.sub(replacement, markdown_with_hallucination)
    
    print(f"After fix:")
    print(f"  {fixed_markdown}")
    print()
    
    if "Armada" not in fixed_markdown and actual_employer in fixed_markdown:
        print("✅ PASSED: Hallucination successfully replaced!")
    else:
        print("❌ FAILED: Replacement did not work correctly")
    
    print()

if __name__ == "__main__":
    test_target_company_pattern()
