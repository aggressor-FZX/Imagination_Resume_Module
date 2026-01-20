# Target Company Guard Implementation

## Date: January 19, 2026

## Problem Statement
The LLM was experiencing "context bleed" where it confused:
- **"Tailor resume FOR Armada"** (the target company in the job ad)
- **"Worked AT Armada"** (falsely listing them as a past employer)

This caused hallucinated employment history where the target company appeared as an employer in the resume header, even though the user never worked there.

## Solution: Multi-Layer Defense

### 1. **System Prompt Update** (Rule #6)
Added explicit anti-hallucination rule to `STAR_EDITOR_SYSTEM_PROMPT`:

```
6. **ANTI-HALLUCINATION:** NEVER list the Target Company (the company from the Job Ad) as an employer. 
   You must ONLY use the company names provided in the user's input data.
```

### 2. **User Prompt Enhancement**
Modified `_create_user_prompt()` to include:
- New parameter: `target_company: str = ""`
- Explicit warning when target company is present:
  ```
  CRITICAL: The user is applying to '{target_company}'. 
  DO NOT list '{target_company}' as their employer in the headers 
  unless it explicitly appears in the experience list below.
  ```

### 3. **Code-Level Guard**
Enhanced `_apply_hallucination_guard()` with:

#### New Parameter
- `target_company: str = ""` - The target company name to guard against

#### Target Company Detection Logic
```python
# 2. TARGET COMPANY GUARD (The "Armada" Fix)
if target_company:
    # Check if user actually worked there
    is_legit_employee = any(target_company.lower() in c.lower() 
                           for c in original_companies)
    
    if not is_legit_employee:
        # Regex to find target company in header: "| **Armada**" or "| Armada"
        pattern = re.compile(rf'\|\s*\**{re.escape(target_company)}\**', 
                           re.IGNORECASE)
        
        if pattern.search(markdown):
            # Replace with actual employer or safe fallback
            replacement = f"| **{original_companies[0] if original_companies 
                                else 'Previous Employer'}**"
            markdown = pattern.sub(replacement, markdown)
            
            result["editorial_notes"] += f" Fixed target company hallucination ({target_company})."
```

### 4. **Data Flow Update**
Modified `polish()` method to:
1. Extract target company from research data: `target_company = research_data.get("company_name", "")`
2. Pass it to prompt creation: `_create_user_prompt(..., target_company)`
3. Pass it to hallucination guard: `_apply_hallucination_guard(..., target_company=target_company)`

## How It Works

### Defense Layer 1: Prompt Engineering
- Tells the LLM explicitly: "You are writing a resume FOR Armada, not FROM Armada"
- Provides clear negative constraint before processing

### Defense Layer 2: Pattern Detection
- Scans the final markdown for patterns like `| **Armada**` in header positions
- Only triggers if "Armada" is NOT in the user's actual work history

### Defense Layer 3: Surgical Replacement
- If hallucination detected, replaces target company with:
  - User's most recent actual employer (from `original_companies[0]`)
  - Or "Previous Employer" as safe fallback
- Logs the fix in `editorial_notes`

## Example Scenario

### Input
- User's actual employer: "Washington State Data Exchange"
- Target company (from job ad): "Armada"

### Without Guard (Bug)
```markdown
## **Senior Software Engineer** | **Armada**
*2020 - Present | Seattle, WA*
```

### With Guard (Fixed)
```markdown
## **Senior Software Engineer** | **Washington State Data Exchange**
*2020 - Present | Seattle, WA*
```

## Files Modified
- `/home/skystarved/Render_Dockers/Imaginator/stages/star_editor.py`
  - Updated `STAR_EDITOR_SYSTEM_PROMPT` (added Rule #6)
  - Modified `polish()` method
  - Enhanced `_create_user_prompt()` signature and logic
  - Upgraded `_apply_hallucination_guard()` with target company detection

## Testing Recommendations
1. Test with job ad for "Armada" and resume without Armada employment
2. Test with job ad for "Armada" and resume WITH Armada employment (should NOT trigger)
3. Verify `editorial_notes` contains fix notification when triggered
4. Check logs for warning: `[STAR_EDITOR] Target company 'Armada' hallucinated as employer. Fixing...`

## Backup
Backup created at: `star_editor.py.backup_YYYYMMDD_HHMMSS`

## Status
✅ Implementation Complete
✅ Syntax Validated
✅ Ready for Testing
