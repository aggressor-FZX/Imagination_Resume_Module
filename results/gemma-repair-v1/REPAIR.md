# Release trace — `faithful_paraphrase`

- original sha256: `cf1b3c1aeb26c8e4`
- first decision: `approve` (valid: True)
- repair: `not_attempted`
- sections present: `{'Professional Summary': True, 'Professional Experience': True}`
- **released: True** — `initial_llm_approve`
- released document sha256: `cf1b3c1aeb26c8e4`
- receipt verified: ok

## Steps

1. start
2. first_review:approve
3. structure:ok
4. released


# Release trace — `cal_ops_faithful`

- original sha256: `d3a1e4e2defcb461`
- first decision: `approve` (valid: True)
- repair: `not_attempted`
- sections present: `{'Professional Summary': True, 'Professional Experience': True}`
- **released: True** — `initial_llm_approve`
- released document sha256: `d3a1e4e2defcb461`
- receipt verified: ok

## Steps

1. start
2. first_review:approve
3. structure:ok
4. released


# Release trace — `unsupported_method`

- original sha256: `9bc6856fb944165a`
- first decision: `revise` (valid: True)
- repair: `applied`
- proposed sha256: `5cb5f3c418ab62f9`
- re-review decision: `approve` (valid: True)
- sections present: `{'Professional Summary': True, 'Professional Experience': True}`
- **released: True** — `corrected_document_llm_approved`
- released document sha256: `5cb5f3c418ab62f9`
- receipt verified: ok

## Steps

1. start
2. first_review:revise
3. repair:applied
4. re_review:approve
5. structure:ok
6. released

## Repair diff

```diff
--- before
+++ after
@@ -20,3 +20,3 @@
 *Cost Reporting Coordinator | 07/2015 - 03/2019*
-- Cut rework by 18% through standardizing panel labeling and targeted quality audits across the terminal fleet.
+- Cut rework by 18% through standardizing panel labeling across the terminal fleet.
 - Prepared the monthly cost summary that the regional manager reviewed.
```


# Release trace — `wrong_record_attribution`

- original sha256: `ae00ed8322bb2bbb`
- first decision: `revise` (valid: True)
- repair: `applied`
- proposed sha256: `2f3708f19dae73b2`
- re-review decision: `approve` (valid: True)
- sections present: `{'Professional Summary': True, 'Professional Experience': True}`
- **released: True** — `corrected_document_llm_approved`
- released document sha256: `2f3708f19dae73b2`
- receipt verified: ok

## Steps

1. start
2. first_review:revise
3. repair:applied
4. re_review:approve
5. structure:ok
6. released

## Repair diff

```diff
--- before
+++ after
@@ -21,3 +21,3 @@
 - Cut rework by 18% by standardizing panel labeling across the terminal fleet.
-- Prepared the monthly cost summary that the regional manager reviewed and reconciled 1,180 fuel invoices each quarter.
+- Prepared the monthly cost summary reviewed by the regional manager.
 
```


# Release trace — `omitted_record`

- original sha256: `99e392de879f9d07`
- first decision: `revise` (valid: True)
- repair: `applied`
- proposed sha256: `29d407367190f5d5`
- re-review decision: `approve` (valid: True)
- sections present: `{'Professional Summary': True, 'Professional Experience': True}`
- **released: True** — `corrected_document_llm_approved`
- released document sha256: `29d407367190f5d5`
- receipt verified: ok

## Steps

1. start
2. first_review:revise
3. repair:applied
4. re_review:approve
5. structure:ok
6. released

## Repair diff

```diff
--- before
+++ after
@@ -18,2 +18,7 @@
 
+**Ridgeline Equipment Services** -- Fife, WA
+*Inventory Analyst | 08/2023 - Present*
+- Maintain parts availability records for 7 service branches.
+- Compared quarterly parts usage against the prior four quarters to set reorder points.
+
 ## Certifications
```
