# Run 12 — capture cleanup record

| Step | Result |
|---|---|
| Capture enabled (`CORRECTION_LIVE_CAPTURE=1`) on `7cc415ba` | deploy `dep-dbl34sp42hec73dnbomg`, live 19:16:26Z |
| In-container verification | `capture=1`, dir `/app/data/live-capture`, `files=0` |
| Preflight probe | `PREFLIGHT_OK` — wrote and deleted `preflight-probe-run12.json`, free 1,003,868,160 bytes |
| Analysis submitted | ordinary clicks only; correlation `83e2eb3f-…` |
| **Raw capture preserved BEFORE cleanup** | `.live-run-2026-11-02/captures/run12/83e2eb3f-….json`, 35,984 bytes, sha256 `b2d4b11e25222764b957af70cee6651b24e9b089cf24cd0764b3c439dbc9e67f` |
| Local hash re-verified before purge | matches |
| Remote purge | `removed`, `0` files |
| Capture disabled (`CORRECTION_LIVE_CAPTURE=0`) on `7cc415ba` | deploy `dep-dbl38l5g1s2s73e548e0` |
| Final in-container state | **`capture=0`, `files=0`** |
| Local evidence after cleanup | sha256 still `b2d4b11e…` |

Two redeploys, both the same revision `7cc415ba`. No code push, no merge, no reviewer change,
no gate change.

The withheld draft is held only in `captures/run12/`, outside git, `chmod 600` via scp of a
`0600` container file. It was never published to the customer surface and no export was
attempted.
