# Login Landing DB SSL Fix Remediation Guide

## Problem
Multiple files in the `Login_Landing_page` repo use insecure SSL configurations for database connections:
`ssl: { rejectUnauthorized: false }`

This allows for potential Man-in-the-Middle (MITM) attacks.

## Solution
Centralize SSL configuration using a helper that enforces certificate validation in production.

### 1. Helper Utility
Create `backend/lib/getDbSslConfig.js`:
```javascript
const { getDbSslConfig } = require('../lib/getDbSslConfig');
const pool = new Pool({ connectionString, ssl: getDbSslConfig() });
```

### 2. Apply to Files
Replace all instances of `ssl: { rejectUnauthorized: false }` with `ssl: getDbSslConfig()`.

### 3. Environment Variables
Set the following in production:
- `NODE_ENV=production`
- `DB_SSL_CA` (Base64 encoded CA certificate if using custom CA)
- `DB_SSL_ENFORCE=true` (Optional override)

## Testing
Use `SECURITY_FIXES/test_db_connection.sh` to verify SSL enforcement.
