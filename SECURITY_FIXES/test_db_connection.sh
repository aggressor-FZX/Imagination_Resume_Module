#!/bin/bash
# Test DB SSL connection with CA
# Usage: DATABASE_URL=... DB_SSL_CA=... ./test_db_connection.sh

if [ -z "$DATABASE_URL" ]; then
  echo "Error: DATABASE_URL is not set"
  exit 1
fi

echo "Testing DB Connection..."

# Use a small node script to test the connection
node -e "
const { Pool } = require('pg');
const { getDbSslConfig } = require('../Login_Landing_page/backend/lib/getDbSslConfig');

async function test() {
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: getDbSslConfig()
  });

  try {
    const client = await pool.connect();
    console.log('Successfully connected to the database with SSL config:', JSON.stringify(getDbSslConfig(), null, 2));
    const res = await client.query('SELECT NOW()');
    console.log('Query result:', res.rows[0]);
    client.release();
  } catch (err) {
    console.error('Connection failed:', err.message);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

test();
"
