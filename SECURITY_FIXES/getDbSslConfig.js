/**
 * Utility to get DB SSL configuration based on environment.
 * Enforces SSL certificate validation in production.
 */
function getDbSslConfig() {
  const isProduction = process.env.NODE_ENV === 'production' || 
                       process.env.DATABASE_URL?.includes('render.com') ||
                       process.env.DB_SSL_ENFORCE === 'true';

  if (!isProduction) {
    return false;
  }

  const sslConfig = {
    rejectUnauthorized: true,
  };

  if (process.env.DB_SSL_CA) {
    // Check if CA is base64 encoded
    let ca = process.env.DB_SSL_CA;
    if (!ca.includes('-----BEGIN CERTIFICATE-----')) {
      try {
        ca = Buffer.from(ca, 'base64').toString('utf-8');
      } catch (e) {
        console.error('Failed to decode DB_SSL_CA as base64, using raw value');
      }
    }
    sslConfig.ca = ca;
  }

  return sslConfig;
}

module.exports = { getDbSslConfig };
