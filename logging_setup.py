import logging
import os
import sys
try:
    from pythonjsonlogger import jsonlogger
    _HAS_JSONLOGGER = True
except Exception:
    _HAS_JSONLOGGER = False
from starlette.middleware.base import BaseHTTPMiddleware
import uuid


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id
        # Caller-controlled IDs may contain private text. Keep the existing
        # response correlation contract but use a server ID for analyze logs.
        if request.url.path == "/analyze":
            request.state.log_request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response


class ServiceContextFilter(logging.Filter):
    def __init__(self, service_name: str, env: str):
        super().__init__()
        self.service_name = service_name
        self.env = env

    def filter(self, record):
        # Attach standard context fields to every record
        record.service = self.service_name
        record.env = self.env
        # Ensure request_id is present (may be added via extra)
        if not hasattr(record, "request_id"):
            record.request_id = None
        return True


def configure_logging(service_name: str = "imaginator"):
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()

    # HTTPX logs complete request URLs at INFO, including query values derived
    # from user input. Network debug logs can also contain URLs and headers.
    # Do not rewrite root records: unrelated application logs must be unchanged.
    # WARNING and above are not sanitized by this configuration.
    for name in ("httpx", "httpcore", "urllib3", "aiohttp.client"):
        logging.getLogger(name).setLevel(logging.WARNING)

    root = logging.getLogger()
    # Remove default handlers to avoid duplicate logging
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json" and _HAS_JSONLOGGER:
        fmt = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s service=%(service)s env=%(env)s request_id=%(request_id)s')
    else:
        # Fallback to text formatter if json logger isn't available
        # Use safer format that doesn't crash if service/env are missing
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler.setFormatter(fmt)
    root.setLevel(getattr(logging, log_level, logging.INFO))
    root.addHandler(handler)

    # Add service context filter so all logs contain service and env
    root.addFilter(ServiceContextFilter(service_name, os.getenv("ENVIRONMENT", "dev")))

    # Return a configured logger instance for convenience
    return logging.getLogger(service_name)
