"""
Rate Limiting Middleware
========================

Production-grade rate limiting using Redis.

Features:
- Per-IP rate limiting
- Configurable limits (requests per time window)
- Different limits for different endpoints
- Redis-backed (distributed, persistent)
- Graceful fallback if Redis unavailable

Limits:
- Query endpoint: 60 requests/minute per IP
- Ingest endpoint: 10 requests/minute per IP
- Other endpoints: 120 requests/minute per IP
"""

import time
from typing import Optional
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from app.core.config import settings


class RateLimiter:
    """
    Redis-backed rate limiter.

    Uses sliding window algorithm for accurate rate limiting.
    Falls back to no limiting if Redis unavailable.
    """

    def __init__(self):
        self.redis_client = None
        self.enabled = False

        # Try to connect to Redis with connection pooling
        if REDIS_AVAILABLE and settings.redis_url:
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=False,  # Work with bytes for performance
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    max_connections=settings.redis_max_connections
                )
                # Test connection
                self.redis_client.ping()
                self.enabled = True
                print("[OK] Rate limiter enabled (Redis)")
            except Exception as e:
                print(f"[!] Rate limiter disabled (Redis unavailable): {e}")
                self.redis_client = None
        else:
            print("[i] Rate limiter disabled (Redis not configured)")

    def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window_seconds: int
    ) -> tuple[bool, Optional[int]]:
        """
        Check if request is within rate limit.

        Args:
            identifier: Unique identifier (e.g., IP address)
            limit: Max requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            (allowed, retry_after_seconds)
            - allowed: True if request is allowed
            - retry_after_seconds: Seconds to wait if not allowed (None if allowed)
        """
        if not self.enabled or not self.redis_client:
            # Rate limiting disabled
            return True, None

        try:
            key = f"ratelimit:{identifier}"
            now = time.time()
            window_start = now - window_seconds

            # Remove old entries (outside current window)
            self.redis_client.zremrangebyscore(key, 0, window_start)

            # Count requests in current window
            request_count = self.redis_client.zcard(key)

            if request_count >= limit:
                # Rate limit exceeded
                # Get oldest request timestamp to calculate retry time
                oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_timestamp = oldest[0][1]
                    retry_after = int(oldest_timestamp + window_seconds - now)
                    return False, max(1, retry_after)
                return False, window_seconds

            # Add current request
            self.redis_client.zadd(key, {str(now): now})

            # Set expiry on key (cleanup)
            self.redis_client.expire(key, window_seconds)

            return True, None

        except Exception as e:
            # On error, allow request (fail open)
            print(f"[!] Rate limit check error: {e}")
            return True, None


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce rate limits on API endpoints.

    Rate limits:
    - /api/query: 60 requests/minute
    - /api/ingest: 10 requests/minute
    - Other endpoints: 120 requests/minute
    """

    def __init__(self, app):
        super().__init__(app)

        # Define rate limits for different endpoints
        self.limits = {
            "/api/query": (60, 60),      # 60 requests per 60 seconds
            "/api/ingest": (10, 60),     # 10 requests per 60 seconds
            "default": (120, 60)         # 120 requests per 60 seconds
        }

    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""

        # Skip rate limiting for health check and docs
        if request.url.path in ["/api/health", "/docs", "/redoc", "/openapi.json", "/"]:
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Get rate limit for this endpoint
        path = request.url.path
        limit, window = self.limits.get(path, self.limits["default"])

        # Create identifier (IP + endpoint)
        identifier = f"{client_ip}:{path}"

        # Check rate limit
        allowed, retry_after = rate_limiter.check_rate_limit(
            identifier, limit, window
        )

        if not allowed:
            # Rate limit exceeded
            return Response(
                content=f'{{"detail":"Rate limit exceeded. Try again in {retry_after} seconds."}}',
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Window": str(window),
                    "Content-Type": "application/json"
                }
            )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Window"] = str(window)

        return response
