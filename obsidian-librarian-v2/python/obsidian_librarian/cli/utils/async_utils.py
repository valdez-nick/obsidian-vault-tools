"""Async utilities for CLI commands."""

import asyncio
import sys
from typing import Callable, Any, Awaitable
from functools import wraps

from .console import print_error
from .logging import get_logger

logger = get_logger(__name__)


def run_async(coro_func: Callable[..., Awaitable[Any]]):
    """Decorator to run async functions in CLI commands."""
    @wraps(coro_func)
    def wrapper(*args, **kwargs):
        try:
            return asyncio.run(coro_func(*args, **kwargs))
        except KeyboardInterrupt:
            print_error("Operation cancelled by user")
            sys.exit(1)
        except Exception as e:
            print_error(f"Operation failed: {e}")
            logger.error("CLI operation failed", error=str(e), exc_info=True)
            sys.exit(1)
    
    return wrapper


async def gather_with_concurrency(tasks, max_concurrency: int = 10):
    """Execute tasks with limited concurrency."""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def bounded_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*[bounded_task(task) for task in tasks])


async def run_with_timeout(coro, timeout_seconds: int):
    """Run a coroutine with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")


def handle_async_errors(func):
    """Decorator to handle common async errors in CLI commands."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except FileNotFoundError as e:
            print_error(f"File not found: {e}")
            logger.error("File not found", error=str(e))
            raise
        except PermissionError as e:
            print_error(f"Permission denied: {e}")
            logger.error("Permission denied", error=str(e))
            raise
        except ValueError as e:
            print_error(f"Invalid value: {e}")
            logger.error("Invalid value", error=str(e))
            raise
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            logger.error("Unexpected error", error=str(e), exc_info=True)
            raise
    
    return wrapper