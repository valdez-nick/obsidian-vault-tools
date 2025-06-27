"""
Base Connector Abstract Class

Defines the interface that all data source connectors must implement.
Provides common functionality for authentication, rate limiting,
connection pooling, and error handling.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataSourceConnector(ABC):
    """
    Abstract base class for all data source connectors.
    
    Attributes:
        name: Connector name for identification
        config: Configuration dictionary
        is_connected: Current connection status
        last_error: Last error encountered
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the connector with configuration.
        
        Args:
            name: Unique name for this connector instance
            config: Configuration parameters specific to the connector
        """
        self.name = name
        self.config = config
        self.is_connected = False
        self.last_error: Optional[str] = None
        self._rate_limiter = None
        self._connection_pool = None
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the data source.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate that the connection is active and healthy.
        
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_data(self, query: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract data from the source based on the query.
        
        Args:
            query: Query parameters specific to the data source
            
        Returns:
            DataFrame containing the extracted data
            
        Raises:
            ConnectionError: If not connected
            ValueError: If query is invalid
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the data source.
        
        Returns:
            Dictionary containing source metadata
        """
        pass
    
    def execute_with_retry(self, func, *args, max_retries: int = 3, **kwargs):
        """
        Execute a function with retry logic and exponential backoff.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the function execution
            
        Raises:
            Exception: The last exception if all retries fail
        """
        import time
        import random
        
        last_exception = None
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"[{self.name}] Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"[{self.name}] All {max_retries + 1} attempts failed. "
                        f"Last error: {str(e)}"
                    )
        
        raise last_exception
    
    def apply_rate_limit(self):
        """Apply rate limiting to prevent API throttling."""
        import time
        
        if self._rate_limiter is None:
            return
        
        # Simple token bucket rate limiter
        current_time = time.time()
        
        # Initialize rate limiter state if needed
        if not hasattr(self._rate_limiter, 'last_request_time'):
            self._rate_limiter.last_request_time = 0
            self._rate_limiter.min_interval = 1.0  # Default to 1 request per second
        
        # Calculate time since last request
        time_since_last = current_time - self._rate_limiter.last_request_time
        
        # If not enough time has passed, sleep
        if time_since_last < self._rate_limiter.min_interval:
            sleep_time = self._rate_limiter.min_interval - time_since_last
            logger.debug(f"[{self.name}] Rate limiting: sleeping for {sleep_time:.3f} seconds")
            time.sleep(sleep_time)
        
        # Update last request time
        self._rate_limiter.last_request_time = time.time()
    
    def log_request(self, endpoint: str, params: Dict[str, Any]):
        """Log API requests for debugging and monitoring."""
        logger.info(f"[{self.name}] Request to {endpoint} with params: {params}")
    
    def log_response(self, endpoint: str, status_code: int, response_time: float):
        """Log API responses for debugging and monitoring."""
        logger.info(f"[{self.name}] Response from {endpoint}: {status_code} in {response_time}s")