#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data caching functionality for tennis match features.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache directory configuration
CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Cache metadata file
CACHE_METADATA_FILE = CACHE_DIR / "cache_metadata.json"

def get_cache_key(query: str, limit: Optional[int] = None) -> str:
    """
    Generate a unique cache key for a database query.
    
    Args:
        query: The SQL query
        limit: Optional row limit
        
    Returns:
        Unique cache key
    """
    # Create a string representation of the query parameters
    params = f"{query}_{limit}"
    
    # Generate a hash of the parameters
    return hashlib.md5(params.encode()).hexdigest()

def get_cache_path(cache_key: str) -> Path:
    """
    Get the path for a cache file.
    
    Args:
        cache_key: Unique cache key
        
    Returns:
        Path to cache file
    """
    return CACHE_DIR / f"{cache_key}.parquet"

def load_cache_metadata() -> Dict[str, Any]:
    """
    Load cache metadata from file.
    
    Returns:
        Dictionary of cache metadata
    """
    if CACHE_METADATA_FILE.exists():
        try:
            with open(CACHE_METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
            return {}
    return {}

def save_cache_metadata(metadata: Dict[str, Any]) -> None:
    """
    Save cache metadata to file.
    
    Args:
        metadata: Dictionary of cache metadata
    """
    try:
        with open(CACHE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache metadata: {e}")

def get_cached_data(cache_key: str) -> Optional[pd.DataFrame]:
    """
    Try to load data from cache.
    
    Args:
        cache_key: Unique cache key
        
    Returns:
        Cached DataFrame if found, None otherwise
    """
    cache_path = get_cache_path(cache_key)
    
    if cache_path.exists():
        try:
            # Load metadata
            metadata = load_cache_metadata()
            cache_info = metadata.get(cache_key, {})
            
            # Check if cache is expired (older than 24 hours)
            if cache_info:
                cache_time = datetime.fromisoformat(cache_info['timestamp'])
                if (datetime.now() - cache_time).total_seconds() > 24 * 3600:
                    logger.info(f"Cache expired for key {cache_key}")
                    return None
            
            # Load cached data
            df = pd.read_parquet(cache_path)
            logger.info(f"Loaded {len(df)} rows from cache")
            return df
            
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
            return None
    
    return None

def save_to_cache(df: pd.DataFrame, cache_key: str) -> None:
    """
    Save data to cache.
    
    Args:
        df: DataFrame to cache
        cache_key: Unique cache key
    """
    try:
        # Save DataFrame to parquet file
        cache_path = get_cache_path(cache_key)
        df.to_parquet(cache_path, index=False)
        
        # Update metadata
        metadata = load_cache_metadata()
        metadata[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'rows': len(df),
            'columns': list(df.columns)
        }
        save_cache_metadata(metadata)
        
        logger.info(f"Cached {len(df)} rows with key {cache_key}")
        
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")

def clear_cache() -> None:
    """
    Clear all cached data.
    """
    try:
        # Remove all parquet files
        for file in CACHE_DIR.glob("*.parquet"):
            file.unlink()
        
        # Clear metadata
        save_cache_metadata({})
        
        logger.info("Cache cleared successfully")
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")

def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the cache.
    
    Returns:
        Dictionary of cache statistics
    """
    try:
        metadata = load_cache_metadata()
        total_rows = sum(info['rows'] for info in metadata.values())
        total_files = len(list(CACHE_DIR.glob("*.parquet")))
        
        return {
            'total_cached_rows': total_rows,
            'total_cached_files': total_files,
            'cache_size_mb': sum(f.stat().st_size for f in CACHE_DIR.glob("*.parquet")) / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {
            'total_cached_rows': 0,
            'total_cached_files': 0,
            'cache_size_mb': 0
        } 