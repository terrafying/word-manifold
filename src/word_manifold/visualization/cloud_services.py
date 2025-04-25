"""
Cloud storage services for visualization data.

This module provides cloud storage providers and services for managing
visualization data across different cloud platforms.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
import importlib

logger = logging.getLogger(__name__)

# Optional imports for cloud providers
def import_cloud_modules():
    modules = {}
    try:
        modules['boto3'] = importlib.import_module('boto3')
    except ImportError:
        logger.debug("AWS S3 support not available - boto3 not installed")
        
    try:
        modules['gcs'] = importlib.import_module('google.cloud.storage')
    except ImportError:
        logger.debug("Google Cloud Storage support not available - google-cloud-storage not installed")
        
    try:
        modules['azure'] = importlib.import_module('azure.storage.blob')
    except ImportError:
        logger.debug("Azure Blob Storage support not available - azure-storage-blob not installed")
        
    try:
        modules['huggingface_hub'] = importlib.import_module('huggingface_hub')
    except ImportError:
        logger.debug("HuggingFace support not available - huggingface-hub not installed")
    
    return modules

# Import cloud modules
CLOUD_MODULES = import_cloud_modules()

class CloudStorageProvider:
    """Base class for cloud storage providers."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        self.credentials = credentials or {}
        
    def upload(self, file_path: Path, remote_path: str) -> str:
        """Upload a file to cloud storage.
        
        Args:
            file_path: Local path to file
            remote_path: Remote path/key for storage
            
        Returns:
            str: URL of uploaded file
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
        
    def download(self, remote_path: str, local_path: Path) -> Path:
        """Download a file from cloud storage.
        
        Args:
            remote_path: Remote path/key in storage
            local_path: Local path to save file
            
        Returns:
            Path: Path to downloaded file
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

class S3Provider(CloudStorageProvider):
    """AWS S3 storage provider."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        super().__init__(credentials)
        if 'boto3' not in CLOUD_MODULES:
            raise ImportError("AWS S3 support requires boto3 package")
        
        self.s3 = CLOUD_MODULES['boto3'].client(
            's3',
            aws_access_key_id=credentials.get('aws_access_key_id'),
            aws_secret_access_key=credentials.get('aws_secret_access_key'),
            region_name=credentials.get('region_name', 'us-east-1')
        )
        self.bucket = credentials.get('bucket_name')
        
    def upload(self, file_path: Path, remote_path: str) -> str:
        try:
            self.s3.upload_file(str(file_path), self.bucket, remote_path)
            return f"https://{self.bucket}.s3.amazonaws.com/{remote_path}"
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            raise

class GCSProvider(CloudStorageProvider):
    """Google Cloud Storage provider."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        super().__init__(credentials)
        if 'gcs' not in CLOUD_MODULES:
            raise ImportError("Google Cloud Storage support requires google-cloud-storage package")
            
        self.storage_client = CLOUD_MODULES['gcs'].Client.from_service_account_info(credentials)
        self.bucket = self.storage_client.bucket(credentials.get('bucket_name'))
        
    def upload(self, file_path: Path, remote_path: str) -> str:
        try:
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(str(file_path))
            return f"https://storage.googleapis.com/{self.bucket.name}/{remote_path}"
        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            raise

class AzureProvider(CloudStorageProvider):
    """Azure Blob Storage provider."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        super().__init__(credentials)
        if 'azure' not in CLOUD_MODULES:
            raise ImportError("Azure Blob Storage support requires azure-storage-blob package")
            
        self.blob_service = CLOUD_MODULES['azure'].BlobServiceClient(
            account_url=f"https://{credentials.get('account_name')}.blob.core.windows.net",
            credential=credentials.get('account_key')
        )
        self.container = credentials.get('container_name')
        
    def upload(self, file_path: Path, remote_path: str) -> str:
        try:
            blob_client = self.blob_service.get_blob_client(
                container=self.container,
                blob=remote_path
            )
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data)
            return blob_client.url
        except Exception as e:
            logger.error(f"Failed to upload to Azure: {e}")
            raise

class HuggingFaceProvider(CloudStorageProvider):
    """HuggingFace Spaces provider."""
    
    def __init__(self, credentials: Optional[Dict[str, Any]] = None):
        super().__init__(credentials)
        if 'huggingface_hub' not in CLOUD_MODULES:
            raise ImportError("HuggingFace support requires huggingface-hub package")
            
        self.api = CLOUD_MODULES['huggingface_hub'].HfApi()
        self.token = credentials.get('token')
        self.space_id = credentials.get('space_id')
        
    def upload(self, file_path: Path, remote_path: str) -> str:
        try:
            repo_id = f"{self.space_id}/visualizations"
            url = self.api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=remote_path,
                repo_id=repo_id,
                token=self.token
            )
            return url
        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace: {e}")
            raise

class CloudVisualizationService:
    """Service for managing cloud-based visualizations."""
    
    PROVIDERS = {
        's3': S3Provider,
        'gcs': GCSProvider,
        'azure': AzureProvider,
        'huggingface': HuggingFaceProvider
    }
    
    def __init__(self, provider: str = 'huggingface', credentials: Optional[Dict[str, Any]] = None):
        """Initialize the cloud visualization service.
        
        Args:
            provider: Cloud provider to use ('s3', 'gcs', 'azure', 'huggingface')
            credentials: Provider-specific credentials
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
            
        try:
            self.provider = self.PROVIDERS[provider](credentials)
        except ImportError as e:
            logger.warning(f"Failed to initialize {provider} provider: {e}")
            self.provider = None
            
    def upload_visualization(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
        public: bool = True
    ) -> Dict[str, Any]:
        """Upload a visualization to cloud storage.
        
        Args:
            file_path: Path to visualization file
            metadata: Optional metadata to store with visualization
            public: Whether the visualization should be publicly accessible
            
        Returns:
            Dict containing upload details including URL
        """
        if not self.provider:
            raise RuntimeError("No cloud provider available")
            
        try:
            # Generate remote path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            remote_path = f"visualizations/{timestamp}_{file_path.name}"
            
            # Upload file
            url = self.provider.upload(file_path, remote_path)
            
            # Create response with details
            response = {
                'url': url,
                'timestamp': timestamp,
                'filename': file_path.name,
                'provider': self.provider.__class__.__name__,
                'public': public
            }
            
            if metadata:
                response['metadata'] = metadata
                
            return response
            
        except Exception as e:
            logger.error(f"Failed to upload visualization: {e}")
            raise
            
    def create_sharing_link(
        self,
        visualization_url: str,
        expires_in: Optional[int] = None
    ) -> str:
        """Create a sharing link for a visualization.
        
        Args:
            visualization_url: URL of the visualization
            expires_in: Optional expiration time in seconds
            
        Returns:
            Sharing URL for the visualization
        """
        if not self.provider:
            raise RuntimeError("No cloud provider available")
            
        try:
            # For providers that support temporary URLs
            if hasattr(self.provider, 'create_temporary_url'):
                return self.provider.create_temporary_url(
                    visualization_url,
                    expires_in or 3600  # Default 1 hour
                )
            return visualization_url
            
        except Exception as e:
            logger.error(f"Failed to create sharing link: {e}")
            raise
            
    def get_visualization_info(self, url: str) -> Dict[str, Any]:
        """Get information about a visualization.
        
        Args:
            url: URL of the visualization
            
        Returns:
            Dict containing visualization details
        """
        if not self.provider:
            raise RuntimeError("No cloud provider available")
            
        try:
            # Extract basic info from URL
            info = {
                'url': url,
                'provider': self.provider.__class__.__name__
            }
            
            # Add provider-specific details if available
            if hasattr(self.provider, 'get_object_info'):
                info.update(self.provider.get_object_info(url))
                
            return info
            
        except Exception as e:
            logger.error(f"Failed to get visualization info: {e}")
            raise 