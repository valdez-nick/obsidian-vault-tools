"""
Enhanced Google Suite Connector with OAuth Authentication.

This module extends the base GoogleConnector to use OAuth 2.0
authentication via the AuthenticationManager instead of service accounts.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
from datetime import datetime
import logging

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

from .google_connector import GoogleConnector
from authentication.auth_manager import AuthenticationManager

logger = logging.getLogger(__name__)


class GoogleConnectorOAuth(GoogleConnector):
    """
    Enhanced Google Workspace connector using OAuth 2.0 authentication.
    
    This connector uses the AuthenticationManager for OAuth flows
    instead of service account authentication.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Google OAuth connector.
        
        Config should include:
        - tenant_id: Identifier for multi-tenant support
        - client_id: Google OAuth client ID
        - client_secret: Google OAuth client secret
        - scopes: List of OAuth scopes needed (optional)
        - default_folder_id: Optional default Drive folder
        """
        super().__init__(config)
        
        # OAuth specific configuration
        self.tenant_id = config.get('tenant_id', 'default')
        self.client_id = config.get('client_id')
        self.client_secret = config.get('client_secret')
        
        # Initialize authentication manager
        self.auth_manager = AuthenticationManager()
        self.oauth_credential = None
        
    def connect(self) -> bool:
        """
        Establish connection to Google APIs using OAuth.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not GOOGLE_API_AVAILABLE:
            logger.error("Google API client library not installed. Run: pip install google-api-python-client")
            return False
            
        try:
            # Get valid OAuth credential
            self.oauth_credential = self.auth_manager.get_valid_credential("google", self.tenant_id)
            
            if not self.oauth_credential:
                logger.info("No valid credential found, initiating OAuth flow...")
                
                # Run async authentication in sync context
                import asyncio
                loop = asyncio.get_event_loop()
                self.oauth_credential = loop.run_until_complete(
                    self.auth_manager.authenticate(
                        provider="google",
                        tenant_id=self.tenant_id,
                        client_id=self.client_id,
                        client_secret=self.client_secret,
                        scopes=self.scopes
                    )
                )
            
            if self.oauth_credential:
                # Build Google API services
                self._build_services()
                logger.info(f"Successfully connected to Google APIs for tenant: {self.tenant_id}")
                return True
            else:
                logger.error("Failed to obtain OAuth credential")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Google APIs: {e}")
            return False
    
    def _build_services(self):
        """Build Google API service clients."""
        if not self.oauth_credential:
            raise ValueError("No OAuth credential available")
        
        # Create authorized HTTP client
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        
        # Convert our credential to Google's format
        google_creds = Credentials(
            token=self.oauth_credential.access_token,
            refresh_token=self.oauth_credential.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self.oauth_credential.client_id,
            client_secret=self.oauth_credential.client_secret,
            scopes=self.oauth_credential.scopes
        )
        
        # Build services
        self.services = {
            'slides': build('slides', 'v1', credentials=google_creds),
            'sheets': build('sheets', 'v4', credentials=google_creds),
            'drive': build('drive', 'v3', credentials=google_creds)
        }
    
    def validate_connection(self) -> bool:
        """
        Validate Google API connection by testing access.
        
        Returns:
            True if connection is valid, False otherwise
        """
        if not self.services:
            return False
            
        try:
            # Test Drive API access
            self.services['drive'].files().list(
                pageSize=1,
                fields="files(id, name)"
            ).execute()
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get Google Workspace metadata with OAuth info.
        
        Returns:
            Dictionary containing service and auth information
        """
        metadata = {
            "available_services": list(self.services.keys()) if self.services else [],
            "auth_type": "OAuth 2.0",
            "tenant_id": self.tenant_id,
            "token_valid": False,
            "token_expires_at": None,
            "authorized_scopes": []
        }
        
        if self.oauth_credential:
            metadata.update({
                "token_valid": not self.oauth_credential.is_expired(),
                "token_expires_at": self.oauth_credential.expires_at.isoformat() if self.oauth_credential.expires_at else None,
                "authorized_scopes": self.oauth_credential.scopes
            })
        
        # Get user info if connected
        if self.services and 'drive' in self.services:
            try:
                about = self.services['drive'].about().get(fields="user").execute()
                metadata["user_info"] = about.get('user', {})
            except Exception as e:
                logger.warning(f"Failed to get user info: {e}")
        
        return metadata
    
    def create_presentation(self, title: str, template_id: Optional[str] = None) -> str:
        """
        Create a new Google Slides presentation.
        
        Args:
            title: Presentation title
            template_id: Optional template presentation ID
            
        Returns:
            Created presentation ID
        """
        if not self.services or 'slides' not in self.services:
            logger.error("Slides service not available")
            return ""
        
        try:
            body = {"title": title}
            
            if template_id:
                # Copy from template
                drive_service = self.services['drive']
                copy = drive_service.files().copy(
                    fileId=template_id,
                    body=body
                ).execute()
                presentation_id = copy.get('id')
            else:
                # Create new presentation
                presentation = self.services['slides'].presentations().create(
                    body=body
                ).execute()
                presentation_id = presentation.get('presentationId')
            
            logger.info(f"Created presentation '{title}' with ID: {presentation_id}")
            return presentation_id
            
        except HttpError as e:
            logger.error(f"Failed to create presentation: {e}")
            return ""
    
    def read_sheet(self, spreadsheet_id: str, range_name: str) -> pd.DataFrame:
        """
        Read data from a Google Sheet.
        
        Args:
            spreadsheet_id: Sheet ID
            range_name: A1 notation range
            
        Returns:
            DataFrame with sheet data
        """
        if not self.services or 'sheets' not in self.services:
            logger.error("Sheets service not available")
            return pd.DataFrame()
        
        try:
            result = self.services['sheets'].spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            
            if not values:
                logger.warning(f"No data found in range {range_name}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(values[1:], columns=values[0])
            logger.info(f"Read {len(df)} rows from sheet {spreadsheet_id}")
            return df
            
        except HttpError as e:
            logger.error(f"Failed to read sheet: {e}")
            return pd.DataFrame()
    
    def write_sheet(self, spreadsheet_id: str, range_name: str, 
                   data: Union[pd.DataFrame, List[List[Any]]]) -> bool:
        """
        Write data to a Google Sheet.
        
        Args:
            spreadsheet_id: Sheet ID
            range_name: A1 notation range
            data: Data to write
            
        Returns:
            True if write successful
        """
        if not self.services or 'sheets' not in self.services:
            logger.error("Sheets service not available")
            return False
        
        try:
            # Convert DataFrame to list if needed
            if isinstance(data, pd.DataFrame):
                values = [data.columns.tolist()] + data.values.tolist()
            else:
                values = data
            
            body = {'values': values}
            
            result = self.services['sheets'].spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()
            
            updated_cells = result.get('updatedCells', 0)
            logger.info(f"Updated {updated_cells} cells in sheet {spreadsheet_id}")
            return True
            
        except HttpError as e:
            logger.error(f"Failed to write to sheet: {e}")
            return False
    
    def extract_data(self, query: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract data from Google services based on query parameters.
        
        Query parameters:
        - service: Which service to query (sheets, drive)
        - resource_id: ID of the resource
        - range: For Sheets, the range to extract
        - query: For Drive, search query
        
        Returns:
            DataFrame containing extracted data
        """
        service = query.get('service', 'sheets')
        
        if service == 'sheets':
            spreadsheet_id = query.get('resource_id')
            range_name = query.get('range', 'A:Z')
            return self.read_sheet(spreadsheet_id, range_name)
            
        elif service == 'drive':
            # Search files in Drive
            if not self.services or 'drive' not in self.services:
                return pd.DataFrame()
            
            try:
                search_query = query.get('query', '')
                page_size = query.get('page_size', 100)
                
                results = self.services['drive'].files().list(
                    q=search_query,
                    pageSize=page_size,
                    fields="files(id, name, mimeType, createdTime, modifiedTime, size)"
                ).execute()
                
                files = results.get('files', [])
                df = pd.DataFrame(files)
                
                # Convert timestamps
                if not df.empty and 'createdTime' in df.columns:
                    df['createdTime'] = pd.to_datetime(df['createdTime'])
                    df['modifiedTime'] = pd.to_datetime(df['modifiedTime'])
                
                return df
                
            except HttpError as e:
                logger.error(f"Failed to search Drive: {e}")
                return pd.DataFrame()
        
        else:
            logger.error(f"Unsupported service: {service}")
            return pd.DataFrame()
    
    def revoke_access(self) -> bool:
        """
        Revoke OAuth access for this connector.
        
        Returns:
            True if revoked successfully
        """
        if not self.oauth_credential:
            logger.warning("No credential to revoke")
            return False
        
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self.auth_manager.revoke_token(self.oauth_credential)
            )
            
            if result:
                self.oauth_credential = None
                self.services = {}
                logger.info(f"Revoked OAuth access for tenant: {self.tenant_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to revoke access: {e}")
            return False