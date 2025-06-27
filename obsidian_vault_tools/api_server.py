#!/usr/bin/env python3
"""
FastAPI Server for PM Automation Suite External Integrations

Provides REST API endpoints for external tools to integrate with
the PM Automation Suite.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import asyncio

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from obsidian_vault_tools.utils import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class VaultInfo(BaseModel):
    """Vault information model."""
    path: str
    name: str
    total_files: int
    last_modified: str


class WBRRequest(BaseModel):
    """WBR generation request model."""
    project: str = Field(default="ALL", description="Jira project key")
    format: str = Field(default="slides", description="Output format")
    email_recipients: Optional[List[str]] = Field(default=None, description="Email recipients")


class FeaturePipelineRequest(BaseModel):
    """Feature pipeline request model."""
    prd_content: str = Field(..., description="PRD content or file path")
    project: str = Field(..., description="Jira project key")
    epic_key: Optional[str] = Field(default=None, description="Epic to link stories to")
    auto_create: bool = Field(default=False, description="Auto-create stories in Jira")


class QualityAnalysisResponse(BaseModel):
    """Quality analysis response model."""
    overall_score: float
    total_files: int
    issues_count: int
    naming_issues: Dict[str, List[str]]
    suggestions: List[str]


class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None


if FASTAPI_AVAILABLE:
    # Initialize FastAPI app
    app = FastAPI(
        title="PM Automation Suite API",
        description="REST API for PM Automation Suite integrations",
        version="2.3.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure based on security requirements
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global vault path
    vault_path: Optional[Path] = None

    def get_vault_path() -> Path:
        """Get the configured vault path."""
        global vault_path
        if vault_path is None:
            config = Config()
            vault_path = config.get_vault_path()
            if not vault_path:
                raise HTTPException(
                    status_code=400, 
                    detail="No vault path configured. Set via ovt config set-vault"
                )
        return vault_path

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """API root endpoint with documentation."""
        return """
        <html>
            <head><title>PM Automation Suite API</title></head>
            <body>
                <h1>PM Automation Suite API</h1>
                <p>REST API for PM Automation Suite integrations</p>
                <ul>
                    <li><a href="/docs">API Documentation (Swagger)</a></li>
                    <li><a href="/redoc">API Documentation (ReDoc)</a></li>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/vault/info">Vault Information</a></li>
                </ul>
            </body>
        </html>
        """

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            vault = get_vault_path()
            vault_accessible = vault.exists()
        except Exception:
            vault_accessible = False
        
        return APIResponse(
            success=True,
            message="PM Automation Suite API is running",
            data={
                "status": "healthy",
                "vault_accessible": vault_accessible,
                "features_available": {
                    "pm_automation": True,  # Since API is running
                    "content_quality": True,
                    "wbr_automation": True,
                    "feature_pipeline": True,
                    "analytics": True
                }
            }
        )

    @app.get("/vault/info", response_model=VaultInfo)
    async def get_vault_info(vault: Path = Depends(get_vault_path)):
        """Get information about the configured vault."""
        if not vault.exists():
            raise HTTPException(status_code=404, detail="Vault not found")
        
        # Count markdown files
        md_files = list(vault.rglob("*.md"))
        
        # Get last modified time
        try:
            last_modified = max(f.stat().st_mtime for f in md_files)
            import datetime
            last_modified_str = datetime.datetime.fromtimestamp(last_modified).isoformat()
        except (ValueError, OSError):
            last_modified_str = "unknown"
        
        return VaultInfo(
            path=str(vault),
            name=vault.name,
            total_files=len(md_files),
            last_modified=last_modified_str
        )

    @app.post("/wbr/generate")
    async def generate_wbr(
        request: WBRRequest,
        background_tasks: BackgroundTasks,
        vault: Path = Depends(get_vault_path)
    ):
        """Generate a Weekly Business Review."""
        try:
            from pm_automation_suite.wbr import WBRWorkflow
            
            async def run_wbr():
                workflow = WBRWorkflow({
                    'vault_path': str(vault),
                    'jira_project': request.project
                })
                
                result = await workflow.run_complete_workflow()
                
                # If email recipients specified, send the report
                if request.email_recipients:
                    # TODO: Implement email sending
                    logger.info(f"Would send WBR to: {request.email_recipients}")
                
                return result
            
            # Run in background for long-running operations
            if request.format == "slides":
                background_tasks.add_task(run_wbr)
                return APIResponse(
                    success=True,
                    message="WBR generation started in background",
                    data={"status": "processing", "project": request.project}
                )
            else:
                # Run synchronously for quick operations
                result = await run_wbr()
                return APIResponse(
                    success=True,
                    message="WBR generated successfully",
                    data=result
                )
                
        except ImportError:
            raise HTTPException(
                status_code=501, 
                detail="PM Automation Suite not available. Install with pm-automation extras."
            )
        except Exception as e:
            logger.error(f"WBR generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/features/pipeline")
    async def run_feature_pipeline(
        request: FeaturePipelineRequest,
        vault: Path = Depends(get_vault_path)
    ):
        """Run the feature development pipeline."""
        try:
            from pm_automation_suite.feature_pipeline import FeaturePipeline
            
            pipeline = FeaturePipeline({
                'vault_path': str(vault),
                'jira_project': request.project
            })
            
            # Generate stories from PRD content
            stories = await pipeline.generate_stories_from_content(
                request.prd_content, 
                request.project
            )
            
            created_stories = []
            if request.auto_create:
                # Create stories in Jira
                created_stories = await pipeline.bulk_create_stories(
                    stories, 
                    request.epic_key
                )
            
            return APIResponse(
                success=True,
                message=f"Generated {len(stories)} stories" + 
                       (f", created {len(created_stories)} in Jira" if created_stories else ""),
                data={
                    "stories_generated": len(stories),
                    "stories_created": len(created_stories),
                    "stories": stories[:5],  # Return first 5 for preview
                    "project": request.project,
                    "epic_key": request.epic_key
                }
            )
            
        except ImportError:
            raise HTTPException(
                status_code=501, 
                detail="PM Automation Suite not available"
            )
        except Exception as e:
            logger.error(f"Feature pipeline error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/analytics/quality", response_model=QualityAnalysisResponse)
    async def analyze_content_quality(vault: Path = Depends(get_vault_path)):
        """Analyze content quality in the vault."""
        try:
            from obsidian_vault_tools.pm_tools import ContentQualityEngine
            
            engine = ContentQualityEngine(str(vault))
            report = engine.analyze_vault()
            
            return QualityAnalysisResponse(
                overall_score=report.overall_score,
                total_files=report.total_files,
                issues_count=len(report.issues_found),
                naming_issues=report.naming_inconsistencies,
                suggestions=report.standardization_suggestions[:5]  # Top 5 suggestions
            )
            
        except Exception as e:
            logger.error(f"Quality analysis error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/analytics/dashboard/{dashboard_type}")
    async def get_analytics_dashboard(
        dashboard_type: str,
        vault: Path = Depends(get_vault_path)
    ):
        """Get analytics dashboard HTML."""
        if dashboard_type not in ['team', 'executive', 'product']:
            raise HTTPException(
                status_code=400, 
                detail="Invalid dashboard type. Use: team, executive, or product"
            )
        
        try:
            from pm_automation_suite.analytics_hub import DashboardGenerator
            
            generator = DashboardGenerator({
                'vault_path': str(vault)
            })
            
            dashboard_html = await generator.create_pm_dashboard(
                dashboard_type=dashboard_type
            )
            
            return HTMLResponse(content=dashboard_html)
            
        except ImportError:
            raise HTTPException(
                status_code=501, 
                detail="Analytics dashboard not available"
            )
        except Exception as e:
            logger.error(f"Dashboard generation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/monitoring/start")
    async def start_monitoring(
        metrics: Optional[List[str]] = None,
        vault: Path = Depends(get_vault_path)
    ):
        """Start real-time monitoring."""
        try:
            from pm_automation_suite.analytics_hub import MonitoringSystem
            
            monitoring = MonitoringSystem({
                'vault_path': str(vault)
            })
            
            if not metrics:
                await monitoring.create_default_pm_monitoring()
            else:
                for metric in metrics:
                    await monitoring.add_metric_monitor(metric)
            
            # Start monitoring in background
            # Note: In production, this should use a proper task queue
            asyncio.create_task(monitoring.start_monitoring())
            
            return APIResponse(
                success=True,
                message="Monitoring started",
                data={
                    "monitoring_metrics": metrics or "all_default",
                    "status": "active"
                }
            )
            
        except ImportError:
            raise HTTPException(
                status_code=501, 
                detail="Monitoring system not available"
            )
        except Exception as e:
            logger.error(f"Monitoring start error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/config")
    async def get_config():
        """Get current configuration."""
        try:
            vault = get_vault_path()
            config = Config()
            
            return APIResponse(
                success=True,
                message="Configuration retrieved",
                data={
                    "vault_path": str(vault),
                    "vault_exists": vault.exists(),
                    "api_version": "2.3.0"
                }
            )
        except Exception as e:
            return APIResponse(
                success=False,
                message="Configuration error",
                errors=[str(e)]
            )

else:
    # Fallback when FastAPI is not available
    class MockApp:
        def __init__(self):
            logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")
        
        def run(self):
            print("❌ FastAPI not available. Install with:")
            print("pip install fastapi uvicorn")
    
    app = MockApp()


def run_server(host: str = "127.0.0.1", port: int = 8000, debug: bool = False):
    """Run the FastAPI server."""
    if not FASTAPI_AVAILABLE:
        print("❌ FastAPI not available. Install with:")
        print("pip install fastapi uvicorn")
        return
    
    logger.info(f"Starting PM Automation API server on {host}:{port}")
    
    try:
        uvicorn.run(
            "obsidian_vault_tools.api_server:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info" if not debug else "debug"
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PM Automation Suite API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    run_server(args.host, args.port, args.debug)