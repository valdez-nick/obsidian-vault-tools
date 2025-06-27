"""
Feature Development Pipeline Implementation

Orchestrates the complete feature development workflow from PRD analysis
to Jira story creation with monitoring, error handling, and reporting.
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

from .prd_parser import PRDParser, PRDContent
from .story_generator import StoryGenerator, UserStory
from .jira_bulk_creator import JiraBulkCreator, StoryHierarchy

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages."""
    INITIALIZATION = "initialization"
    PRD_PARSING = "prd_parsing"
    STORY_GENERATION = "story_generation"
    JIRA_CREATION = "jira_creation"
    VALIDATION = "validation"
    COMPLETION = "completion"


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution."""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    prd_requirements_count: int = 0
    stories_generated: int = 0
    stories_created_in_jira: int = 0
    validation_errors: int = 0
    stage_durations: Dict[str, float] = field(default_factory=dict)
    
    def calculate_duration(self):
        """Calculate total pipeline duration."""
        if self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "prd_requirements_count": self.prd_requirements_count,
            "stories_generated": self.stories_generated,
            "stories_created_in_jira": self.stories_created_in_jira,
            "validation_errors": self.validation_errors,
            "stage_durations": self.stage_durations
        }


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    status: PipelineStatus
    prd_content: Optional[PRDContent] = None
    generated_stories: List[UserStory] = field(default_factory=list)
    jira_hierarchy: Optional[StoryHierarchy] = None
    metrics: Optional[PipelineMetrics] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> file_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "prd_summary": {
                "title": self.prd_content.metadata.title if self.prd_content else None,
                "requirements_count": len(self.prd_content.requirements) if self.prd_content else 0,
                "validation_errors": len(self.prd_content.validation_errors) if self.prd_content else 0
            } if self.prd_content else None,
            "stories_generated": len(self.generated_stories),
            "jira_summary": self.jira_hierarchy.creation_summary if self.jira_hierarchy else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "artifacts": self.artifacts
        }


class FeaturePipeline:
    """
    Feature Development Pipeline orchestrator.
    
    Manages the complete workflow from PRD analysis to Jira story creation
    with comprehensive monitoring, error handling, and artifact generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Feature Pipeline.
        
        Args:
            config: Configuration for all pipeline components
        """
        self.config = config
        self.pipeline_id = config.get('pipeline_id', f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir = Path(config.get('output_dir', './feature_pipeline_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.prd_parser = PRDParser(config.get('prd_parser', {}))
        self.story_generator = StoryGenerator(config.get('story_generator', {}))
        self.jira_creator = JiraBulkCreator(config.get('jira_bulk_creator', {}))
        
        # Pipeline settings
        self.validate_at_each_stage = config.get('validate_at_each_stage', True)
        self.save_artifacts = config.get('save_artifacts', True)
        self.continue_on_warnings = config.get('continue_on_warnings', True)
        self.max_retries = config.get('max_retries', 2)
        
        # Progress callbacks
        self.progress_callbacks: List[Callable] = []
        
        # Current execution state
        self.current_stage = PipelineStage.INITIALIZATION
        self.current_status = PipelineStatus.PENDING
        self.execution_log = []
    
    def add_progress_callback(self, callback: Callable[[PipelineStage, Dict[str, Any]], None]):
        """Add progress callback for monitoring."""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, stage: PipelineStage, data: Dict[str, Any]):
        """Notify all progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(stage, data)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    async def execute_pipeline(self, prd_file_path: str) -> PipelineResult:
        """
        Execute the complete feature development pipeline.
        
        Args:
            prd_file_path: Path to PRD file
            
        Returns:
            Pipeline execution result
        """
        logger.info(f"Starting feature pipeline: {self.pipeline_id}")
        
        # Initialize result and metrics
        result = PipelineResult(status=PipelineStatus.RUNNING)
        metrics = PipelineMetrics(start_time=datetime.now())
        
        try:
            self.current_status = PipelineStatus.RUNNING
            
            # Stage 1: Parse PRD
            await self._execute_stage(
                PipelineStage.PRD_PARSING,
                self._parse_prd_stage,
                prd_file_path,
                result,
                metrics
            )
            
            # Stage 2: Generate Stories
            await self._execute_stage(
                PipelineStage.STORY_GENERATION,
                self._generate_stories_stage,
                result.prd_content,
                result,
                metrics
            )
            
            # Stage 3: Create Jira Issues
            await self._execute_stage(
                PipelineStage.JIRA_CREATION,
                self._create_jira_issues_stage,
                result.generated_stories,
                result,
                metrics
            )
            
            # Stage 4: Validation
            await self._execute_stage(
                PipelineStage.VALIDATION,
                self._validation_stage,
                result,
                result,
                metrics
            )
            
            # Completion
            self.current_stage = PipelineStage.COMPLETION
            result.status = PipelineStatus.COMPLETED
            self.current_status = PipelineStatus.COMPLETED
            
            metrics.end_time = datetime.now()
            metrics.calculate_duration()
            result.metrics = metrics
            
            logger.info(f"Pipeline completed successfully in {metrics.duration_seconds:.2f}s")
            
            # Save final artifacts
            if self.save_artifacts:
                await self._save_pipeline_artifacts(result)
            
            self._notify_progress(PipelineStage.COMPLETION, {
                "status": "completed",
                "metrics": metrics.to_dict()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result.status = PipelineStatus.FAILED
            result.errors.append(str(e))
            self.current_status = PipelineStatus.FAILED
            
            metrics.end_time = datetime.now()
            metrics.calculate_duration()
            result.metrics = metrics
            
            if self.save_artifacts:
                await self._save_pipeline_artifacts(result)
            
            return result
    
    async def _execute_stage(
        self,
        stage: PipelineStage,
        stage_func: Callable,
        stage_input: Any,
        result: PipelineResult,
        metrics: PipelineMetrics
    ):
        """Execute a pipeline stage with error handling and metrics."""
        self.current_stage = stage
        stage_start = datetime.now()
        
        logger.info(f"Executing stage: {stage.value}")
        self._notify_progress(stage, {"status": "started"})
        
        try:
            # Execute stage with retries
            for attempt in range(self.max_retries + 1):
                try:
                    await stage_func(stage_input, result, metrics)
                    break
                except Exception as e:
                    if attempt < self.max_retries:
                        logger.warning(f"Stage {stage.value} attempt {attempt + 1} failed: {e}. Retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
            
            # Calculate stage duration
            stage_duration = (datetime.now() - stage_start).total_seconds()
            metrics.stage_durations[stage.value] = stage_duration
            
            logger.info(f"Stage {stage.value} completed in {stage_duration:.2f}s")
            self._notify_progress(stage, {
                "status": "completed",
                "duration": stage_duration
            })
            
            # Stage validation
            if self.validate_at_each_stage:
                validation_errors = await self._validate_stage_output(stage, result)
                if validation_errors:
                    if any("error" in err.lower() for err in validation_errors):
                        raise ValueError(f"Stage validation failed: {validation_errors}")
                    else:
                        result.warnings.extend(validation_errors)
                        if not self.continue_on_warnings:
                            raise ValueError(f"Stage warnings treated as errors: {validation_errors}")
            
        except Exception as e:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            metrics.stage_durations[stage.value] = stage_duration
            
            logger.error(f"Stage {stage.value} failed after {stage_duration:.2f}s: {e}")
            self._notify_progress(stage, {
                "status": "failed",
                "duration": stage_duration,
                "error": str(e)
            })
            raise
    
    async def _parse_prd_stage(self, prd_file_path: str, result: PipelineResult, metrics: PipelineMetrics):
        """Execute PRD parsing stage."""
        logger.info(f"Parsing PRD: {prd_file_path}")
        
        if not os.path.exists(prd_file_path):
            raise FileNotFoundError(f"PRD file not found: {prd_file_path}")
        
        # Parse PRD
        prd_content = self.prd_parser.parse_prd(prd_file_path)
        result.prd_content = prd_content
        
        # Update metrics
        metrics.prd_requirements_count = len(prd_content.requirements)
        metrics.validation_errors += len(prd_content.validation_errors)
        
        # Check for critical validation errors
        if prd_content.validation_errors:
            critical_errors = [err for err in prd_content.validation_errors if "missing required" in err.lower()]
            if critical_errors:
                raise ValueError(f"PRD validation failed with critical errors: {critical_errors}")
            else:
                result.warnings.extend(prd_content.validation_errors)
        
        logger.info(f"PRD parsed successfully: {len(prd_content.requirements)} requirements found")
    
    async def _generate_stories_stage(self, prd_content: PRDContent, result: PipelineResult, metrics: PipelineMetrics):
        """Execute story generation stage."""
        logger.info("Generating user stories from PRD")
        
        if not prd_content or not prd_content.requirements:
            raise ValueError("No requirements found in PRD for story generation")
        
        # Generate stories
        generated_stories = await self.story_generator.generate_stories_from_prd(prd_content)
        result.generated_stories = generated_stories
        
        # Update metrics
        metrics.stories_generated = len(generated_stories)
        
        if not generated_stories:
            raise ValueError("No stories were generated from PRD requirements")
        
        logger.info(f"Generated {len(generated_stories)} user stories")
    
    async def _create_jira_issues_stage(self, user_stories: List[UserStory], result: PipelineResult, metrics: PipelineMetrics):
        """Execute Jira issue creation stage."""
        logger.info("Creating Jira issues from user stories")
        
        if not user_stories:
            raise ValueError("No user stories available for Jira creation")
        
        # Validate Jira configuration
        validation_result = self.jira_creator.validate_project_configuration()
        if not validation_result["valid"]:
            result.errors.extend(validation_result["errors"])
            if not self.jira_creator.dry_run:
                raise ValueError(f"Jira configuration invalid: {validation_result['errors']}")
        
        if validation_result["warnings"]:
            result.warnings.extend(validation_result["warnings"])
        
        # Create issues in Jira
        jira_hierarchy = await self.jira_creator.create_stories_bulk(user_stories)
        result.jira_hierarchy = jira_hierarchy
        
        # Update metrics
        metrics.stories_created_in_jira = jira_hierarchy.get_total_issues()
        
        logger.info(f"Created {jira_hierarchy.get_total_issues()} Jira issues")
    
    async def _validation_stage(self, pipeline_result: PipelineResult, result: PipelineResult, metrics: PipelineMetrics):
        """Execute final validation stage."""
        logger.info("Performing final pipeline validation")
        
        validation_errors = []
        
        # Validate PRD parsing
        if not result.prd_content:
            validation_errors.append("PRD content is missing")
        elif not result.prd_content.requirements:
            validation_errors.append("No requirements found in PRD")
        
        # Validate story generation
        if not result.generated_stories:
            validation_errors.append("No user stories were generated")
        elif result.prd_content and len(result.generated_stories) < len(result.prd_content.requirements) * 0.5:
            validation_errors.append("Low story generation rate (less than 50% of requirements)")
        
        # Validate Jira creation
        if not result.jira_hierarchy:
            validation_errors.append("No Jira hierarchy created")
        elif result.jira_hierarchy.get_total_issues() == 0:
            validation_errors.append("No Jira issues were created")
        
        # Data consistency checks
        if result.generated_stories and result.jira_hierarchy:
            expected_issues = len(result.generated_stories) + sum(len(story.tasks) for story in result.generated_stories)
            actual_issues = result.jira_hierarchy.get_total_issues()
            
            if actual_issues < expected_issues * 0.8:  # Allow for some creation failures
                validation_errors.append(f"Jira creation incomplete: {actual_issues}/{expected_issues} issues created")
        
        if validation_errors:
            result.errors.extend(validation_errors)
            metrics.validation_errors += len(validation_errors)
            raise ValueError(f"Pipeline validation failed: {validation_errors}")
        
        logger.info("Pipeline validation passed")
    
    async def _validate_stage_output(self, stage: PipelineStage, result: PipelineResult) -> List[str]:
        """Validate output of a specific stage."""
        validation_errors = []
        
        if stage == PipelineStage.PRD_PARSING:
            if not result.prd_content:
                validation_errors.append("PRD parsing failed - no content generated")
            elif not result.prd_content.requirements:
                validation_errors.append("PRD parsing - no requirements extracted")
        
        elif stage == PipelineStage.STORY_GENERATION:
            if not result.generated_stories:
                validation_errors.append("Story generation failed - no stories created")
            elif len(result.generated_stories) < 1:
                validation_errors.append("Story generation - insufficient stories created")
        
        elif stage == PipelineStage.JIRA_CREATION:
            if not result.jira_hierarchy:
                validation_errors.append("Jira creation failed - no hierarchy created")
            elif result.jira_hierarchy.get_total_issues() == 0:
                validation_errors.append("Jira creation - no issues created")
        
        return validation_errors
    
    async def _save_pipeline_artifacts(self, result: PipelineResult):
        """Save pipeline artifacts to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save PRD analysis
            if result.prd_content:
                prd_artifact_path = self.output_dir / f"prd_analysis_{timestamp}.json"
                with open(prd_artifact_path, 'w') as f:
                    json.dump(result.prd_content.to_dict(), f, indent=2)
                result.artifacts["prd_analysis"] = str(prd_artifact_path)
            
            # Save generated stories
            if result.generated_stories:
                stories_artifact_path = self.output_dir / f"generated_stories_{timestamp}.json"
                stories_data = [story.to_dict() for story in result.generated_stories]
                with open(stories_artifact_path, 'w') as f:
                    json.dump(stories_data, f, indent=2)
                result.artifacts["generated_stories"] = str(stories_artifact_path)
            
            # Save Jira creation report
            if result.jira_hierarchy:
                jira_report_path = self.output_dir / f"jira_creation_report_{timestamp}.txt"
                with open(jira_report_path, 'w') as f:
                    f.write(self.jira_creator.get_creation_report(result.jira_hierarchy))
                result.artifacts["jira_report"] = str(jira_report_path)
                
                # Save Jira hierarchy data
                jira_data_path = self.output_dir / f"jira_hierarchy_{timestamp}.json"
                with open(jira_data_path, 'w') as f:
                    json.dump(result.jira_hierarchy.creation_summary, f, indent=2)
                result.artifacts["jira_data"] = str(jira_data_path)
            
            # Save pipeline execution summary
            summary_path = self.output_dir / f"pipeline_summary_{timestamp}.json"
            with open(summary_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            result.artifacts["pipeline_summary"] = str(summary_path)
            
            logger.info(f"Pipeline artifacts saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline artifacts: {e}")
            result.warnings.append(f"Artifact saving failed: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "pipeline_id": self.pipeline_id,
            "current_stage": self.current_stage.value,
            "current_status": self.current_status.value,
            "output_dir": str(self.output_dir)
        }
    
    async def cancel_pipeline(self):
        """Cancel running pipeline."""
        logger.info("Cancelling pipeline execution")
        self.current_status = PipelineStatus.CANCELLED
        # Implementation would include task cancellation logic
    
    def generate_pipeline_report(self, result: PipelineResult) -> str:
        """
        Generate comprehensive pipeline execution report.
        
        Args:
            result: Pipeline execution result
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=== Feature Development Pipeline Report ===",
            f"Pipeline ID: {self.pipeline_id}",
            f"Status: {result.status.value.upper()}",
            f"Execution Time: {result.metrics.duration_seconds:.2f}s" if result.metrics else "Unknown",
            ""
        ]
        
        # PRD Analysis Section
        if result.prd_content:
            report_lines.extend([
                "PRD Analysis:",
                f"  Title: {result.prd_content.metadata.title}",
                f"  Author: {result.prd_content.metadata.author}",
                f"  Requirements Found: {len(result.prd_content.requirements)}",
                f"  Validation Errors: {len(result.prd_content.validation_errors)}",
                ""
            ])
        
        # Story Generation Section
        if result.generated_stories:
            story_types = {}
            for story in result.generated_stories:
                story_types[story.story_type.value] = story_types.get(story.story_type.value, 0) + 1
            
            report_lines.extend([
                "Story Generation:",
                f"  Total Stories: {len(result.generated_stories)}",
                "  By Type:"
            ])
            
            for story_type, count in story_types.items():
                report_lines.append(f"    {story_type.title()}: {count}")
            
            report_lines.append("")
        
        # Jira Creation Section
        if result.jira_hierarchy:
            summary = result.jira_hierarchy.creation_summary
            report_lines.extend([
                "Jira Creation:",
                f"  Project: {summary.get('project_key', 'Unknown')}",
                f"  Total Issues: {summary.get('total_issues_created', 0)}",
                f"  Epics: {summary.get('epics_created', 0)}",
                f"  Features: {summary.get('features_created', 0)}",
                f"  Stories: {summary.get('stories_created', 0)}",
                f"  Tasks: {summary.get('tasks_created', 0)}",
                ""
            ])
        
        # Metrics Section
        if result.metrics:
            report_lines.extend([
                "Performance Metrics:",
                f"  Total Duration: {result.metrics.duration_seconds:.2f}s",
                "  Stage Durations:"
            ])
            
            for stage, duration in result.metrics.stage_durations.items():
                report_lines.append(f"    {stage.replace('_', ' ').title()}: {duration:.2f}s")
            
            report_lines.append("")
        
        # Issues Section
        if result.errors:
            report_lines.extend([
                "Errors:",
                *[f"  - {error}" for error in result.errors],
                ""
            ])
        
        if result.warnings:
            report_lines.extend([
                "Warnings:",
                *[f"  - {warning}" for warning in result.warnings],
                ""
            ])
        
        # Artifacts Section
        if result.artifacts:
            report_lines.extend([
                "Generated Artifacts:",
                *[f"  {name}: {path}" for name, path in result.artifacts.items()]
            ])
        
        return "\n".join(report_lines)