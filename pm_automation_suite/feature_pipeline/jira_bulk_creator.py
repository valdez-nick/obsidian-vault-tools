"""
Jira Bulk Creator Implementation

Automates bulk creation of Jira stories with hierarchy management, linking,
and field mapping from generated user stories.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

try:
    from jira import JIRA
    from jira.exceptions import JIRAError
    JIRA_AVAILABLE = True
except ImportError:
    JIRA_AVAILABLE = False

from .story_generator import UserStory, StoryType, AcceptanceCriteria, Task
from .prd_parser import RequirementPriority

logger = logging.getLogger(__name__)


class JiraIssueType(Enum):
    """Jira issue types."""
    EPIC = "Epic"
    STORY = "Story"
    TASK = "Task"
    SUB_TASK = "Sub-task"
    BUG = "Bug"
    SPIKE = "Spike"


class JiraStatus(Enum):
    """Common Jira statuses."""
    TODO = "To Do"
    IN_PROGRESS = "In Progress"
    DONE = "Done"
    BLOCKED = "Blocked"
    IN_REVIEW = "In Review"


@dataclass
class JiraFieldMapping:
    """Mapping between story fields and Jira fields."""
    summary_field: str = "summary"
    description_field: str = "description"
    story_points_field: str = "customfield_10016"  # Common story points field
    epic_link_field: str = "customfield_10014"    # Common epic link field
    priority_field: str = "priority"
    assignee_field: str = "assignee"
    labels_field: str = "labels"
    components_field: str = "components"
    fix_versions_field: str = "fixVersions"


@dataclass
class JiraIssue:
    """Represents a Jira issue to be created."""
    key: Optional[str] = None
    issue_type: JiraIssueType = JiraIssueType.STORY
    summary: str = ""
    description: str = ""
    priority: str = "Medium"
    assignee: Optional[str] = None
    story_points: Optional[int] = None
    labels: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    epic_link: Optional[str] = None
    parent_key: Optional[str] = None
    acceptance_criteria: List[str] = field(default_factory=list)
    tasks: List['JiraIssue'] = field(default_factory=list)
    
    def to_jira_fields(self, field_mapping: JiraFieldMapping, project_key: str) -> Dict[str, Any]:
        """Convert to Jira API fields format."""
        fields = {
            "project": {"key": project_key},
            "issuetype": {"name": self.issue_type.value},
            field_mapping.summary_field: self.summary,
            field_mapping.description_field: self._format_description(),
            field_mapping.priority_field: {"name": self.priority}
        }
        
        # Add optional fields
        if self.assignee:
            fields[field_mapping.assignee_field] = {"name": self.assignee}
        
        if self.story_points:
            fields[field_mapping.story_points_field] = self.story_points
        
        if self.labels:
            fields[field_mapping.labels_field] = self.labels
        
        if self.components:
            fields[field_mapping.components_field] = [{"name": comp} for comp in self.components]
        
        if self.epic_link and self.issue_type != JiraIssueType.EPIC:
            fields[field_mapping.epic_link_field] = self.epic_link
        
        if self.parent_key and self.issue_type == JiraIssueType.SUB_TASK:
            fields["parent"] = {"key": self.parent_key}
        
        return {"fields": fields}
    
    def _format_description(self) -> str:
        """Format description with acceptance criteria."""
        description_parts = [self.description]
        
        if self.acceptance_criteria:
            description_parts.append("\n\nh3. Acceptance Criteria")
            for i, criteria in enumerate(self.acceptance_criteria, 1):
                description_parts.append(f"{i}. {criteria}")
        
        return "\n".join(description_parts)


@dataclass
class StoryHierarchy:
    """Represents the hierarchy of stories created in Jira."""
    epics: List[JiraIssue] = field(default_factory=list)
    features: List[JiraIssue] = field(default_factory=list)
    stories: List[JiraIssue] = field(default_factory=list)
    tasks: List[JiraIssue] = field(default_factory=list)
    creation_summary: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_issues(self) -> int:
        """Get total number of issues created."""
        return len(self.epics) + len(self.features) + len(self.stories) + len(self.tasks)
    
    def get_issues_by_type(self, issue_type: JiraIssueType) -> List[JiraIssue]:
        """Get issues by type."""
        type_mapping = {
            JiraIssueType.EPIC: self.epics,
            JiraIssueType.STORY: self.features + self.stories,
            JiraIssueType.TASK: self.tasks,
            JiraIssueType.SUB_TASK: self.tasks
        }
        return type_mapping.get(issue_type, [])


class JiraBulkCreator:
    """
    Bulk creator for Jira stories from generated user stories.
    
    Handles epic/feature/story hierarchy, field mapping, error handling,
    and bulk operations with rate limiting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Jira Bulk Creator.
        
        Args:
            config: Configuration with Jira connection and mapping settings
        """
        self.config = config
        self.jira_client = None
        self.project_key = config.get('jira_project_key', 'PROJ')
        self.field_mapping = JiraFieldMapping(**config.get('field_mapping', {}))
        self.dry_run = config.get('dry_run', False)
        self.batch_size = config.get('batch_size', 10)
        self.rate_limit_delay = config.get('rate_limit_delay', 1.0)
        
        # Issue type mappings
        self.story_type_mapping = {
            StoryType.EPIC: JiraIssueType.EPIC,
            StoryType.FEATURE: JiraIssueType.STORY,
            StoryType.USER_STORY: JiraIssueType.STORY,
            StoryType.TASK: JiraIssueType.TASK,
            StoryType.BUG: JiraIssueType.BUG,
            StoryType.SPIKE: JiraIssueType.SPIKE
        }
        
        # Priority mappings
        self.priority_mapping = {
            RequirementPriority.CRITICAL: "Highest",
            RequirementPriority.HIGH: "High", 
            RequirementPriority.MEDIUM: "Medium",
            RequirementPriority.LOW: "Low",
            RequirementPriority.NICE_TO_HAVE: "Lowest"
        }
        
        # Initialize Jira client
        self._initialize_jira_client()
    
    def _initialize_jira_client(self):
        """Initialize Jira client connection."""
        if not JIRA_AVAILABLE:
            logger.warning("Jira library not available. Running in simulation mode.")
            return
        
        try:
            jira_config = self.config.get('jira', {})
            server = jira_config.get('server')
            username = jira_config.get('username')
            api_token = jira_config.get('api_token')
            
            if not all([server, username, api_token]):
                logger.warning("Incomplete Jira configuration. Running in dry run mode.")
                self.dry_run = True
                return
            
            self.jira_client = JIRA(
                server=server,
                basic_auth=(username, api_token),
                options={'rest_api_version': '3'}
            )
            
            logger.info(f"Connected to Jira: {server}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Jira: {e}")
            self.dry_run = True
    
    async def create_stories_bulk(self, user_stories: List[UserStory]) -> StoryHierarchy:
        """
        Create user stories in Jira with proper hierarchy.
        
        Args:
            user_stories: List of user stories to create
            
        Returns:
            StoryHierarchy with created issues
        """
        logger.info(f"Starting bulk creation of {len(user_stories)} stories")
        
        try:
            # Convert user stories to Jira issues
            jira_issues = self._convert_stories_to_jira_issues(user_stories)
            
            # Group by hierarchy level
            epics = [issue for issue in jira_issues if issue.issue_type == JiraIssueType.EPIC]
            features = [issue for issue in jira_issues if issue.issue_type == JiraIssueType.STORY and not issue.epic_link]
            stories = [issue for issue in jira_issues if issue.issue_type == JiraIssueType.STORY and issue.epic_link]
            tasks = [issue for issue in jira_issues if issue.issue_type in [JiraIssueType.TASK, JiraIssueType.SUB_TASK]]
            
            hierarchy = StoryHierarchy()
            
            # Create in order: Epics -> Features -> Stories -> Tasks
            if epics:
                created_epics = await self._create_issues_batch(epics, "epics")
                hierarchy.epics = created_epics
                
                # Update epic links for child issues
                self._update_epic_links(created_epics, features + stories)
            
            if features:
                created_features = await self._create_issues_batch(features, "features")
                hierarchy.features = created_features
            
            if stories:
                created_stories = await self._create_issues_batch(stories, "stories")
                hierarchy.stories = created_stories
            
            if tasks:
                # Update parent links for sub-tasks
                self._update_parent_links(hierarchy.stories, tasks)
                created_tasks = await self._create_issues_batch(tasks, "tasks")
                hierarchy.tasks = created_tasks
            
            # Generate creation summary
            hierarchy.creation_summary = self._generate_creation_summary(hierarchy)
            
            logger.info(f"Bulk creation completed. Total issues: {hierarchy.get_total_issues()}")
            return hierarchy
            
        except Exception as e:
            logger.error(f"Bulk creation failed: {e}")
            raise
    
    def _convert_stories_to_jira_issues(self, user_stories: List[UserStory]) -> List[JiraIssue]:
        """Convert user stories to Jira issues."""
        jira_issues = []
        
        for story in user_stories:
            jira_issue = JiraIssue(
                issue_type=self.story_type_mapping.get(story.story_type, JiraIssueType.STORY),
                summary=story.title,
                description=story.description,
                priority=self.priority_mapping.get(story.priority, "Medium"),
                assignee=getattr(story, 'assignee', None),
                story_points=story.story_points,
                labels=story.labels,
                acceptance_criteria=[ac.description for ac in story.acceptance_criteria]
            )
            
            # Set epic link for non-epic issues
            if story.parent_epic and jira_issue.issue_type != JiraIssueType.EPIC:
                jira_issue.epic_link = story.parent_epic
            
            # Convert tasks to sub-tasks
            for task in story.tasks:
                sub_task = JiraIssue(
                    issue_type=JiraIssueType.SUB_TASK,
                    summary=task.title,
                    description=task.description,
                    assignee=getattr(task, 'assignee', None),
                    labels=getattr(task, 'labels', [])
                )
                jira_issue.tasks.append(sub_task)
            
            jira_issues.append(jira_issue)
            jira_issues.extend(jira_issue.tasks)
        
        return jira_issues
    
    async def _create_issues_batch(self, issues: List[JiraIssue], batch_name: str) -> List[JiraIssue]:
        """Create issues in batches with rate limiting."""
        if not issues:
            return []
        
        logger.info(f"Creating {len(issues)} {batch_name}")
        created_issues = []
        
        for i in range(0, len(issues), self.batch_size):
            batch = issues[i:i + self.batch_size]
            batch_results = await self._create_issue_batch(batch)
            created_issues.extend(batch_results)
            
            # Rate limiting
            if i + self.batch_size < len(issues):
                await asyncio.sleep(self.rate_limit_delay)
        
        logger.info(f"Created {len(created_issues)} {batch_name}")
        return created_issues
    
    async def _create_issue_batch(self, issues: List[JiraIssue]) -> List[JiraIssue]:
        """Create a batch of issues."""
        created_issues = []
        
        for issue in issues:
            try:
                if self.dry_run or not self.jira_client:
                    # Simulate creation
                    issue.key = f"PROJ-{len(created_issues) + 1:04d}"
                    logger.info(f"[DRY RUN] Would create: {issue.key} - {issue.summary}")
                else:
                    # Create in Jira
                    jira_fields = issue.to_jira_fields(self.field_mapping, self.project_key)
                    created_issue = self.jira_client.create_issue(fields=jira_fields['fields'])
                    issue.key = created_issue.key
                    logger.info(f"Created: {issue.key} - {issue.summary}")
                
                created_issues.append(issue)
                
            except Exception as e:
                logger.error(f"Failed to create issue '{issue.summary}': {e}")
                # Continue with other issues
                continue
        
        return created_issues
    
    def _update_epic_links(self, epics: List[JiraIssue], child_issues: List[JiraIssue]):
        """Update epic links based on created epic keys."""
        epic_mapping = {}
        
        # Create mapping from epic titles to keys
        for epic in epics:
            if epic.key:
                # Simple matching by title similarity
                epic_mapping[epic.summary.lower()] = epic.key
        
        # Update child issue epic links
        for issue in child_issues:
            if issue.epic_link:
                # Try to find matching epic
                for epic_title, epic_key in epic_mapping.items():
                    if epic_title in issue.epic_link.lower() or issue.epic_link.lower() in epic_title:
                        issue.epic_link = epic_key
                        break
    
    def _update_parent_links(self, parent_stories: List[JiraIssue], sub_tasks: List[JiraIssue]):
        """Update parent links for sub-tasks."""
        parent_mapping = {}
        
        # Create mapping from story titles to keys
        for story in parent_stories:
            if story.key:
                parent_mapping[story.summary.lower()] = story.key
        
        # Update sub-task parent links
        for task in sub_tasks:
            if task.issue_type == JiraIssueType.SUB_TASK:
                # Try to find matching parent story
                for story_title, story_key in parent_mapping.items():
                    if story_title in task.summary.lower():
                        task.parent_key = story_key
                        break
    
    def _generate_creation_summary(self, hierarchy: StoryHierarchy) -> Dict[str, Any]:
        """Generate summary of creation results."""
        return {
            "total_issues_created": hierarchy.get_total_issues(),
            "epics_created": len(hierarchy.epics),
            "features_created": len(hierarchy.features),
            "stories_created": len(hierarchy.stories),
            "tasks_created": len(hierarchy.tasks),
            "creation_time": datetime.now().isoformat(),
            "project_key": self.project_key,
            "dry_run": self.dry_run,
            "epic_keys": [epic.key for epic in hierarchy.epics if epic.key],
            "feature_keys": [feature.key for feature in hierarchy.features if feature.key]
        }
    
    async def create_single_story(self, story: UserStory) -> Optional[JiraIssue]:
        """
        Create a single user story in Jira.
        
        Args:
            story: User story to create
            
        Returns:
            Created Jira issue or None if failed
        """
        try:
            jira_issues = self._convert_stories_to_jira_issues([story])
            if jira_issues:
                created_issues = await self._create_issue_batch([jira_issues[0]])
                return created_issues[0] if created_issues else None
        except Exception as e:
            logger.error(f"Failed to create single story: {e}")
            return None
    
    async def update_issue(self, issue_key: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing Jira issue.
        
        Args:
            issue_key: Jira issue key
            updates: Fields to update
            
        Returns:
            True if successful
        """
        if self.dry_run or not self.jira_client:
            logger.info(f"[DRY RUN] Would update {issue_key} with {updates}")
            return True
        
        try:
            issue = self.jira_client.issue(issue_key)
            issue.update(fields=updates)
            logger.info(f"Updated {issue_key}")
            return True
        except Exception as e:
            logger.error(f"Failed to update {issue_key}: {e}")
            return False
    
    async def link_issues(self, inward_issue: str, outward_issue: str, link_type: str = "Blocks") -> bool:
        """
        Create link between issues.
        
        Args:
            inward_issue: Source issue key
            outward_issue: Target issue key
            link_type: Type of link
            
        Returns:
            True if successful
        """
        if self.dry_run or not self.jira_client:
            logger.info(f"[DRY RUN] Would link {inward_issue} -> {outward_issue} ({link_type})")
            return True
        
        try:
            self.jira_client.create_issue_link(
                type=link_type,
                inwardIssue=inward_issue,
                outwardIssue=outward_issue
            )
            logger.info(f"Linked {inward_issue} -> {outward_issue}")
            return True
        except Exception as e:
            logger.error(f"Failed to link issues: {e}")
            return False
    
    def validate_project_configuration(self) -> Dict[str, Any]:
        """
        Validate Jira project configuration.
        
        Returns:
            Validation results
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "project_info": {}
        }
        
        if not JIRA_AVAILABLE:
            validation["errors"].append("Jira library not available")
            validation["valid"] = False
            return validation
        
        if not self.jira_client:
            validation["errors"].append("Jira client not connected")
            validation["valid"] = False
            return validation
        
        try:
            # Check project exists
            project = self.jira_client.project(self.project_key)
            validation["project_info"] = {
                "key": project.key,
                "name": project.name,
                "lead": project.lead.displayName if project.lead else None
            }
            
            # Check issue types
            issue_types = self.jira_client.issue_types()
            available_types = [it.name for it in issue_types]
            
            for jira_type in JiraIssueType:
                if jira_type.value not in available_types:
                    validation["warnings"].append(f"Issue type '{jira_type.value}' not available")
            
            # Check custom fields
            fields = self.jira_client.fields()
            field_mapping_dict = {
                "Story Points": self.field_mapping.story_points_field,
                "Epic Link": self.field_mapping.epic_link_field
            }
            
            for field_name, field_id in field_mapping_dict.items():
                if not any(f['id'] == field_id for f in fields):
                    validation["warnings"].append(f"Custom field '{field_name}' ({field_id}) not found")
            
        except Exception as e:
            validation["errors"].append(f"Project validation failed: {e}")
            validation["valid"] = False
        
        return validation
    
    def get_creation_report(self, hierarchy: StoryHierarchy) -> str:
        """
        Generate human-readable creation report.
        
        Args:
            hierarchy: Created story hierarchy
            
        Returns:
            Formatted report string
        """
        summary = hierarchy.creation_summary
        
        report_lines = [
            "=== Jira Story Creation Report ===",
            f"Project: {summary.get('project_key', 'Unknown')}",
            f"Creation Time: {summary.get('creation_time', 'Unknown')}",
            f"Mode: {'DRY RUN' if summary.get('dry_run') else 'LIVE'}",
            "",
            "Issues Created:",
            f"  Epics: {summary.get('epics_created', 0)}",
            f"  Features: {summary.get('features_created', 0)}",
            f"  Stories: {summary.get('stories_created', 0)}",
            f"  Tasks: {summary.get('tasks_created', 0)}",
            f"  Total: {summary.get('total_issues_created', 0)}",
            ""
        ]
        
        if summary.get('epic_keys'):
            report_lines.append("Epic Keys:")
            for key in summary['epic_keys']:
                report_lines.append(f"  - {key}")
            report_lines.append("")
        
        if summary.get('feature_keys'):
            report_lines.append("Feature Keys:")
            for key in summary['feature_keys']:
                report_lines.append(f"  - {key}")
        
        return "\n".join(report_lines)