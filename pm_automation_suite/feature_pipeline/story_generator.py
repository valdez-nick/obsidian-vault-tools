"""
Story Generator Implementation

AI-powered generation of user stories from PRD requirements with acceptance criteria,
task breakdown, and effort estimation.
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .prd_parser import PRDContent, Requirement, RequirementType, RequirementPriority

logger = logging.getLogger(__name__)


class StoryType(Enum):
    """Types of user stories."""
    EPIC = "epic"
    FEATURE = "feature"
    USER_STORY = "user_story"
    TASK = "task"
    BUG = "bug"
    SPIKE = "spike"


class EstimationMethod(Enum):
    """Story point estimation methods."""
    FIBONACCI = "fibonacci"  # 1, 2, 3, 5, 8, 13, 21
    T_SHIRT = "t_shirt"      # XS, S, M, L, XL, XXL
    HOURS = "hours"          # Hour-based estimates


@dataclass
class AcceptanceCriteria:
    """Acceptance criteria for user stories."""
    id: str
    description: str
    given: str
    when: str
    then: str
    priority: RequirementPriority = RequirementPriority.MEDIUM
    
    def to_gherkin(self) -> str:
        """Convert to Gherkin format."""
        return f"Given {self.given}\nWhen {self.when}\nThen {self.then}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'description': self.description,
            'given': self.given,
            'when': self.when,
            'then': self.then,
            'priority': self.priority.value
        }


@dataclass
class Task:
    """Individual task within a user story."""
    id: str
    title: str
    description: str
    estimated_hours: float
    assignee: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'estimated_hours': self.estimated_hours,
            'assignee': self.assignee,
            'dependencies': self.dependencies,
            'labels': self.labels
        }


@dataclass
class UserStory:
    """User story with full details."""
    id: str
    title: str
    description: str
    story_type: StoryType
    priority: RequirementPriority
    story_points: int
    user_persona: str
    business_value: str
    acceptance_criteria: List[AcceptanceCriteria] = field(default_factory=list)
    tasks: List[Task] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    parent_epic: Optional[str] = None
    source_requirement: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_as_a_user_format(self) -> str:
        """Generate 'As a user' format story."""
        return f"As a {self.user_persona}, I want {self.description}, so that {self.business_value}"
    
    def estimate_total_hours(self) -> float:
        """Calculate total estimated hours from tasks."""
        return sum(task.estimated_hours for task in self.tasks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'story_type': self.story_type.value,
            'priority': self.priority.value,
            'story_points': self.story_points,
            'user_persona': self.user_persona,
            'business_value': self.business_value,
            'acceptance_criteria': [ac.to_dict() for ac in self.acceptance_criteria],
            'tasks': [task.to_dict() for task in self.tasks],
            'dependencies': self.dependencies,
            'labels': self.labels,
            'parent_epic': self.parent_epic,
            'source_requirement': self.source_requirement,
            'created_at': self.created_at.isoformat(),
            'user_story_format': self.get_as_a_user_format()
        }


class StoryGenerator:
    """
    AI-powered story generator for converting PRD requirements into user stories.
    
    Uses LLMs to create well-structured user stories with acceptance criteria,
    task breakdown, and effort estimation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Story Generator.
        
        Args:
            config: Configuration with AI settings and story templates
        """
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize AI clients
        if OPENAI_AVAILABLE and 'openai_api_key' in config:
            openai.api_key = config['openai_api_key']
            self.openai_client = openai
        
        if ANTHROPIC_AVAILABLE and 'anthropic_api_key' in config:
            self.anthropic_client = anthropic.Client(api_key=config['anthropic_api_key'])
        
        # Configuration settings
        self.estimation_method = EstimationMethod(config.get('estimation_method', 'fibonacci'))
        self.default_personas = config.get('user_personas', [
            'end user', 'administrator', 'developer', 'business user', 'customer'
        ])
        
        # Story templates
        self.story_templates = self._load_story_templates()
        
        # Estimation mappings
        self.fibonacci_scale = [1, 2, 3, 5, 8, 13, 21]
        self.complexity_factors = {
            RequirementType.FUNCTIONAL: 1.0,
            RequirementType.NON_FUNCTIONAL: 1.5,
            RequirementType.TECHNICAL: 1.3,
            RequirementType.BUSINESS: 0.8
        }
    
    def _load_story_templates(self) -> Dict[StoryType, Dict[str, Any]]:
        """Load story templates for different story types."""
        return {
            StoryType.EPIC: {
                'title_format': 'Epic: {feature_name}',
                'description_template': 'As a {persona}, I need {capability} so that {business_value}',
                'default_story_points': 21,
                'max_child_stories': 8
            },
            StoryType.FEATURE: {
                'title_format': '{feature_name}',
                'description_template': 'As a {persona}, I want {feature} so that {benefit}',
                'default_story_points': 8,
                'max_child_stories': 5
            },
            StoryType.USER_STORY: {
                'title_format': '{action} {object}',
                'description_template': 'As a {persona}, I want to {action} so that {benefit}',
                'default_story_points': 3,
                'max_child_stories': 0
            },
            StoryType.TASK: {
                'title_format': '{task_name}',
                'description_template': '{technical_description}',
                'default_story_points': 1,
                'max_child_stories': 0
            }
        }
    
    async def generate_stories_from_prd(self, prd_content: PRDContent) -> List[UserStory]:
        """
        Generate user stories from PRD content.
        
        Args:
            prd_content: Parsed PRD content
            
        Returns:
            List of generated user stories
        """
        logger.info(f"Generating stories from PRD: {prd_content.metadata.title}")
        
        try:
            stories = []
            
            # Group requirements by type and priority
            grouped_requirements = self._group_requirements(prd_content.requirements)
            
            # Generate epics from high-level requirements
            epics = await self._generate_epics(grouped_requirements, prd_content)
            stories.extend(epics)
            
            # Generate features and user stories
            for epic in epics:
                features = await self._generate_features_for_epic(epic, grouped_requirements)
                stories.extend(features)
                
                for feature in features:
                    user_stories = await self._generate_user_stories_for_feature(feature, grouped_requirements)
                    stories.extend(user_stories)
            
            # Generate standalone stories for remaining requirements
            remaining_requirements = self._get_unprocessed_requirements(
                prd_content.requirements, stories
            )
            
            standalone_stories = await self._generate_standalone_stories(remaining_requirements)
            stories.extend(standalone_stories)
            
            # Post-process stories
            stories = self._post_process_stories(stories)
            
            logger.info(f"Generated {len(stories)} stories ({len(epics)} epics)")
            return stories
            
        except Exception as e:
            logger.error(f"Failed to generate stories: {e}")
            raise
    
    def _group_requirements(self, requirements: List[Requirement]) -> Dict[str, List[Requirement]]:
        """Group requirements by type and characteristics."""
        groups = {
            'high_level': [],  # Potential epics
            'features': [],    # Feature-level requirements
            'detailed': [],    # Detailed user stories
            'technical': [],   # Technical tasks
            'business': []     # Business requirements
        }
        
        for req in requirements:
            if req.priority in [RequirementPriority.CRITICAL, RequirementPriority.HIGH]:
                if req.requirement_type == RequirementType.BUSINESS:
                    groups['high_level'].append(req)
                elif req.requirement_type == RequirementType.FUNCTIONAL:
                    groups['features'].append(req)
                else:
                    groups['detailed'].append(req)
            elif req.requirement_type == RequirementType.TECHNICAL:
                groups['technical'].append(req)
            else:
                groups['detailed'].append(req)
        
        return groups
    
    async def _generate_epics(
        self, 
        grouped_requirements: Dict[str, List[Requirement]], 
        prd_content: PRDContent
    ) -> List[UserStory]:
        """Generate epic-level stories."""
        epics = []
        high_level_reqs = grouped_requirements.get('high_level', [])
        
        if not high_level_reqs:
            # Create a default epic from PRD metadata
            epic = UserStory(
                id=f"EPIC-{str(uuid.uuid4())[:8]}",
                title=f"Epic: {prd_content.metadata.title}",
                description=f"Implement {prd_content.metadata.title} functionality",
                story_type=StoryType.EPIC,
                priority=RequirementPriority.HIGH,
                story_points=21,
                user_persona="end user",
                business_value="deliver the planned feature set"
            )
            epics.append(epic)
        else:
            # Generate epics from high-level requirements
            for req in high_level_reqs:
                epic = await self._generate_epic_from_requirement(req)
                epics.append(epic)
        
        return epics
    
    async def _generate_epic_from_requirement(self, requirement: Requirement) -> UserStory:
        """Generate an epic from a high-level requirement."""
        if self.openai_client:
            epic_data = await self._generate_epic_with_ai(requirement)
        else:
            epic_data = self._generate_epic_template(requirement)
        
        return UserStory(
            id=f"EPIC-{str(uuid.uuid4())[:8]}",
            title=epic_data['title'],
            description=epic_data['description'],
            story_type=StoryType.EPIC,
            priority=requirement.priority,
            story_points=epic_data.get('story_points', 21),
            user_persona=epic_data.get('user_persona', 'end user'),
            business_value=epic_data.get('business_value', 'deliver business value'),
            source_requirement=requirement.id
        )
    
    async def _generate_epic_with_ai(self, requirement: Requirement) -> Dict[str, Any]:
        """Generate epic using AI."""
        try:
            prompt = f"""
            Create an epic-level user story from this requirement:
            
            Requirement: {requirement.text}
            Priority: {requirement.priority.value}
            Type: {requirement.requirement_type.value}
            
            Generate:
            1. Epic title (concise, descriptive)
            2. Epic description in "As a X, I need Y so that Z" format
            3. User persona (who benefits)
            4. Business value (why it matters)
            5. Story points estimate (13-21 for epics)
            
            Format as JSON with keys: title, description, user_persona, business_value, story_points
            """
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"AI epic generation failed: {e}")
            return self._generate_epic_template(requirement)
    
    def _generate_epic_template(self, requirement: Requirement) -> Dict[str, Any]:
        """Generate epic using template (fallback)."""
        # Extract key concepts from requirement text
        key_concepts = self._extract_key_concepts(requirement.text)
        main_concept = key_concepts[0] if key_concepts else "feature"
        
        return {
            'title': f"Epic: {main_concept.title()}",
            'description': f"As an end user, I need {main_concept} capability so that I can accomplish my goals",
            'user_persona': 'end user',
            'business_value': 'deliver value to users',
            'story_points': 21
        }
    
    async def _generate_features_for_epic(
        self, 
        epic: UserStory, 
        grouped_requirements: Dict[str, List[Requirement]]
    ) -> List[UserStory]:
        """Generate feature stories for an epic."""
        features = []
        feature_reqs = grouped_requirements.get('features', [])
        
        # Limit features per epic
        max_features = self.story_templates[StoryType.EPIC]['max_child_stories']
        feature_reqs = feature_reqs[:max_features]
        
        for req in feature_reqs:
            if self.openai_client:
                feature_data = await self._generate_feature_with_ai(req, epic)
            else:
                feature_data = self._generate_feature_template(req)
            
            feature = UserStory(
                id=f"FEAT-{str(uuid.uuid4())[:8]}",
                title=feature_data['title'],
                description=feature_data['description'],
                story_type=StoryType.FEATURE,
                priority=req.priority,
                story_points=feature_data.get('story_points', 8),
                user_persona=feature_data.get('user_persona', epic.user_persona),
                business_value=feature_data.get('business_value', epic.business_value),
                parent_epic=epic.id,
                source_requirement=req.id
            )
            
            # Generate acceptance criteria
            feature.acceptance_criteria = await self._generate_acceptance_criteria(req, feature)
            
            features.append(feature)
        
        return features
    
    async def _generate_feature_with_ai(self, requirement: Requirement, epic: UserStory) -> Dict[str, Any]:
        """Generate feature using AI."""
        try:
            prompt = f"""
            Create a feature-level user story from this requirement for the epic "{epic.title}":
            
            Requirement: {requirement.text}
            Epic Context: {epic.description}
            
            Generate:
            1. Feature title (specific, actionable)
            2. Feature description in "As a X, I want Y so that Z" format
            3. User persona (inherit from epic or specify)
            4. Business value (how it contributes to epic)
            5. Story points estimate (5-13 for features)
            
            Format as JSON with keys: title, description, user_persona, business_value, story_points
            """
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"AI feature generation failed: {e}")
            return self._generate_feature_template(requirement)
    
    def _generate_feature_template(self, requirement: Requirement) -> Dict[str, Any]:
        """Generate feature using template (fallback)."""
        concepts = self._extract_key_concepts(requirement.text)
        action_verb = self._extract_action_verb(requirement.text)
        
        title = f"{action_verb.title()} {concepts[0] if concepts else 'Feature'}"
        
        return {
            'title': title,
            'description': f"As a user, I want to {action_verb} {concepts[0] if concepts else 'feature'} so that I can meet my needs",
            'user_persona': 'user',
            'business_value': 'provide user functionality',
            'story_points': 8
        }
    
    async def _generate_user_stories_for_feature(
        self, 
        feature: UserStory, 
        grouped_requirements: Dict[str, List[Requirement]]
    ) -> List[UserStory]:
        """Generate user stories for a feature."""
        user_stories = []
        detailed_reqs = grouped_requirements.get('detailed', [])
        
        # Select requirements related to this feature
        related_reqs = self._find_related_requirements(feature, detailed_reqs)
        
        for req in related_reqs:
            if self.openai_client:
                story_data = await self._generate_user_story_with_ai(req, feature)
            else:
                story_data = self._generate_user_story_template(req)
            
            user_story = UserStory(
                id=f"US-{str(uuid.uuid4())[:8]}",
                title=story_data['title'],
                description=story_data['description'],
                story_type=StoryType.USER_STORY,
                priority=req.priority,
                story_points=story_data.get('story_points', 3),
                user_persona=story_data.get('user_persona', feature.user_persona),
                business_value=story_data.get('business_value', feature.business_value),
                parent_epic=feature.parent_epic,
                source_requirement=req.id
            )
            
            # Generate acceptance criteria and tasks
            user_story.acceptance_criteria = await self._generate_acceptance_criteria(req, user_story)
            user_story.tasks = await self._generate_tasks_for_story(user_story, req)
            
            user_stories.append(user_story)
        
        return user_stories
    
    async def _generate_user_story_with_ai(self, requirement: Requirement, feature: UserStory) -> Dict[str, Any]:
        """Generate user story using AI."""
        try:
            prompt = f"""
            Create a user story from this requirement for the feature "{feature.title}":
            
            Requirement: {requirement.text}
            Feature Context: {feature.description}
            
            Generate:
            1. Story title (action-oriented, specific)
            2. Story description in "As a X, I want to Y so that Z" format
            3. User persona (who performs this action)
            4. Business value (benefit to user/business)
            5. Story points estimate (1-8 for user stories)
            
            Format as JSON with keys: title, description, user_persona, business_value, story_points
            """
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"AI user story generation failed: {e}")
            return self._generate_user_story_template(requirement)
    
    def _generate_user_story_template(self, requirement: Requirement) -> Dict[str, Any]:
        """Generate user story using template (fallback)."""
        action = self._extract_action_verb(requirement.text)
        object_concept = self._extract_key_concepts(requirement.text)[0] if self._extract_key_concepts(requirement.text) else "data"
        
        return {
            'title': f"{action.title()} {object_concept}",
            'description': f"As a user, I want to {action} {object_concept} so that I can complete my task",
            'user_persona': 'user',
            'business_value': 'complete user tasks efficiently',
            'story_points': self._estimate_story_points(requirement)
        }
    
    async def _generate_acceptance_criteria(
        self, 
        requirement: Requirement, 
        story: UserStory
    ) -> List[AcceptanceCriteria]:
        """Generate acceptance criteria for a story."""
        criteria = []
        
        # Use existing acceptance criteria from requirement
        for i, ac_text in enumerate(requirement.acceptance_criteria):
            # Parse Gherkin format if present
            given, when, then = self._parse_gherkin(ac_text)
            
            criterion = AcceptanceCriteria(
                id=f"AC-{story.id}-{i+1:02d}",
                description=ac_text,
                given=given,
                when=when,
                then=then,
                priority=requirement.priority
            )
            criteria.append(criterion)
        
        # Generate additional criteria if needed
        if not criteria and self.openai_client:
            ai_criteria = await self._generate_acceptance_criteria_with_ai(requirement, story)
            criteria.extend(ai_criteria)
        
        # Fallback: generate basic criteria
        if not criteria:
            criteria.append(AcceptanceCriteria(
                id=f"AC-{story.id}-01",
                description="Basic functionality works as expected",
                given="the user is in the system",
                when="they perform the required action",
                then="the expected result occurs"
            ))
        
        return criteria
    
    async def _generate_acceptance_criteria_with_ai(
        self, 
        requirement: Requirement, 
        story: UserStory
    ) -> List[AcceptanceCriteria]:
        """Generate acceptance criteria using AI."""
        try:
            prompt = f"""
            Generate 2-3 acceptance criteria for this user story:
            
            Story: {story.title}
            Description: {story.description}
            Requirement: {requirement.text}
            
            Create acceptance criteria in Gherkin format:
            Given [context]
            When [action]
            Then [expected result]
            
            Format as JSON array with objects containing: description, given, when, then
            """
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3
            )
            
            criteria_data = json.loads(response.choices[0].message.content)
            criteria = []
            
            for i, data in enumerate(criteria_data):
                criterion = AcceptanceCriteria(
                    id=f"AC-{story.id}-{i+1:02d}",
                    description=data['description'],
                    given=data['given'],
                    when=data['when'],
                    then=data['then']
                )
                criteria.append(criterion)
            
            return criteria
            
        except Exception as e:
            logger.warning(f"AI acceptance criteria generation failed: {e}")
            return []
    
    async def _generate_tasks_for_story(self, story: UserStory, requirement: Requirement) -> List[Task]:
        """Generate tasks for a user story."""
        tasks = []
        
        if self.openai_client:
            ai_tasks = await self._generate_tasks_with_ai(story, requirement)
            tasks.extend(ai_tasks)
        else:
            # Generate basic tasks
            basic_tasks = self._generate_basic_tasks(story)
            tasks.extend(basic_tasks)
        
        return tasks
    
    async def _generate_tasks_with_ai(self, story: UserStory, requirement: Requirement) -> List[Task]:
        """Generate tasks using AI."""
        try:
            prompt = f"""
            Break down this user story into 3-5 development tasks:
            
            Story: {story.title}
            Description: {story.description}
            Story Points: {story.story_points}
            
            Generate specific, actionable tasks that developers would work on.
            Include estimated hours for each task.
            
            Format as JSON array with objects containing: title, description, estimated_hours
            """
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            tasks_data = json.loads(response.choices[0].message.content)
            tasks = []
            
            for i, data in enumerate(tasks_data):
                task = Task(
                    id=f"TASK-{story.id}-{i+1:02d}",
                    title=data['title'],
                    description=data['description'],
                    estimated_hours=data['estimated_hours']
                )
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.warning(f"AI task generation failed: {e}")
            return []
    
    def _generate_basic_tasks(self, story: UserStory) -> List[Task]:
        """Generate basic tasks (fallback)."""
        base_hours = story.story_points * 2  # Rough conversion
        
        tasks = [
            Task(
                id=f"TASK-{story.id}-01",
                title="Implementation",
                description=f"Implement core functionality for {story.title}",
                estimated_hours=base_hours * 0.6
            ),
            Task(
                id=f"TASK-{story.id}-02",
                title="Testing",
                description=f"Write and execute tests for {story.title}",
                estimated_hours=base_hours * 0.3
            ),
            Task(
                id=f"TASK-{story.id}-03",
                title="Documentation",
                description=f"Document {story.title} functionality",
                estimated_hours=base_hours * 0.1
            )
        ]
        
        return tasks
    
    async def _generate_standalone_stories(self, requirements: List[Requirement]) -> List[UserStory]:
        """Generate standalone stories for unprocessed requirements."""
        stories = []
        
        for req in requirements:
            if self.openai_client:
                story_data = await self._generate_standalone_story_with_ai(req)
            else:
                story_data = self._generate_user_story_template(req)
            
            story = UserStory(
                id=f"US-{str(uuid.uuid4())[:8]}",
                title=story_data['title'],
                description=story_data['description'],
                story_type=StoryType.USER_STORY,
                priority=req.priority,
                story_points=story_data.get('story_points', 3),
                user_persona=story_data.get('user_persona', 'user'),
                business_value=story_data.get('business_value', 'provide functionality'),
                source_requirement=req.id
            )
            
            # Generate acceptance criteria and tasks
            story.acceptance_criteria = await self._generate_acceptance_criteria(req, story)
            story.tasks = await self._generate_tasks_for_story(story, req)
            
            stories.append(story)
        
        return stories
    
    async def _generate_standalone_story_with_ai(self, requirement: Requirement) -> Dict[str, Any]:
        """Generate standalone story using AI."""
        try:
            prompt = f"""
            Create a standalone user story from this requirement:
            
            Requirement: {requirement.text}
            Type: {requirement.requirement_type.value}
            Priority: {requirement.priority.value}
            
            Generate:
            1. Story title (clear, specific)
            2. Story description in "As a X, I want Y so that Z" format
            3. User persona (who benefits)
            4. Business value (why it matters)
            5. Story points estimate (1-8)
            
            Format as JSON with keys: title, description, user_persona, business_value, story_points
            """
            
            response = await self.openai_client.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            logger.warning(f"AI standalone story generation failed: {e}")
            return self._generate_user_story_template(requirement)
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from requirement text."""
        # Simple keyword extraction - could be enhanced with NLP
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'should', 'must', 'will', 'shall', 'can',
            'could', 'would', 'may', 'might', 'be', 'is', 'are', 'was', 'were', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'get', 'got', 'make', 'made'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        concepts = [word for word in words if len(word) > 3 and word not in common_words]
        
        # Return top concepts by frequency
        from collections import Counter
        concept_counts = Counter(concepts)
        return [concept for concept, _ in concept_counts.most_common(5)]
    
    def _extract_action_verb(self, text: str) -> str:
        """Extract action verb from requirement text."""
        action_patterns = [
            r'(?i)(create|add|build|implement|develop|generate)',
            r'(?i)(view|display|show|see|visualize)',
            r'(?i)(edit|update|modify|change|configure)',
            r'(?i)(delete|remove|clear|reset)',
            r'(?i)(search|find|filter|sort|query)',
            r'(?i)(send|receive|transmit|communicate)',
            r'(?i)(save|store|persist|backup)',
            r'(?i)(load|retrieve|fetch|import|export)'
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return "manage"  # Default action
    
    def _estimate_story_points(self, requirement: Requirement) -> int:
        """Estimate story points for a requirement."""
        base_points = 3  # Default
        
        # Adjust based on complexity factors
        complexity_factor = self.complexity_factors.get(requirement.requirement_type, 1.0)
        
        # Adjust based on priority
        priority_multiplier = {
            RequirementPriority.CRITICAL: 1.5,
            RequirementPriority.HIGH: 1.2,
            RequirementPriority.MEDIUM: 1.0,
            RequirementPriority.LOW: 0.8,
            RequirementPriority.NICE_TO_HAVE: 0.5
        }
        
        # Calculate estimate
        estimate = base_points * complexity_factor * priority_multiplier.get(requirement.priority, 1.0)
        
        # Round to nearest Fibonacci number
        if self.estimation_method == EstimationMethod.FIBONACCI:
            return min(self.fibonacci_scale, key=lambda x: abs(x - estimate))
        
        return max(1, min(21, int(round(estimate))))
    
    def _find_related_requirements(
        self, 
        feature: UserStory, 
        requirements: List[Requirement]
    ) -> List[Requirement]:
        """Find requirements related to a feature."""
        # Simple keyword matching - could be enhanced with semantic similarity
        feature_keywords = self._extract_key_concepts(feature.description)
        related = []
        
        for req in requirements:
            req_keywords = self._extract_key_concepts(req.text)
            
            # Check for keyword overlap
            overlap = set(feature_keywords) & set(req_keywords)
            if overlap and len(overlap) >= 1:
                related.append(req)
        
        return related[:5]  # Limit to prevent too many stories per feature
    
    def _get_unprocessed_requirements(
        self, 
        all_requirements: List[Requirement], 
        generated_stories: List[UserStory]
    ) -> List[Requirement]:
        """Get requirements that haven't been converted to stories."""
        processed_req_ids = {story.source_requirement for story in generated_stories if story.source_requirement}
        
        return [req for req in all_requirements if req.id not in processed_req_ids]
    
    def _parse_gherkin(self, text: str) -> Tuple[str, str, str]:
        """Parse Gherkin format from text."""
        given_match = re.search(r'(?i)given\s+([^.]+)', text)
        when_match = re.search(r'(?i)when\s+([^.]+)', text)
        then_match = re.search(r'(?i)then\s+([^.]+)', text)
        
        given = given_match.group(1).strip() if given_match else "the system is ready"
        when = when_match.group(1).strip() if when_match else "the user performs an action"
        then = then_match.group(1).strip() if then_match else "the expected result occurs"
        
        return given, when, then
    
    def _post_process_stories(self, stories: List[UserStory]) -> List[UserStory]:
        """Post-process generated stories for consistency."""
        # Validate story point distribution
        total_points = sum(story.story_points for story in stories)
        
        # Ensure epics have higher points than features
        epics = [s for s in stories if s.story_type == StoryType.EPIC]
        features = [s for s in stories if s.story_type == StoryType.FEATURE]
        user_stories = [s for s in stories if s.story_type == StoryType.USER_STORY]
        
        # Adjust if needed
        for epic in epics:
            if epic.story_points < 13:
                epic.story_points = 13
        
        for feature in features:
            if feature.story_points > 13:
                feature.story_points = 8
        
        for story in user_stories:
            if story.story_points > 8:
                story.story_points = 5
        
        return stories
    
    def export_stories(self, stories: List[UserStory], output_path: str, format: str = 'json'):
        """Export stories to various formats."""
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump([story.to_dict() for story in stories], f, indent=2)
        
        elif format.lower() == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'ID', 'Title', 'Type', 'Priority', 'Story Points', 
                    'User Persona', 'Description', 'Business Value'
                ])
                
                for story in stories:
                    writer.writerow([
                        story.id, story.title, story.story_type.value,
                        story.priority.value, story.story_points,
                        story.user_persona, story.description, story.business_value
                    ])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(stories)} stories to {output_path} in {format} format")
    
    def get_story_metrics(self, stories: List[UserStory]) -> Dict[str, Any]:
        """Get metrics about generated stories."""
        if not stories:
            return {}
        
        by_type = {}
        by_priority = {}
        total_points = 0
        total_hours = 0
        
        for story in stories:
            # Count by type
            story_type = story.story_type.value
            by_type[story_type] = by_type.get(story_type, 0) + 1
            
            # Count by priority
            priority = story.priority.value
            by_priority[priority] = by_priority.get(priority, 0) + 1
            
            # Sum points and hours
            total_points += story.story_points
            total_hours += story.estimate_total_hours()
        
        return {
            'total_stories': len(stories),
            'by_type': by_type,
            'by_priority': by_priority,
            'total_story_points': total_points,
            'total_estimated_hours': total_hours,
            'avg_story_points': total_points / len(stories),
            'avg_estimated_hours': total_hours / len(stories)
        }