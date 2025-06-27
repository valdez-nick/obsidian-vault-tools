"""
Content Generator

AI-powered content generation for PM deliverables:
- WBR/QBR narrative generation
- PRD section writing
- User story creation
- Executive summary generation
- Meeting notes summarization
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content that can be generated."""
    WBR_SUMMARY = "wbr_summary"
    QBR_NARRATIVE = "qbr_narrative"
    PRD_SECTION = "prd_section"
    USER_STORY = "user_story"
    EXECUTIVE_SUMMARY = "executive_summary"
    MEETING_NOTES = "meeting_notes"
    SPRINT_REPORT = "sprint_report"
    FEATURE_ANNOUNCEMENT = "feature_announcement"


@dataclass
class ContentTemplate:
    """
    Template for content generation.
    
    Attributes:
        type: Type of content
        structure: Expected structure/sections
        tone: Writing tone (formal, casual, technical)
        length: Target length in words
        audience: Target audience
        required_data: Required data fields
    """
    type: ContentType
    structure: List[str]
    tone: str = "professional"
    length: int = 500
    audience: str = "stakeholders"
    required_data: List[str] = field(default_factory=list)


@dataclass
class GeneratedContent:
    """
    Generated content result.
    
    Attributes:
        type: Type of content generated
        title: Content title
        sections: Dictionary of section names to content
        metadata: Generation metadata
        quality_score: Self-assessed quality (0-1)
        timestamp: When content was generated
    """
    type: ContentType
    title: str
    sections: Dict[str, str]
    metadata: Dict[str, Any]
    quality_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ContentGenerator:
    """
    Generates various types of PM content using AI.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the content generator.
        
        Args:
            config: Configuration including API keys and settings
        """
        self.config = config
        self.api_key = config.get('openai_api_key')
        self.model = config.get('model', 'gpt-4')
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[ContentType, ContentTemplate]:
        """Load content templates."""
        return {
            ContentType.WBR_SUMMARY: ContentTemplate(
                type=ContentType.WBR_SUMMARY,
                structure=["Highlights", "Metrics", "Challenges", "Next Steps"],
                tone="professional",
                length=800,
                audience="leadership",
                required_data=["metrics", "achievements", "blockers"]
            ),
            ContentType.USER_STORY: ContentTemplate(
                type=ContentType.USER_STORY,
                structure=["User Story", "Acceptance Criteria", "Technical Notes"],
                tone="technical",
                length=200,
                audience="engineering",
                required_data=["feature", "user_type", "goal"]
            )
        }
        
    async def generate_wbr_summary(self, data: Dict[str, Any]) -> GeneratedContent:
        """
        Generate a Weekly Business Review summary.
        
        Args:
            data: Dictionary containing metrics, achievements, etc.
            
        Returns:
            GeneratedContent with WBR summary
        """
        logger.info("Generating WBR summary")
        
        # Placeholder for WBR generation
        sections = {
            "Highlights": """
### Key Achievements
- Launched Feature X reaching 10K users in first week
- Reduced customer churn by 15% through targeted campaigns
- Completed migration to new infrastructure ahead of schedule

### Metrics Performance
- MAU: 1.2M (+8% WoW)
- Revenue: $450K (+12% WoW)
- NPS: 72 (+3 points)
            """,
            "Metrics": """
### Detailed Metrics

| Metric | This Week | Last Week | Change | Target |
|--------|-----------|-----------|---------|---------|
| MAU | 1.2M | 1.11M | +8.1% | 1.15M |
| Revenue | $450K | $402K | +11.9% | $425K |
| Conversion | 3.2% | 2.9% | +0.3pp | 3.0% |
| Churn | 2.1% | 2.5% | -0.4pp | 2.3% |
            """,
            "Challenges": """
### Current Challenges
1. **Mobile App Performance**: Crash rate increased to 1.2%
   - Root cause: Memory leak in image caching
   - Fix deployed, monitoring improvement

2. **Support Volume**: 20% increase in tickets
   - Mainly related to new feature confusion
   - Creating in-app tutorials and FAQ updates

3. **Infrastructure Costs**: 15% over budget
   - Due to unexpected traffic spike
   - Implementing auto-scaling optimizations
            """,
            "Next Steps": """
### Priority Actions for Next Week
1. Launch user onboarding flow improvements
2. Complete A/B test for pricing page redesign  
3. Ship mobile app performance fixes
4. Begin customer interview series for Q3 planning
5. Finalize partnership agreement with Provider X
            """
        }
        
        return GeneratedContent(
            type=ContentType.WBR_SUMMARY,
            title=f"Weekly Business Review - Week of {datetime.now().strftime('%B %d, %Y')}",
            sections=sections,
            metadata={"data_sources": list(data.keys())},
            quality_score=0.92
        )
        
    async def generate_user_stories(self, prd_content: str,
                                  max_stories: int = 10) -> List[GeneratedContent]:
        """
        Generate user stories from PRD content.
        
        Args:
            prd_content: PRD text to parse
            max_stories: Maximum number of stories to generate
            
        Returns:
            List of generated user stories
        """
        logger.info(f"Generating up to {max_stories} user stories from PRD")
        
        # Placeholder for story generation
        stories = []
        
        story_data = [
            {
                "feature": "User Authentication",
                "user_type": "new user",
                "goal": "quickly create an account"
            },
            {
                "feature": "Dashboard Analytics",
                "user_type": "product manager",
                "goal": "view key metrics at a glance"
            }
        ]
        
        for data in story_data[:max_stories]:
            sections = {
                "User Story": f"""
As a {data['user_type']},
I want to {data['goal']},
So that I can effectively use the product.
                """,
                "Acceptance Criteria": """
- User can complete action in under 3 clicks
- All required fields have clear labels
- Error messages are descriptive and actionable
- Success state is clearly indicated
- Mobile and desktop experiences are optimized
                """,
                "Technical Notes": """
- Implement using existing auth service
- Add analytics tracking for funnel analysis
- Ensure WCAG 2.1 AA compliance
- Cache user preferences locally
                """
            }
            
            stories.append(GeneratedContent(
                type=ContentType.USER_STORY,
                title=f"Story: {data['feature']}",
                sections=sections,
                metadata={"source": "prd", "priority": "high"},
                quality_score=0.88
            ))
            
        return stories
        
    async def generate_executive_summary(self, 
                                       reports: List[Dict[str, Any]],
                                       focus_points: List[str]) -> GeneratedContent:
        """
        Generate executive summary from multiple reports.
        
        Args:
            reports: List of report data
            focus_points: Key points to emphasize
            
        Returns:
            GeneratedContent with executive summary
        """
        logger.info(f"Generating executive summary with {len(focus_points)} focus points")
        
        # Placeholder for executive summary generation
        sections = {
            "Overview": """
## Executive Summary - Q2 2025

This quarter marked significant progress across all key metrics, 
with particularly strong performance in user acquisition and 
revenue growth. We successfully launched 3 major features while 
maintaining system stability and customer satisfaction.
            """,
            "Key Results": """
### Quarterly Achievements

**Financial Performance**
- Revenue: $5.2M (+32% QoQ)
- Gross Margin: 72% (+5pp QoQ)
- CAC: $45 (-12% QoQ)

**Product Metrics**
- MAU: 1.5M (+45% QoQ)
- Feature Adoption: 68% of users
- NPS: 74 (+6 points QoQ)

**Operational Excellence**
- Uptime: 99.95%
- Sprint Velocity: +20%
- Team Satisfaction: 8.5/10
            """,
            "Strategic Initiatives": """
### Strategic Progress

1. **Market Expansion**: Entered 3 new markets ahead of schedule
2. **Product Innovation**: AI features showing 2x engagement
3. **Partnership Program**: Signed 5 enterprise partners
4. **Team Growth**: Added 12 key hires across engineering and product
            """,
            "Outlook": """
### Q3 Outlook and Priorities

Based on current trajectory and market conditions, we project:
- Revenue: $6.5M-$7M
- MAU: 2M-2.2M
- New Features: 5 major releases planned

**Top Priorities:**
1. Scale infrastructure for growth
2. Launch mobile app v2.0
3. Expand enterprise offering
4. Strengthen data analytics capabilities
            """
        }
        
        return GeneratedContent(
            type=ContentType.EXECUTIVE_SUMMARY,
            title="Q2 2025 Executive Summary",
            sections=sections,
            metadata={"report_count": len(reports), "focus_points": focus_points},
            quality_score=0.95
        )
        
    async def enhance_content(self, content: str, 
                            enhancement_type: str = "clarity") -> str:
        """
        Enhance existing content for clarity, brevity, or impact.
        
        Args:
            content: Original content
            enhancement_type: Type of enhancement
            
        Returns:
            Enhanced content
        """
        logger.info(f"Enhancing content for {enhancement_type}")
        
        # Placeholder for content enhancement
        if enhancement_type == "clarity":
            return content.replace("utilize", "use").replace("implement", "build")
        elif enhancement_type == "brevity":
            return content[:int(len(content) * 0.8)]
        else:
            return content + "\n\n**Key Takeaway**: This drives significant business value."
            
    def validate_content(self, content: GeneratedContent) -> Dict[str, Any]:
        """
        Validate generated content for quality and completeness.
        
        Args:
            content: Content to validate
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check for required sections
        template = self.templates.get(content.type)
        if template:
            for required_section in template.structure:
                if required_section not in content.sections:
                    issues.append(f"Missing required section: {required_section}")
                    
        # Check content length
        total_length = sum(len(section) for section in content.sections.values())
        if total_length < 100:
            issues.append("Content too short")
            
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "quality_score": content.quality_score
        }