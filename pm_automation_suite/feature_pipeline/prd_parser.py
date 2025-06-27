"""
PRD Parser Implementation

Extracts and analyzes Product Requirements Documents to identify requirements,
features, and dependencies for automated story generation.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

try:
    import pypandoc
    PYPANDOC_AVAILABLE = True
except ImportError:
    PYPANDOC_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class RequirementType(Enum):
    """Types of requirements found in PRDs."""
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS = "business"
    TECHNICAL = "technical"
    USER_STORY = "user_story"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    DEPENDENCY = "dependency"
    ASSUMPTION = "assumption"


class RequirementPriority(Enum):
    """Priority levels for requirements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NICE_TO_HAVE = "nice_to_have"


@dataclass
class Requirement:
    """Individual requirement extracted from PRD."""
    id: str
    text: str
    requirement_type: RequirementType
    priority: RequirementPriority
    section: str
    dependencies: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    effort_estimate: Optional[int] = None  # Story points
    assignee: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'text': self.text,
            'type': self.requirement_type.value,
            'priority': self.priority.value,
            'section': self.section,
            'dependencies': self.dependencies,
            'acceptance_criteria': self.acceptance_criteria,
            'effort_estimate': self.effort_estimate,
            'assignee': self.assignee,
            'labels': self.labels
        }


@dataclass
class PRDMetadata:
    """Metadata extracted from PRD."""
    title: str
    version: str
    author: str
    created_date: Optional[datetime]
    last_modified: Optional[datetime]
    status: str
    stakeholders: List[str] = field(default_factory=list)
    target_release: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'version': self.version,
            'author': self.author,
            'created_date': self.created_date.isoformat() if self.created_date else None,
            'last_modified': self.last_modified.isoformat() if self.last_modified else None,
            'status': self.status,
            'stakeholders': self.stakeholders,
            'target_release': self.target_release
        }


@dataclass
class PRDContent:
    """Complete PRD content structure."""
    metadata: PRDMetadata
    requirements: List[Requirement]
    sections: Dict[str, str]  # section_name -> content
    raw_text: str
    validation_errors: List[str] = field(default_factory=list)
    
    def get_requirements_by_type(self, req_type: RequirementType) -> List[Requirement]:
        """Get requirements of specific type."""
        return [req for req in self.requirements if req.requirement_type == req_type]
    
    def get_requirements_by_priority(self, priority: RequirementPriority) -> List[Requirement]:
        """Get requirements of specific priority."""
        return [req for req in self.requirements if req.priority == priority]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metadata': self.metadata.to_dict(),
            'requirements': [req.to_dict() for req in self.requirements],
            'sections': self.sections,
            'validation_errors': self.validation_errors
        }


class PRDParser:
    """
    PRD Parser for extracting structured requirements from documents.
    
    Supports multiple input formats (Markdown, Word, PDF) and uses AI
    to intelligently extract and categorize requirements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PRD Parser.
        
        Args:
            config: Configuration with AI settings and parsing rules
        """
        self.config = config
        self.openai_client = None
        
        # Initialize AI client
        if OPENAI_AVAILABLE and 'openai_api_key' in config:
            openai.api_key = config['openai_api_key']
            self.openai_client = openai
        
        # Parsing rules and patterns
        self.section_patterns = {
            'overview': r'(?i)(overview|summary|executive summary)',
            'objectives': r'(?i)(objectives?|goals?|purpose)',
            'requirements': r'(?i)(requirements?|functional requirements?)',
            'user_stories': r'(?i)(user stories|stories|scenarios)',
            'acceptance_criteria': r'(?i)(acceptance criteria|success criteria)',
            'technical_requirements': r'(?i)(technical requirements?|architecture)',
            'dependencies': r'(?i)(dependencies|prerequisites)',
            'assumptions': r'(?i)(assumptions|constraints)',
            'timeline': r'(?i)(timeline|schedule|milestones)',
            'risks': r'(?i)(risks?|issues|concerns)'
        }
        
        self.requirement_patterns = {
            RequirementType.FUNCTIONAL: [
                r'(?i)the system (must|shall|should|will)',
                r'(?i)users? (must|shall|should|will) be able to',
                r'(?i)(feature|functionality|capability) (must|shall|should|will)',
            ],
            RequirementType.NON_FUNCTIONAL: [
                r'(?i)(performance|security|scalability|reliability)',
                r'(?i)(response time|throughput|availability)',
                r'(?i)(shall support|must handle) \d+',
            ],
            RequirementType.USER_STORY: [
                r'(?i)as an? .+, I want .+, so that',
                r'(?i)given .+, when .+, then',
            ]
        }
        
        self.priority_patterns = {
            RequirementPriority.CRITICAL: [r'(?i)(critical|must have|required)', r'(?i)p0'],
            RequirementPriority.HIGH: [r'(?i)(high|important|should have)', r'(?i)p1'],
            RequirementPriority.MEDIUM: [r'(?i)(medium|could have)', r'(?i)p2'],
            RequirementPriority.LOW: [r'(?i)(low|nice to have|optional)', r'(?i)p3'],
        }
        
        # Load validation rules
        self.validation_rules = self._load_validation_rules()
    
    def parse_prd(self, file_path: str) -> PRDContent:
        """
        Parse PRD from file.
        
        Args:
            file_path: Path to PRD file
            
        Returns:
            Parsed PRD content
        """
        logger.info(f"Parsing PRD: {file_path}")
        
        try:
            # Read file content
            raw_text = self._read_file(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(raw_text, file_path)
            
            # Parse sections
            sections = self._parse_sections(raw_text)
            
            # Extract requirements
            requirements = self._extract_requirements(raw_text, sections)
            
            # Enhance with AI if available
            if self.openai_client:
                requirements = self._enhance_with_ai(requirements, raw_text)
            
            # Create PRD content
            prd_content = PRDContent(
                metadata=metadata,
                requirements=requirements,
                sections=sections,
                raw_text=raw_text
            )
            
            # Validate content
            validation_errors = self._validate_prd(prd_content)
            prd_content.validation_errors = validation_errors
            
            logger.info(f"Parsed PRD with {len(requirements)} requirements, {len(validation_errors)} validation errors")
            return prd_content
            
        except Exception as e:
            logger.error(f"Failed to parse PRD: {e}")
            raise
    
    def _read_file(self, file_path: str) -> str:
        """Read content from various file formats."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PRD file not found: {file_path}")
        
        if path.suffix.lower() == '.md':
            return path.read_text(encoding='utf-8')
        
        elif path.suffix.lower() == '.txt':
            return path.read_text(encoding='utf-8')
        
        elif path.suffix.lower() == '.docx' and DOCX_AVAILABLE:
            doc = Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        elif path.suffix.lower() == '.pdf' and PYPANDOC_AVAILABLE:
            try:
                return pypandoc.convert_file(file_path, 'plain')
            except Exception as e:
                logger.warning(f"Failed to convert PDF: {e}")
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _extract_metadata(self, text: str, file_path: str) -> PRDMetadata:
        """Extract metadata from PRD text."""
        # Title extraction
        title_match = re.search(r'^#\s+(.+)', text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else Path(file_path).stem
        
        # Version extraction
        version_match = re.search(r'(?i)version[:\s]+([0-9.]+)', text)
        version = version_match.group(1) if version_match else "1.0"
        
        # Author extraction
        author_match = re.search(r'(?i)author[:\s]+([^\n]+)', text)
        author = author_match.group(1).strip() if author_match else "Unknown"
        
        # Status extraction
        status_match = re.search(r'(?i)status[:\s]+([^\n]+)', text)
        status = status_match.group(1).strip() if status_match else "Draft"
        
        # Stakeholders extraction
        stakeholders = []
        stakeholder_match = re.search(r'(?i)stakeholders?[:\s]+([^\n]+)', text)
        if stakeholder_match:
            stakeholders = [s.strip() for s in stakeholder_match.group(1).split(',')]
        
        # Target release extraction
        release_match = re.search(r'(?i)target[:\s]+([^\n]+)', text)
        target_release = release_match.group(1).strip() if release_match else None
        
        return PRDMetadata(
            title=title,
            version=version,
            author=author,
            created_date=datetime.now(),  # Would extract from file metadata in real implementation
            last_modified=datetime.now(),
            status=status,
            stakeholders=stakeholders,
            target_release=target_release
        )
    
    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Parse document into sections."""
        sections = {}
        
        # Split by headers (markdown style)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        
        current_section = "introduction"
        current_content = []
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                header_text = header_match.group(2).lower().strip()
                current_section = self._normalize_section_name(header_text)
                current_content = []
            else:
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _normalize_section_name(self, header_text: str) -> str:
        """Normalize section name using patterns."""
        for section_name, pattern in self.section_patterns.items():
            if re.search(pattern, header_text):
                return section_name
        
        # Default: clean the header text
        return re.sub(r'[^\w\s]', '', header_text).replace(' ', '_').lower()
    
    def _extract_requirements(self, text: str, sections: Dict[str, str]) -> List[Requirement]:
        """Extract requirements from text and sections."""
        requirements = []
        req_id_counter = 1
        
        # Extract from each section
        for section_name, section_content in sections.items():
            section_requirements = self._extract_requirements_from_section(
                section_content, section_name, req_id_counter
            )
            requirements.extend(section_requirements)
            req_id_counter += len(section_requirements)
        
        return requirements
    
    def _extract_requirements_from_section(
        self, 
        content: str, 
        section_name: str, 
        start_id: int
    ) -> List[Requirement]:
        """Extract requirements from a specific section."""
        requirements = []
        
        # Split content into sentences/paragraphs
        sentences = self._split_into_sentences(content)
        
        for i, sentence in enumerate(sentences):
            if self._is_requirement(sentence):
                req_type = self._classify_requirement_type(sentence)
                priority = self._extract_priority(sentence)
                
                # Extract acceptance criteria if present
                acceptance_criteria = self._extract_acceptance_criteria(sentence)
                
                requirement = Requirement(
                    id=f"REQ-{start_id + i:03d}",
                    text=sentence.strip(),
                    requirement_type=req_type,
                    priority=priority,
                    section=section_name,
                    acceptance_criteria=acceptance_criteria
                )
                
                requirements.append(requirement)
        
        return requirements
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into meaningful sentences/requirements."""
        # Simple sentence splitting - could be enhanced with NLP
        sentences = []
        
        # Split by bullet points and numbered lists
        bullet_pattern = r'^\s*[-*•]\s*(.+)'
        number_pattern = r'^\s*\d+\.\s*(.+)'
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for bullet point
            bullet_match = re.match(bullet_pattern, line)
            if bullet_match:
                sentences.append(bullet_match.group(1))
                continue
            
            # Check for numbered list
            number_match = re.match(number_pattern, line)
            if number_match:
                sentences.append(number_match.group(1))
                continue
            
            # Regular sentence
            if len(line) > 20:  # Filter out very short lines
                sentences.append(line)
        
        return sentences
    
    def _is_requirement(self, text: str) -> bool:
        """Determine if text contains a requirement."""
        # Check for requirement indicators
        requirement_indicators = [
            r'(?i)(must|shall|should|will|needs? to|requires?)',
            r'(?i)(the system|users?|application)',
            r'(?i)(feature|functionality|capability)',
            r'(?i)(as an?|given|when|then)',
            r'(?i)(acceptance criteria|success criteria)'
        ]
        
        for pattern in requirement_indicators:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _classify_requirement_type(self, text: str) -> RequirementType:
        """Classify the type of requirement."""
        for req_type, patterns in self.requirement_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return req_type
        
        return RequirementType.FUNCTIONAL  # Default
    
    def _extract_priority(self, text: str) -> RequirementPriority:
        """Extract priority from requirement text."""
        for priority, patterns in self.priority_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return priority
        
        return RequirementPriority.MEDIUM  # Default
    
    def _extract_acceptance_criteria(self, text: str) -> List[str]:
        """Extract acceptance criteria from requirement text."""
        criteria = []
        
        # Look for "given/when/then" patterns
        gherkin_pattern = r'(?i)(given|when|then)\s+([^.]+)'
        matches = re.findall(gherkin_pattern, text)
        
        for keyword, criterion in matches:
            criteria.append(f"{keyword.title()} {criterion.strip()}")
        
        return criteria
    
    def _enhance_with_ai(self, requirements: List[Requirement], raw_text: str) -> List[Requirement]:
        """Enhance requirements using AI analysis."""
        try:
            # Prepare prompt for AI analysis
            requirements_text = '\n'.join([req.text for req in requirements])
            
            prompt = f"""
            Analyze these requirements extracted from a PRD and enhance them:
            
            {requirements_text}
            
            For each requirement, provide:
            1. Effort estimate (story points 1-13)
            2. Additional labels/tags
            3. Dependencies on other requirements
            4. Missing acceptance criteria
            
            Format as JSON with requirement ID as key.
            """
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            
            enhancement_data = json.loads(response.choices[0].message.content)
            
            # Apply enhancements
            for req in requirements:
                if req.id in enhancement_data:
                    enhancement = enhancement_data[req.id]
                    req.effort_estimate = enhancement.get('effort_estimate')
                    req.labels.extend(enhancement.get('labels', []))
                    req.dependencies.extend(enhancement.get('dependencies', []))
                    req.acceptance_criteria.extend(enhancement.get('acceptance_criteria', []))
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
        
        return requirements
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load PRD validation rules."""
        return {
            'required_sections': ['overview', 'requirements'],
            'min_requirements': 1,
            'max_requirements': 100,
            'required_metadata': ['title', 'author', 'version'],
            'priority_distribution': {
                RequirementPriority.CRITICAL: (0, 0.2),  # 0-20%
                RequirementPriority.HIGH: (0.1, 0.4),    # 10-40%
                RequirementPriority.MEDIUM: (0.3, 0.6),  # 30-60%
            }
        }
    
    def _validate_prd(self, prd_content: PRDContent) -> List[str]:
        """Validate PRD content against rules."""
        errors = []
        
        # Check required sections
        for section in self.validation_rules['required_sections']:
            if section not in prd_content.sections:
                errors.append(f"Missing required section: {section}")
        
        # Check requirements count
        req_count = len(prd_content.requirements)
        if req_count < self.validation_rules['min_requirements']:
            errors.append(f"Too few requirements: {req_count} (minimum: {self.validation_rules['min_requirements']})")
        
        if req_count > self.validation_rules['max_requirements']:
            errors.append(f"Too many requirements: {req_count} (maximum: {self.validation_rules['max_requirements']})")
        
        # Check metadata completeness
        metadata_dict = prd_content.metadata.to_dict()
        for field in self.validation_rules['required_metadata']:
            if not metadata_dict.get(field):
                errors.append(f"Missing required metadata: {field}")
        
        # Check priority distribution
        if prd_content.requirements:
            priority_counts = {}
            for req in prd_content.requirements:
                priority_counts[req.priority] = priority_counts.get(req.priority, 0) + 1
            
            total_reqs = len(prd_content.requirements)
            for priority, (min_pct, max_pct) in self.validation_rules['priority_distribution'].items():
                actual_pct = priority_counts.get(priority, 0) / total_reqs
                if actual_pct < min_pct or actual_pct > max_pct:
                    errors.append(f"Priority distribution issue: {priority.value} is {actual_pct:.1%} (expected: {min_pct:.1%}-{max_pct:.1%})")
        
        return errors
    
    def analyze_requirements_complexity(self, requirements: List[Requirement]) -> Dict[str, Any]:
        """Analyze complexity of requirements."""
        analysis = {
            'total_requirements': len(requirements),
            'by_type': {},
            'by_priority': {},
            'by_section': {},
            'estimated_effort': 0,
            'complexity_score': 0
        }
        
        # Count by type
        for req in requirements:
            req_type = req.requirement_type.value
            analysis['by_type'][req_type] = analysis['by_type'].get(req_type, 0) + 1
            
            priority = req.priority.value
            analysis['by_priority'][priority] = analysis['by_priority'].get(priority, 0) + 1
            
            section = req.section
            analysis['by_section'][section] = analysis['by_section'].get(section, 0) + 1
            
            # Add effort estimate
            if req.effort_estimate:
                analysis['estimated_effort'] += req.effort_estimate
        
        # Calculate complexity score
        functional_reqs = analysis['by_type'].get('functional', 0)
        non_functional_reqs = analysis['by_type'].get('non_functional', 0)
        critical_reqs = analysis['by_priority'].get('critical', 0)
        
        analysis['complexity_score'] = (
            functional_reqs * 1.0 +
            non_functional_reqs * 1.5 +
            critical_reqs * 2.0
        ) / len(requirements) if requirements else 0
        
        return analysis
    
    def export_requirements(self, prd_content: PRDContent, output_path: str, format: str = 'json'):
        """Export requirements to various formats."""
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(prd_content.to_dict(), f, indent=2)
        
        elif format.lower() == 'csv':
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ID', 'Text', 'Type', 'Priority', 'Section', 'Effort'])
                
                for req in prd_content.requirements:
                    writer.writerow([
                        req.id, req.text, req.requirement_type.value,
                        req.priority.value, req.section, req.effort_estimate or ''
                    ])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported requirements to {output_path} in {format} format")