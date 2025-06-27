#!/usr/bin/env python3
"""
Content Quality Engine for PM Tools

This module provides content quality analysis and standardization
for Product Manager notes and documentation.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """Represents a content quality issue."""
    file_path: str
    issue_type: str
    description: str
    severity: str  # 'low', 'medium', 'high'
    suggestion: str
    line_number: Optional[int] = None


@dataclass
class QualityReport:
    """Contains the results of quality analysis."""
    overall_score: float
    total_files: int
    issues_found: List[QualityIssue]
    naming_inconsistencies: Dict[str, List[str]]
    incomplete_content: List[str]
    standardization_suggestions: List[str]
    metrics: Dict[str, Any]


class ContentQualityEngine:
    """
    Analyzes and improves content quality in Obsidian vaults for PM workflows.
    
    Features:
    - Naming consistency analysis
    - Incomplete content detection
    - Standardization suggestions
    - Quality scoring
    """
    
    def __init__(self, vault_path: str):
        """
        Initialize the Content Quality Engine.
        
        Args:
            vault_path: Path to the Obsidian vault
        """
        self.vault_path = Path(vault_path)
        self.issues = []
        self.metrics = {}
        
        # Common PM terminology and their standard forms
        self.pm_terminology = {
            'dfp': 'Device Fingerprinting',
            'device fingerprinting': 'Device Fingerprinting',
            'wsjf': 'WSJF',
            'weighted shortest job first': 'WSJF',
            'okr': 'OKR',
            'objectives key results': 'OKR',
            'kpi': 'KPI',
            'key performance indicator': 'KPI',
            'api': 'API',
            'application programming interface': 'API',
            'ui': 'UI',
            'user interface': 'UI',
            'ux': 'UX',
            'user experience': 'UX',
            'prd': 'PRD',
            'product requirements document': 'PRD',
            'mvp': 'MVP',
            'minimum viable product': 'MVP',
        }
        
        # Patterns for incomplete content
        self.incomplete_patterns = [
            r'TODO\s*:?\s*\w*',
            r'FIXME\s*:?\s*\w*',
            r'XXX\s*:?\s*\w*',
            r'\[placeholder\]',
            r'\[TBD\]',
            r'\[to be determined\]',
            r'\.{3,}',  # Multiple dots indicating incomplete thought
            r'- \s*$',  # Empty bullet points
            r'^#{1,6}\s*$',  # Empty headers
        ]
        
        # Patterns for quality issues
        self.quality_patterns = {
            'long_lines': r'.{120,}',  # Lines over 120 characters
            'duplicate_spaces': r'  +',  # Multiple consecutive spaces
            'trailing_whitespace': r' +$',
            'missing_periods': r'[a-z]\n',  # Lines ending without punctuation
            'inconsistent_headers': r'^#{1,6}[^ ]',  # Headers without space after #
        }

    def analyze_vault(self) -> QualityReport:
        """
        Analyze the entire vault for content quality issues.
        
        Returns:
            QualityReport containing analysis results
        """
        logger.info(f"Starting content quality analysis of {self.vault_path}")
        
        self.issues = []
        self.metrics = {
            'files_analyzed': 0,
            'total_lines': 0,
            'total_words': 0,
            'naming_issues': 0,
            'incomplete_items': 0,
            'quality_violations': 0,
        }
        
        # Find all markdown files
        markdown_files = list(self.vault_path.rglob("*.md"))
        
        naming_issues = defaultdict(list)
        incomplete_files = []
        
        for file_path in markdown_files:
            try:
                self._analyze_file(file_path, naming_issues, incomplete_files)
                self.metrics['files_analyzed'] += 1
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    issue_type="analysis_error",
                    description=f"Failed to analyze file: {e}",
                    severity="medium",
                    suggestion="Check file encoding and permissions"
                ))
        
        # Calculate overall score
        overall_score = self._calculate_overall_score()
        
        # Generate standardization suggestions
        suggestions = self._generate_standardization_suggestions(naming_issues)
        
        return QualityReport(
            overall_score=overall_score,
            total_files=len(markdown_files),
            issues_found=self.issues,
            naming_inconsistencies=dict(naming_issues),
            incomplete_content=incomplete_files,
            standardization_suggestions=suggestions,
            metrics=self.metrics
        )

    def _analyze_file(self, file_path: Path, naming_issues: Dict, incomplete_files: List):
        """Analyze a single file for quality issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    lines = content.split('\n')
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                return
        
        self.metrics['total_lines'] += len(lines)
        self.metrics['total_words'] += len(content.split())
        
        # Check for naming inconsistencies
        self._check_naming_consistency(file_path, content, naming_issues)
        
        # Check for incomplete content
        if self._check_incomplete_content(file_path, content, lines):
            incomplete_files.append(str(file_path))
        
        # Check for quality violations
        self._check_quality_violations(file_path, lines)

    def _check_naming_consistency(self, file_path: Path, content: str, naming_issues: Dict):
        """Check for inconsistent terminology usage."""
        content_lower = content.lower()
        
        # Group related terms
        term_groups = {}
        for term, standard in self.pm_terminology.items():
            if standard not in term_groups:
                term_groups[standard] = []
            term_groups[standard].append(term)
        
        # Find inconsistencies within each group
        for standard_term, variants in term_groups.items():
            found_variants = []
            for variant in variants:
                if variant in content_lower:
                    found_variants.append(variant)
            
            if len(found_variants) > 1:
                naming_issues[standard_term].extend(found_variants)
                self.metrics['naming_issues'] += 1
                
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    issue_type="naming_inconsistency",
                    description=f"Inconsistent terminology: {', '.join(found_variants)}",
                    severity="medium",
                    suggestion=f"Use standardized term: {standard_term}"
                ))

    def _check_incomplete_content(self, file_path: Path, content: str, lines: List[str]) -> bool:
        """Check for incomplete content markers."""
        has_incomplete = False
        
        for i, line in enumerate(lines, 1):
            for pattern in self.incomplete_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    has_incomplete = True
                    self.metrics['incomplete_items'] += 1
                    
                    self.issues.append(QualityIssue(
                        file_path=str(file_path),
                        issue_type="incomplete_content",
                        description=f"Incomplete content detected: {line.strip()}",
                        severity="high",
                        suggestion="Complete the incomplete content or remove placeholder",
                        line_number=i
                    ))
        
        return has_incomplete

    def _check_quality_violations(self, file_path: Path, lines: List[str]):
        """Check for general quality violations."""
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 120:
                self.metrics['quality_violations'] += 1
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    issue_type="long_line",
                    description=f"Line too long ({len(line)} characters)",
                    severity="low",
                    suggestion="Break long lines for better readability",
                    line_number=i
                ))
            
            # Check for duplicate spaces
            if re.search(r'  +', line):
                self.metrics['quality_violations'] += 1
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    issue_type="formatting",
                    description="Multiple consecutive spaces found",
                    severity="low",
                    suggestion="Use single spaces between words",
                    line_number=i
                ))
            
            # Check for trailing whitespace
            if re.search(r' +$', line):
                self.metrics['quality_violations'] += 1
                self.issues.append(QualityIssue(
                    file_path=str(file_path),
                    issue_type="formatting",
                    description="Trailing whitespace found",
                    severity="low",
                    suggestion="Remove trailing spaces",
                    line_number=i
                ))

    def _calculate_overall_score(self) -> float:
        """Calculate an overall quality score (0-100)."""
        if self.metrics['files_analyzed'] == 0:
            return 0.0
        
        # Base score starts at 100
        score = 100.0
        
        # Deduct points for various issues
        high_severity_count = sum(1 for issue in self.issues if issue.severity == "high")
        medium_severity_count = sum(1 for issue in self.issues if issue.severity == "medium")
        low_severity_count = sum(1 for issue in self.issues if issue.severity == "low")
        
        # Weight deductions by severity
        score -= (high_severity_count * 5)
        score -= (medium_severity_count * 2)
        score -= (low_severity_count * 0.5)
        
        # Ensure score doesn't go below 0
        return max(0.0, score)

    def _generate_standardization_suggestions(self, naming_issues: Dict) -> List[str]:
        """Generate suggestions for standardization improvements."""
        suggestions = []
        
        if naming_issues:
            suggestions.append("Standardize terminology usage across all documents:")
            for standard_term, variants in naming_issues.items():
                suggestions.append(f"  - Use '{standard_term}' instead of: {', '.join(set(variants))}")
        
        if self.metrics['incomplete_items'] > 0:
            suggestions.append(f"Complete {self.metrics['incomplete_items']} incomplete content items")
        
        if self.metrics['quality_violations'] > 0:
            suggestions.append("Fix formatting issues for better readability")
        
        suggestions.extend([
            "Consider using consistent header hierarchy",
            "Add frontmatter to important documents",
            "Use templates for recurring document types",
            "Implement regular quality reviews"
        ])
        
        return suggestions

    def fix_naming_inconsistencies(self, file_path: str, dry_run: bool = True) -> List[str]:
        """
        Fix naming inconsistencies in a specific file.
        
        Args:
            file_path: Path to the file to fix
            dry_run: If True, return changes without applying them
            
        Returns:
            List of changes made or to be made
        """
        changes = []
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply terminology standardization
            for term, standard in self.pm_terminology.items():
                if term != standard.lower():
                    # Use word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(term) + r'\b'
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    
                    if matches:
                        content = re.sub(pattern, standard, content, flags=re.IGNORECASE)
                        changes.append(f"'{term}' → '{standard}' ({len(matches)} occurrences)")
            
            if not dry_run and content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                changes.append("Changes applied to file")
            
        except Exception as e:
            logger.error(f"Error fixing {file_path}: {e}")
            changes.append(f"Error: {e}")
        
        return changes

    def generate_quality_report(self, report: QualityReport, output_path: Optional[str] = None) -> str:
        """
        Generate a formatted quality report.
        
        Args:
            report: QualityReport to format
            output_path: Optional path to save the report
            
        Returns:
            Formatted report as string
        """
        report_lines = [
            "# Content Quality Report",
            f"Generated for vault: {self.vault_path}",
            "",
            "## Summary",
            f"- Overall Score: {report.overall_score:.1f}/100",
            f"- Files Analyzed: {report.total_files}",
            f"- Issues Found: {len(report.issues_found)}",
            "",
            "## Metrics",
            f"- Total Lines: {report.metrics.get('total_lines', 0):,}",
            f"- Total Words: {report.metrics.get('total_words', 0):,}",
            f"- Naming Issues: {report.metrics.get('naming_issues', 0)}",
            f"- Incomplete Items: {report.metrics.get('incomplete_items', 0)}",
            f"- Quality Violations: {report.metrics.get('quality_violations', 0)}",
            "",
        ]
        
        if report.naming_inconsistencies:
            report_lines.extend([
                "## Naming Inconsistencies",
                ""
            ])
            for standard_term, variants in report.naming_inconsistencies.items():
                report_lines.append(f"**{standard_term}**: {', '.join(set(variants))}")
            report_lines.append("")
        
        if report.incomplete_content:
            report_lines.extend([
                "## Files with Incomplete Content",
                ""
            ])
            for file_path in report.incomplete_content[:10]:  # Limit to first 10
                report_lines.append(f"- {file_path}")
            if len(report.incomplete_content) > 10:
                report_lines.append(f"- ... and {len(report.incomplete_content) - 10} more")
            report_lines.append("")
        
        if report.standardization_suggestions:
            report_lines.extend([
                "## Standardization Suggestions",
                ""
            ])
            for suggestion in report.standardization_suggestions:
                report_lines.append(f"- {suggestion}")
            report_lines.append("")
        
        # Top issues by severity
        high_issues = [i for i in report.issues_found if i.severity == "high"]
        if high_issues:
            report_lines.extend([
                "## High Priority Issues",
                ""
            ])
            for issue in high_issues[:5]:  # Show top 5
                report_lines.extend([
                    f"**{issue.file_path}** (Line {issue.line_number or 'N/A'})",
                    f"- Issue: {issue.description}",
                    f"- Suggestion: {issue.suggestion}",
                    ""
                ])
        
        report_text = "\n".join(report_lines)
        
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"Quality report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report_text


def main():
    """CLI entry point for the Content Quality Engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze content quality in Obsidian vault")
    parser.add_argument("vault_path", help="Path to Obsidian vault")
    parser.add_argument("--output", "-o", help="Output file for quality report")
    parser.add_argument("--fix", "-f", help="Fix naming inconsistencies in specific file")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying them")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    engine = ContentQualityEngine(args.vault_path)
    
    if args.fix:
        changes = engine.fix_naming_inconsistencies(args.fix, dry_run=args.dry_run)
        print(f"Changes for {args.fix}:")
        for change in changes:
            print(f"  - {change}")
    else:
        report = engine.analyze_vault()
        report_text = engine.generate_quality_report(report, args.output)
        print(report_text)


if __name__ == "__main__":
    main()