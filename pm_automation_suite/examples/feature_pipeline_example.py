"""
Feature Development Pipeline Example

Demonstrates complete workflow from PRD analysis to Jira story creation
with real-world configuration and usage patterns.
"""

import asyncio
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Import pipeline components
from feature_pipeline.feature_pipeline import FeaturePipeline, PipelineStage
from feature_pipeline.prd_parser import RequirementPriority


async def create_sample_prd():
    """Create a sample PRD file for demonstration."""
    prd_content = """
# Mobile App User Onboarding Experience

## Metadata
**Author:** Product Team
**Version:** 1.2
**Status:** Ready for Development
**Stakeholders:** Engineering, Design, Product, Marketing
**Target Release:** Q2 2024

## Overview
Design and implement a comprehensive user onboarding experience for our mobile application
that increases user activation rates and reduces time-to-value for new users.

## Business Objectives
- Increase user activation rate from 45% to 70%
- Reduce time-to-first-value from 5 minutes to 2 minutes
- Decrease support tickets related to onboarding by 50%
- Improve user retention at Day 7 from 25% to 40%

## Functional Requirements

### Account Creation & Verification
- The system must allow users to register using email, phone, or social logins
- Users shall receive verification codes via SMS or email within 30 seconds
- The system should support OAuth integration with Google, Facebook, and Apple
- Users must be able to create accounts with minimal required information

### Onboarding Flow
- The system shall provide a progressive disclosure onboarding experience
- Users should see personalized content based on their selected use case
- The system must collect user preferences during the first session
- Users should be able to skip non-essential onboarding steps
- The system shall provide contextual tooltips and guidance

### Tutorial System
- The system must include interactive tutorials for core features
- Users should be able to replay tutorials at any time
- The system shall track tutorial completion rates per step
- Tutorials must be accessible and support multiple languages

### Profile Setup
- Users must be able to upload profile pictures during onboarding
- The system should suggest connections based on contacts or social networks
- Users shall be able to customize their notification preferences
- The system must validate profile information in real-time

## Non-Functional Requirements

### Performance
- Onboarding screens must load within 2 seconds on 3G networks
- The system should support 10,000 concurrent new user registrations
- Tutorial videos must start playing within 3 seconds of selection
- Profile picture uploads should process within 5 seconds

### Usability
- The onboarding flow should be completable in under 3 minutes
- Users must be able to navigate backwards through onboarding steps
- The system shall maintain progress if users exit and return
- Interface must be accessible and WCAG 2.1 AA compliant

### Reliability
- Account creation success rate must be above 99%
- The system should gracefully handle network interruptions
- User progress must be automatically saved at each step
- Verification systems should have 99.5% uptime

## User Stories & Acceptance Criteria

### Epic: Streamlined Registration
As a new user, I want to quickly create an account so that I can start using the app immediately.

**Acceptance Criteria:**
- Given I am on the registration screen, when I enter valid information, then my account is created within 10 seconds
- Given I choose social login, when I authenticate successfully, then I am automatically logged into the app
- Given I enter an email address, when it already exists, then I see a clear error message with next steps

### Epic: Personalized Onboarding
As a new user, I want the app to understand my needs so that I see relevant features and content.

**Acceptance Criteria:**
- Given I select my primary use case, when I proceed through onboarding, then I see customized examples and features
- Given I complete preference selection, when I reach the main app, then my dashboard reflects my choices
- Given I want to change my preferences, when I access settings, then I can modify my onboarding choices

### Epic: Interactive Learning
As a new user, I want to learn how to use the app effectively so that I can achieve my goals quickly.

**Acceptance Criteria:**
- Given I start a tutorial, when I follow the steps, then I can practice with real app functionality
- Given I complete a tutorial, when I access that feature later, then I have contextual hints available
- Given I skip a tutorial, when I encounter that feature, then I can access help easily

## Technical Requirements

### Platform Support
- iOS 14+ and Android 8+ compatibility required
- Web app responsive design for tablet/desktop access
- Offline mode support for core onboarding steps
- Cross-platform progress synchronization

### Analytics & Tracking
- The system must track conversion rates at each onboarding step
- User behavior analytics should be collected with privacy compliance
- A/B testing framework integration is required for onboarding optimization
- Real-time dashboards for onboarding metrics monitoring

### Security & Privacy
- All user data must be encrypted in transit and at rest
- GDPR and CCPA compliance required for data collection
- Users must provide explicit consent for data usage
- Account verification must prevent automated bot registrations

## Dependencies & Constraints

### External Dependencies
- Email service provider (SendGrid or similar) for verification emails
- SMS gateway (Twilio) for phone verification
- Social login SDKs (Google, Facebook, Apple)
- Analytics platform integration (Mixpanel or Amplitude)

### Technical Constraints
- Must integrate with existing user management system
- Cannot exceed 2MB additional app size for onboarding assets
- Must maintain compatibility with current API architecture
- Limited to 5 additional database tables for onboarding data

### Business Constraints
- Development budget: $50,000
- Timeline: 8 weeks for MVP, 12 weeks for full feature set
- Must comply with app store review guidelines
- Cannot collect more user data than existing privacy policy allows

## Success Metrics

### Primary KPIs
- User activation rate (target: 70%)
- Time to first value (target: 2 minutes)
- Day 7 retention rate (target: 40%)
- Onboarding completion rate (target: 85%)

### Secondary Metrics
- Tutorial completion rates by step
- Support ticket reduction related to setup
- User satisfaction scores for onboarding experience
- Feature adoption rates within first week

## Assumptions & Risks

### Assumptions
- Users prefer progressive disclosure over comprehensive upfront setup
- Social login adoption will be high (>60% of registrations)
- Current app performance allows for additional onboarding features
- Users will engage with interactive tutorials over static help content

### Risks
- iOS/Android platform changes affecting social login integrations
- Increased app complexity may impact performance on older devices
- User preference variations across different demographics
- Potential conflicts with existing user management workflows

## Future Considerations
- AI-powered personalization based on user behavior patterns
- Gamification elements to increase onboarding engagement
- Video-based tutorials with closed captions
- Integration with customer success platform for follow-up
"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(prd_content)
        return f.name


def create_pipeline_config():
    """Create pipeline configuration."""
    return {
        'pipeline_id': f'mobile_onboarding_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'output_dir': './feature_pipeline_outputs',
        
        # PRD Parser Configuration
        'prd_parser': {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),  # Set your API key
            'enhance_with_ai': True,
            'validation_strict': False
        },
        
        # Story Generator Configuration  
        'story_generator': {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'ai_provider': 'openai',  # or 'anthropic'
            'model': 'gpt-4',
            'max_stories_per_epic': 8,
            'include_acceptance_criteria': True,
            'include_task_breakdown': True,
            'estimation_method': 'fibonacci'
        },
        
        # Jira Bulk Creator Configuration
        'jira_bulk_creator': {
            'jira_project_key': 'MOB',  # Your Jira project key
            'dry_run': True,  # Set to False for actual creation
            'batch_size': 5,
            'rate_limit_delay': 1.0,
            
            # Jira Connection (set environment variables)
            'jira': {
                'server': os.getenv('JIRA_SERVER', 'https://your-domain.atlassian.net'),
                'username': os.getenv('JIRA_USERNAME'),
                'api_token': os.getenv('JIRA_API_TOKEN')
            },
            
            # Custom Field Mapping (adjust for your Jira instance)
            'field_mapping': {
                'story_points_field': 'customfield_10016',
                'epic_link_field': 'customfield_10014',
                'priority_field': 'priority',
                'assignee_field': 'assignee',
                'labels_field': 'labels'
            }
        },
        
        # Pipeline Settings
        'validate_at_each_stage': True,
        'save_artifacts': True,
        'continue_on_warnings': True,
        'max_retries': 2
    }


async def pipeline_progress_callback(stage, data):
    """Handle pipeline progress updates."""
    status = data.get('status', 'unknown')
    
    if status == 'started':
        print(f"🚀 Starting {stage.value.replace('_', ' ').title()}")
    elif status == 'completed':
        duration = data.get('duration', 0)
        print(f"✅ Completed {stage.value.replace('_', ' ').title()} in {duration:.2f}s")
    elif status == 'failed':
        error = data.get('error', 'Unknown error')
        print(f"❌ Failed {stage.value.replace('_', ' ').title()}: {error}")
    else:
        print(f"📊 {stage.value.replace('_', ' ').title()}: {status}")


async def run_complete_pipeline_example():
    """Run complete pipeline example with detailed output."""
    print("=" * 60)
    print("Feature Development Pipeline - Complete Example")
    print("=" * 60)
    
    # Create sample PRD
    print("\n📝 Creating sample PRD file...")
    prd_file_path = await create_sample_prd()
    print(f"Created PRD: {prd_file_path}")
    
    try:
        # Create pipeline configuration
        print("\n⚙️  Configuring pipeline...")
        config = create_pipeline_config()
        
        # Initialize pipeline
        pipeline = FeaturePipeline(config)
        pipeline.add_progress_callback(pipeline_progress_callback)
        
        print(f"Pipeline ID: {pipeline.pipeline_id}")
        print(f"Output Directory: {pipeline.output_dir}")
        
        # Execute pipeline
        print("\n🔄 Executing pipeline...")
        result = await pipeline.execute_pipeline(prd_file_path)
        
        # Display results
        print("\n" + "=" * 40)
        print("PIPELINE EXECUTION RESULTS")
        print("=" * 40)
        
        print(f"Status: {result.status.value.upper()}")
        
        if result.metrics:
            print(f"Total Duration: {result.metrics.duration_seconds:.2f} seconds")
            print("\nStage Durations:")
            for stage, duration in result.metrics.stage_durations.items():
                print(f"  {stage.replace('_', ' ').title()}: {duration:.2f}s")
        
        # PRD Analysis Results
        if result.prd_content:
            print(f"\n📋 PRD Analysis:")
            print(f"  Title: {result.prd_content.metadata.title}")
            print(f"  Author: {result.prd_content.metadata.author}")
            print(f"  Requirements Found: {len(result.prd_content.requirements)}")
            print(f"  Validation Errors: {len(result.prd_content.validation_errors)}")
            
            # Show requirement breakdown
            req_by_type = {}
            req_by_priority = {}
            for req in result.prd_content.requirements:
                req_type = req.requirement_type.value
                req_priority = req.priority.value
                req_by_type[req_type] = req_by_type.get(req_type, 0) + 1
                req_by_priority[req_priority] = req_by_priority.get(req_priority, 0) + 1
            
            print("  By Type:", req_by_type)
            print("  By Priority:", req_by_priority)
        
        # Story Generation Results
        if result.generated_stories:
            print(f"\n📖 Story Generation:")
            print(f"  Total Stories: {len(result.generated_stories)}")
            
            # Show story breakdown
            story_by_type = {}
            for story in result.generated_stories:
                story_type = story.story_type.value
                story_by_type[story_type] = story_by_type.get(story_type, 0) + 1
            
            print("  By Type:", story_by_type)
            
            # Show first few stories as examples
            print("\n  Example Stories:")
            for i, story in enumerate(result.generated_stories[:3]):
                print(f"    {i+1}. [{story.story_type.value.upper()}] {story.title}")
                print(f"       Priority: {story.priority.value}")
                if story.acceptance_criteria:
                    print(f"       Acceptance Criteria: {len(story.acceptance_criteria)}")
        
        # Jira Creation Results  
        if result.jira_hierarchy:
            print(f"\n🎫 Jira Creation:")
            summary = result.jira_hierarchy.creation_summary
            print(f"  Project: {summary.get('project_key', 'Unknown')}")
            print(f"  Total Issues: {summary.get('total_issues_created', 0)}")
            print(f"  Epics: {summary.get('epics_created', 0)}")
            print(f"  Features: {summary.get('features_created', 0)}")
            print(f"  Stories: {summary.get('stories_created', 0)}")
            print(f"  Tasks: {summary.get('tasks_created', 0)}")
            print(f"  Mode: {'DRY RUN' if summary.get('dry_run') else 'LIVE CREATION'}")
            
            if summary.get('epic_keys'):
                print(f"  Epic Keys: {', '.join(summary['epic_keys'])}")
        
        # Errors and Warnings
        if result.errors:
            print(f"\n❌ Errors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  - {error}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        # Generated Artifacts
        if result.artifacts:
            print(f"\n📁 Generated Artifacts:")
            for name, path in result.artifacts.items():
                print(f"  {name}: {path}")
        
        # Generate comprehensive report
        print("\n📊 Generating comprehensive report...")
        report = pipeline.generate_pipeline_report(result)
        
        # Save report to file
        report_path = Path(pipeline.output_dir) / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        if os.path.exists(prd_file_path):
            os.unlink(prd_file_path)
            print(f"\n🧹 Cleaned up temporary PRD file")


async def run_individual_component_examples():
    """Run examples for individual pipeline components."""
    print("\n" + "=" * 60)
    print("Individual Component Examples")
    print("=" * 60)
    
    # PRD Parser Example
    print("\n1. 📝 PRD Parser Example")
    print("-" * 30)
    
    from feature_pipeline.prd_parser import PRDParser
    
    # Create simple PRD for testing
    simple_prd = """
# Simple Feature

## Requirements
- Users must be able to login
- The system should be fast
- Critical: Data must be secure
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(simple_prd)
        simple_prd_path = f.name
    
    try:
        parser = PRDParser({})
        prd_content = parser.parse_prd(simple_prd_path)
        
        print(f"Title: {prd_content.metadata.title}")
        print(f"Requirements: {len(prd_content.requirements)}")
        for req in prd_content.requirements:
            print(f"  - [{req.requirement_type.value}] {req.text}")
            print(f"    Priority: {req.priority.value}")
        
        # Analyze complexity
        analysis = parser.analyze_requirements_complexity(prd_content.requirements)
        print(f"Complexity Analysis: {analysis}")
        
    finally:
        os.unlink(simple_prd_path)
    
    # Story Generator Example
    print("\n2. 📖 Story Generator Example")
    print("-" * 30)
    
    from feature_pipeline.story_generator import StoryGenerator
    
    # Use the PRD content from above
    story_generator = StoryGenerator({})
    
    # Mock story generation (without AI)
    print("Story generation would create user stories from requirements...")
    print("Example output structure:")
    
    example_story = {
        "id": "STORY-001",
        "title": "User Login",
        "description": "As a user, I want to login securely",
        "type": "user_story",
        "priority": "high",
        "acceptance_criteria": [
            "Given valid credentials, when I login, then I access the dashboard"
        ]
    }
    
    print(json.dumps(example_story, indent=2))
    
    # Jira Bulk Creator Example
    print("\n3. 🎫 Jira Bulk Creator Example")
    print("-" * 30)
    
    from feature_pipeline.jira_bulk_creator import JiraBulkCreator
    
    jira_config = {
        'jira_project_key': 'DEMO',
        'dry_run': True  # Always dry run for examples
    }
    
    jira_creator = JiraBulkCreator(jira_config)
    
    # Validate configuration
    validation = jira_creator.validate_project_configuration()
    print(f"Jira Configuration Valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    print("Jira bulk creation would create issues from user stories...")
    print("Example creation summary:")
    
    example_summary = {
        "total_issues_created": 5,
        "epics_created": 1,
        "features_created": 2,
        "stories_created": 2,
        "tasks_created": 0,
        "dry_run": True
    }
    
    print(json.dumps(example_summary, indent=2))


async def run_configuration_examples():
    """Show various configuration examples."""
    print("\n" + "=" * 60)
    print("Configuration Examples")
    print("=" * 60)
    
    # Development Configuration
    print("\n1. 🛠️  Development Configuration")
    print("-" * 30)
    
    dev_config = {
        'pipeline_id': 'dev_pipeline',
        'output_dir': './dev_outputs',
        'prd_parser': {
            'enhance_with_ai': False,  # Skip AI for faster dev cycles
            'validation_strict': False
        },
        'story_generator': {
            'max_stories_per_epic': 3,  # Limit for dev testing
            'include_task_breakdown': False
        },
        'jira_bulk_creator': {
            'dry_run': True,  # Never create real issues in dev
            'batch_size': 2
        },
        'validate_at_each_stage': False,  # Speed up dev cycles
        'save_artifacts': True
    }
    
    print("Development config (fast, safe):")
    print(json.dumps(dev_config, indent=2))
    
    # Production Configuration
    print("\n2. 🚀 Production Configuration")
    print("-" * 30)
    
    prod_config = {
        'pipeline_id': 'prod_pipeline',
        'output_dir': '/var/pipeline_outputs',
        'prd_parser': {
            'openai_api_key': '${OPENAI_API_KEY}',
            'enhance_with_ai': True,
            'validation_strict': True
        },
        'story_generator': {
            'openai_api_key': '${OPENAI_API_KEY}',
            'ai_provider': 'openai',
            'model': 'gpt-4',
            'max_stories_per_epic': 12,
            'include_acceptance_criteria': True,
            'include_task_breakdown': True
        },
        'jira_bulk_creator': {
            'jira_project_key': 'PROD',
            'dry_run': False,  # Real creation
            'batch_size': 10,
            'rate_limit_delay': 0.5,
            'jira': {
                'server': '${JIRA_SERVER}',
                'username': '${JIRA_USERNAME}',
                'api_token': '${JIRA_API_TOKEN}'
            }
        },
        'validate_at_each_stage': True,
        'save_artifacts': True,
        'continue_on_warnings': False,  # Strict mode
        'max_retries': 3
    }
    
    print("Production config (comprehensive, validated):")
    print(json.dumps(prod_config, indent=2))
    
    # Enterprise Configuration
    print("\n3. 🏢 Enterprise Configuration")
    print("-" * 30)
    
    enterprise_config = {
        'pipeline_id': 'enterprise_pipeline',
        'output_dir': '/enterprise/pipeline_outputs',
        'prd_parser': {
            'anthropic_api_key': '${ANTHROPIC_API_KEY}',
            'enhance_with_ai': True,
            'ai_provider': 'anthropic',
            'validation_strict': True
        },
        'story_generator': {
            'anthropic_api_key': '${ANTHROPIC_API_KEY}',
            'ai_provider': 'anthropic',
            'model': 'claude-3-opus',
            'max_stories_per_epic': 15,
            'include_acceptance_criteria': True,
            'include_task_breakdown': True,
            'estimation_method': 'fibonacci'
        },
        'jira_bulk_creator': {
            'jira_project_key': 'ENT',
            'dry_run': False,
            'batch_size': 20,
            'rate_limit_delay': 0.2,
            'jira': {
                'server': '${ENTERPRISE_JIRA_SERVER}',
                'username': '${JIRA_SERVICE_ACCOUNT}',
                'api_token': '${JIRA_SERVICE_TOKEN}'
            },
            'field_mapping': {
                'story_points_field': 'customfield_10020',
                'epic_link_field': 'customfield_10018',
                'team_field': 'customfield_10025',
                'priority_field': 'priority'
            }
        },
        'validate_at_each_stage': True,
        'save_artifacts': True,
        'continue_on_warnings': False,
        'max_retries': 5
    }
    
    print("Enterprise config (high-scale, advanced features):")
    print(json.dumps(enterprise_config, indent=2))


async def main():
    """Main example execution."""
    print("Feature Development Pipeline - Examples & Demonstrations")
    print("=" * 70)
    
    # Check for API keys
    api_keys_available = bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))
    jira_configured = bool(os.getenv('JIRA_SERVER') and os.getenv('JIRA_USERNAME') and os.getenv('JIRA_API_TOKEN'))
    
    print(f"\n🔑 API Keys Available: {api_keys_available}")
    print(f"🎫 Jira Configured: {jira_configured}")
    
    if not api_keys_available:
        print("\n⚠️  Note: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables for full functionality")
    
    if not jira_configured:
        print("\n⚠️  Note: Set JIRA_SERVER, JIRA_USERNAME, JIRA_API_TOKEN for Jira integration")
    
    try:
        # Run individual component examples
        await run_individual_component_examples()
        
        # Show configuration examples
        await run_configuration_examples()
        
        # Run complete pipeline if APIs are available
        if api_keys_available:
            print(f"\n🚀 Running complete pipeline example...")
            result = await run_complete_pipeline_example()
            
            if result and result.status.value == 'completed':
                print(f"\n✅ Pipeline example completed successfully!")
            else:
                print(f"\n❌ Pipeline example failed or was incomplete")
        else:
            print(f"\n⏭️  Skipping complete pipeline example (API keys required)")
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Example execution interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n👋 Example execution completed")


if __name__ == "__main__":
    asyncio.run(main())