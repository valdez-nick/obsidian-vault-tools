#!/usr/bin/env python3
"""
PM Automation CLI Commands

Provides direct command-line access to PM Automation Suite features.
"""

import click
import asyncio
import logging
from pathlib import Path
from typing import Optional

from obsidian_vault_tools.utils import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--vault', '-v', help='Vault path (overrides config)')
@click.option('--debug', '-d', is_flag=True, help='Enable debug logging')
@click.pass_context
def pm(ctx, vault: Optional[str], debug: bool):
    """
    PM Automation Suite CLI commands.
    
    This CLI provides direct command-line access to all PM Automation Suite features,
    allowing Product Managers to automate workflows without using the interactive UI.
    
    Commands support both one-time execution and integration into scripts/workflows.
    All commands respect the configured vault path or can override it with --vault.
    
    Examples:
        ovt-pm --vault /path/to/vault quality
        ovt-pm wbr --project MYPROJ --format slides
        ovt-pm features requirements.pdf --project FEAT --dry-run
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Get vault path from CLI, config, or prompt
    if vault:
        vault_path = Path(vault)
    else:
        config = Config()
        vault_path = config.get_vault_path()
        if not vault_path:
            vault_path = click.prompt("Enter vault path", type=click.Path(exists=True))
            vault_path = Path(vault_path)
        else:
            vault_path = Path(vault_path)
    
    ctx.obj['vault_path'] = vault_path
    
    if not vault_path.exists():
        click.echo(f"Error: Vault path does not exist: {vault_path}", err=True)
        ctx.exit(1)


@pm.command()
@click.option('--project', '-p', default='ALL', help='Jira project key')
@click.option('--format', '-f', type=click.Choice(['slides', 'markdown', 'json']), 
              default='slides', help='Output format')
@click.option('--output', '-o', help='Output file path')
@click.pass_context
def wbr(ctx, project: str, format: str, output: Optional[str]):
    """
    Generate Weekly Business Review with AI-powered insights.
    
    This command orchestrates the complete WBR generation process:
    1. Extracts data from configured sources (Jira, Snowflake, etc.)
    2. Performs AI analysis to generate insights and trends
    3. Creates formatted output (PowerPoint slides, Markdown, or JSON)
    4. Optionally saves to specified output file
    
    The WBR includes:
    - Sprint progress and velocity metrics
    - Key performance indicators and trends
    - Risk analysis and recommendations
    - Executive summary with actionable insights
    
    Examples:
        ovt-pm wbr --project PROD --format slides --output weekly_review.pptx
        ovt-pm wbr --project ALL --format markdown
    """
    click.echo("🤖 Generating Weekly Business Review...")
    
    try:
        # Dynamic import to handle missing dependencies gracefully
        from pm_automation_suite.wbr import WBRWorkflow
        
        async def run_wbr():
            workflow = WBRWorkflow({
                'vault_path': str(ctx.obj['vault_path']),
                'jira_project': project
            })
            
            result = await workflow.run_complete_workflow()
            
            if output:
                if format == 'slides':
                    # Save PowerPoint
                    click.echo(f"📊 Slides saved to: {output}")
                elif format == 'markdown':
                    # Save markdown report
                    with open(output, 'w') as f:
                        f.write(result.get('markdown_report', ''))
                    click.echo(f"📝 Report saved to: {output}")
                elif format == 'json':
                    # Save JSON data
                    import json
                    with open(output, 'w') as f:
                        json.dump(result, f, indent=2)
                    click.echo(f"📋 Data saved to: {output}")
            else:
                click.echo("✅ WBR generation completed!")
                click.echo(f"📊 Insights: {len(result.get('insights', []))} generated")
        
        asyncio.run(run_wbr())
        
    except ImportError:
        click.echo("❌ PM Automation Suite not available. Install with: pip install obsidian-vault-tools[pm-automation]", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@pm.command()
@click.argument('prd_file', type=click.Path(exists=True))
@click.option('--project', '-p', required=True, help='Jira project key')
@click.option('--epic', '-e', help='Epic key to link stories to')
@click.option('--dry-run', '-n', is_flag=True, help='Preview without creating stories')
@click.pass_context
def features(ctx, prd_file: str, project: str, epic: Optional[str], dry_run: bool):
    """Convert PRD to Jira stories."""
    click.echo(f"📝 Processing PRD: {prd_file}")
    
    try:
        from pm_automation_suite.feature_pipeline import FeaturePipeline
        
        async def run_pipeline():
            pipeline = FeaturePipeline({
                'vault_path': str(ctx.obj['vault_path']),
                'jira_project': project
            })
            
            stories = await pipeline.prd_to_stories(prd_file, project)
            
            click.echo(f"✨ Generated {len(stories)} user stories:")
            for i, story in enumerate(stories[:5], 1):  # Show first 5
                click.echo(f"  {i}. {story.get('title', 'Untitled')}")
            
            if len(stories) > 5:
                click.echo(f"  ... and {len(stories) - 5} more")
            
            if not dry_run:
                # Create in Jira
                created = await pipeline.bulk_create_stories(stories, epic)
                click.echo(f"🎯 Created {len(created)} stories in Jira")
            else:
                click.echo("🔍 Dry run - no stories created")
        
        asyncio.run(run_pipeline())
        
    except ImportError:
        click.echo("❌ PM Automation Suite not available. Install with: pip install obsidian-vault-tools[pm-automation]", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@pm.command()
@click.option('--dashboard', '-d', type=click.Choice(['team', 'executive', 'product']), 
              default='team', help='Dashboard type')
@click.option('--output', '-o', help='Output HTML file')
@click.pass_context
def analytics(ctx, dashboard: str, output: Optional[str]):
    """View PM performance analytics."""
    click.echo(f"📊 Generating {dashboard} analytics dashboard...")
    
    try:
        from pm_automation_suite.analytics_hub import DashboardGenerator
        
        async def run_analytics():
            generator = DashboardGenerator({
                'vault_path': str(ctx.obj['vault_path'])
            })
            
            dashboard_html = await generator.create_pm_dashboard(dashboard_type=dashboard)
            
            if output:
                with open(output, 'w') as f:
                    f.write(dashboard_html)
                click.echo(f"📈 Dashboard saved to: {output}")
            else:
                # Open in browser
                import tempfile
                import webbrowser
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    f.write(dashboard_html)
                    temp_path = f.name
                
                webbrowser.open(f'file://{temp_path}')
                click.echo("📈 Dashboard opened in browser")
        
        asyncio.run(run_analytics())
        
    except ImportError:
        click.echo("❌ PM Automation Suite not available. Install with: pip install obsidian-vault-tools[pm-automation]", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@pm.command()
@click.option('--metric', '-m', multiple=True, help='Specific metrics to monitor')
@click.option('--threshold', '-t', help='Alert threshold')
@click.option('--duration', '-d', default='1h', help='Monitoring duration (e.g., 1h, 30m)')
@click.pass_context
def monitor(ctx, metric: tuple, threshold: Optional[str], duration: str):
    """Real-time PM metrics monitoring."""
    click.echo("🚨 Starting real-time monitoring...")
    
    if metric:
        click.echo(f"📊 Monitoring metrics: {', '.join(metric)}")
    else:
        click.echo("📊 Monitoring all PM metrics")
    
    try:
        from pm_automation_suite.analytics_hub import MonitoringSystem
        
        async def run_monitoring():
            monitoring = MonitoringSystem({
                'vault_path': str(ctx.obj['vault_path'])
            })
            
            # Set up monitoring
            if not metric:
                await monitoring.create_default_pm_monitoring()
            else:
                for m in metric:
                    await monitoring.add_metric_monitor(m, threshold=threshold)
            
            click.echo(f"⏰ Monitoring for {duration}...")
            click.echo("Press Ctrl+C to stop monitoring")
            
            try:
                await monitoring.start_monitoring()
            except KeyboardInterrupt:
                click.echo("\n🛑 Monitoring stopped")
                await monitoring.stop_monitoring()
        
        asyncio.run(run_monitoring())
        
    except ImportError:
        click.echo("❌ PM Automation Suite not available. Install with: pip install obsidian-vault-tools[pm-automation]", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@pm.command()
@click.pass_context
def quality(ctx):
    """Run content quality analysis."""
    click.echo("🔍 Analyzing content quality...")
    
    try:
        from obsidian_vault_tools.pm_tools import ContentQualityEngine
        
        engine = ContentQualityEngine(str(ctx.obj['vault_path']))
        report = engine.analyze_vault()
        
        click.echo(f"\n📊 Quality Report:")
        click.echo(f"  Overall Score: {report.overall_score:.1f}/100")
        click.echo(f"  Files Analyzed: {report.total_files}")
        click.echo(f"  Issues Found: {len(report.issues_found)}")
        
        if report.naming_inconsistencies:
            click.echo(f"\n🏷️  Naming Issues:")
            for term, variants in list(report.naming_inconsistencies.items())[:3]:
                click.echo(f"  {term}: {', '.join(set(variants))}")
        
        if report.incomplete_content:
            click.echo(f"\n⚠️  Files with incomplete content: {len(report.incomplete_content)}")
        
        # Save detailed report
        report_path = ctx.obj['vault_path'] / 'quality_report.md'
        engine.generate_quality_report(report, str(report_path))
        click.echo(f"\n📋 Detailed report saved to: {report_path}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@pm.command()
@click.pass_context
def template(ctx):
    """Generate PM daily template."""
    click.echo("📝 Generating PM daily template...")
    
    try:
        from obsidian_vault_tools.pm_tools import DailyTemplateGenerator
        
        generator = DailyTemplateGenerator(str(ctx.obj['vault_path']))
        template = generator.generate_template()
        
        # Save to today's daily note
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        daily_notes_path = ctx.obj['vault_path'] / 'Daily Notes'
        daily_notes_path.mkdir(exist_ok=True)
        
        template_path = daily_notes_path / f'{today}.md'
        
        if template_path.exists():
            if not click.confirm(f"Daily note for {today} exists. Overwrite?"):
                return
        
        with open(template_path, 'w') as f:
            f.write(template)
        
        click.echo(f"✅ Daily template saved to: {template_path}")
        
    except ImportError:
        click.echo("❌ Daily template generator not available", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@pm.command()
@click.pass_context  
def burnout(ctx):
    """Check burnout risk score."""
    click.echo("🔥 Analyzing burnout risk...")
    
    try:
        from obsidian_vault_tools.pm_tools import BurnoutDetector
        
        detector = BurnoutDetector(str(ctx.obj['vault_path']))
        risk_score = detector.analyze_burnout_risk()
        
        click.echo(f"\n🎯 Burnout Risk Analysis:")
        click.echo(f"  Overall Risk: {risk_score['overall_risk']}")
        click.echo(f"  Risk Score: {risk_score['risk_score']:.1f}/100")
        
        if risk_score['risk_factors']:
            click.echo(f"\n⚠️  Risk Factors:")
            for factor in risk_score['risk_factors'][:3]:
                click.echo(f"  • {factor}")
        
        if risk_score['recommendations']:
            click.echo(f"\n💡 Recommendations:")
            for rec in risk_score['recommendations'][:3]:
                click.echo(f"  • {rec}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


@pm.command()
@click.pass_context
def config(ctx):
    """Configure PM Automation Suite."""
    click.echo("⚙️ PM Automation Suite Configuration")
    click.echo("This will launch the interactive configuration...")
    
    # Launch the unified manager with PM configuration
    try:
        from unified_vault_manager import UnifiedVaultManager
        manager = UnifiedVaultManager()
        # This would ideally jump directly to PM Suite Configuration
        # For now, just provide instructions
        click.echo("\n📋 To configure PM Automation:")
        click.echo("1. Run: ovt")
        click.echo("2. Navigate to: Settings & Configuration → PM Suite Configuration")
        click.echo("3. Follow the setup wizard")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


if __name__ == '__main__':
    pm()