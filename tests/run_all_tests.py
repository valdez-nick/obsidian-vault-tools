#!/usr/bin/env python3
"""
Master Test Runner - Runs all auto-discovering tests
This is the only test file you need to run!
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class MasterTestRunner:
    """Runs all tests and generates comprehensive report"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'coverage': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'duration': 0
            }
        }
        self.start_time = time.time()
    
    def discover_test_files(self) -> List[Path]:
        """Find all test files"""
        test_files = []
        
        # E2E tests (our main auto-discovering tests)
        e2e_tests = [
            'e2e/test_auto_discover_all_features.py',
            'e2e/test_menu_auto_validation.py',
            'e2e/test_cli_auto_discover.py'
        ]
        
        for test_path in e2e_tests:
            full_path = self.test_dir / test_path
            if full_path.exists():
                test_files.append(full_path)
        
        # Also find any other test_*.py files
        for test_file in self.test_dir.rglob('test_*.py'):
            if test_file not in test_files and test_file.name != 'test_run_all.py':
                test_files.append(test_file)
        
        return sorted(test_files)
    
    def run_test_file(self, test_file: Path) -> Dict[str, Any]:
        """Run a single test file and capture results"""
        print(f"\n{Colors.CYAN}Running: {test_file.name}{Colors.ENDC}")
        print("=" * 60)
        
        result = {
            'file': str(test_file.relative_to(self.test_dir)),
            'passed': False,
            'duration': 0,
            'output': '',
            'error': None
        }
        
        start = time.time()
        
        try:
            # Run the test file
            proc = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout per test file
            )
            
            result['duration'] = time.time() - start
            result['output'] = proc.stdout + proc.stderr
            result['passed'] = proc.returncode == 0
            
            if result['passed']:
                print(f"{Colors.GREEN}✓ PASSED{Colors.ENDC} ({result['duration']:.1f}s)")
            else:
                print(f"{Colors.RED}✗ FAILED{Colors.ENDC} ({result['duration']:.1f}s)")
                print(f"Exit code: {proc.returncode}")
                
                # Show last few lines of output for failures
                lines = result['output'].strip().split('\n')
                if len(lines) > 10:
                    print("Last 10 lines of output:")
                    for line in lines[-10:]:
                        print(f"  {line}")
                
        except subprocess.TimeoutExpired:
            result['duration'] = time.time() - start
            result['error'] = 'Test timed out after 60 seconds'
            print(f"{Colors.RED}✗ TIMEOUT{Colors.ENDC}")
            
        except Exception as e:
            result['duration'] = time.time() - start
            result['error'] = str(e)
            print(f"{Colors.RED}✗ ERROR: {e}{Colors.ENDC}")
        
        return result
    
    def extract_coverage_info(self, test_output: str) -> Dict[str, Any]:
        """Extract coverage information from test output"""
        coverage = {}
        
        # Look for coverage reports in output
        if 'CLI COVERAGE REPORT' in test_output:
            # Extract CLI coverage
            lines = test_output.split('\n')
            for i, line in enumerate(lines):
                if 'Total Commands Discovered:' in line:
                    try:
                        coverage['cli_commands'] = int(line.split(':')[1].strip())
                    except:
                        pass
        
        if 'TEST COVERAGE REPORT' in test_output:
            # Extract feature coverage
            lines = test_output.split('\n')
            for i, line in enumerate(lines):
                if 'CLI Commands:' in line:
                    try:
                        coverage['cli_commands'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Menu Categories:' in line:
                    try:
                        coverage['menu_categories'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Total Menu Options:' in line:
                    try:
                        coverage['menu_options'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Features:' in line and 'available' in line:
                    try:
                        # Parse "Features: X (Y available)"
                        parts = line.split(':')[1].strip()
                        total = int(parts.split('(')[0].strip())
                        available = int(parts.split('(')[1].split()[0])
                        coverage['features'] = {'total': total, 'available': available}
                    except:
                        pass
        
        return coverage
    
    def generate_html_report(self):
        """Generate an HTML report of test results"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Obsidian Vault Tools - Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            border-radius: 5px;
        }}
        .summary {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            text-align: center;
        }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .test-result {{
            background-color: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .test-passed {{ border-left: 5px solid #28a745; }}
        .test-failed {{ border-left: 5px solid #dc3545; }}
        .coverage {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 3px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Obsidian Vault Tools - Automated Test Report</h1>
        <p>Generated: {self.results['timestamp']}</p>
        <p>Duration: {self.results['summary']['duration']:.1f} seconds</p>
    </div>
    
    <div class="summary">
        <div class="summary-card">
            <h2>{self.results['summary']['total']}</h2>
            <p>Total Tests</p>
        </div>
        <div class="summary-card passed">
            <h2>{self.results['summary']['passed']}</h2>
            <p>Passed</p>
        </div>
        <div class="summary-card failed">
            <h2>{self.results['summary']['failed']}</h2>
            <p>Failed</p>
        </div>
        <div class="summary-card">
            <h2>{self.results['summary']['passed'] / self.results['summary']['total'] * 100:.0f}%</h2>
            <p>Success Rate</p>
        </div>
    </div>
"""
        
        # Add coverage information
        if self.results['coverage']:
            html += """
    <div class="coverage">
        <h2>Feature Coverage</h2>
        <ul>
"""
            for key, value in self.results['coverage'].items():
                if isinstance(value, dict):
                    html += f"<li><strong>{key}:</strong> {value.get('available', 0)}/{value.get('total', 0)} available</li>"
                else:
                    html += f"<li><strong>{key}:</strong> {value}</li>"
            html += """
        </ul>
    </div>
"""
        
        # Add test results
        html += "<h2>Test Results</h2>"
        for test_file, result in self.results['tests'].items():
            status_class = 'test-passed' if result['passed'] else 'test-failed'
            status_text = '✓ PASSED' if result['passed'] else '✗ FAILED'
            
            html += f"""
    <div class="test-result {status_class}">
        <h3>{test_file} - {status_text}</h3>
        <p>Duration: {result['duration']:.1f} seconds</p>
"""
            
            if result.get('error'):
                html += f"<p><strong>Error:</strong> {result['error']}</p>"
            
            if not result['passed'] and result['output']:
                # Show last 20 lines of output for failed tests
                lines = result['output'].strip().split('\n')[-20:]
                html += "<details><summary>Output (last 20 lines)</summary><pre>"
                html += '\n'.join(lines)
                html += "</pre></details>"
            
            html += "</div>"
        
        html += """
</body>
</html>
"""
        
        # Save report
        report_path = self.test_dir / 'test_report.html'
        report_path.write_text(html)
        
        return report_path
    
    def run_all_tests(self):
        """Run all tests"""
        print(f"{Colors.BOLD}Obsidian Vault Tools - Automated Test Suite{Colors.ENDC}")
        print(f"{'='*60}")
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Discover test files
        test_files = self.discover_test_files()
        print(f"\nDiscovered {len(test_files)} test files")
        
        # Run each test
        for test_file in test_files:
            result = self.run_test_file(test_file)
            self.results['tests'][result['file']] = result
            
            # Update summary
            self.results['summary']['total'] += 1
            if result['passed']:
                self.results['summary']['passed'] += 1
            else:
                self.results['summary']['failed'] += 1
            
            # Extract coverage info
            if result['output']:
                coverage = self.extract_coverage_info(result['output'])
                self.results['coverage'].update(coverage)
        
        # Calculate total duration
        self.results['summary']['duration'] = time.time() - self.start_time
        
        # Generate reports
        self.print_summary()
        html_report = self.generate_html_report()
        
        # Save JSON report
        json_report = self.test_dir / 'test_report.json'
        with open(json_report, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{Colors.CYAN}Reports generated:{Colors.ENDC}")
        print(f"  - HTML: {html_report}")
        print(f"  - JSON: {json_report}")
        
        return self.results['summary']['failed'] == 0
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print(f"{Colors.BOLD}TEST SUMMARY{Colors.ENDC}")
        print(f"{'='*60}")
        
        total = self.results['summary']['total']
        passed = self.results['summary']['passed']
        failed = self.results['summary']['failed']
        duration = self.results['summary']['duration']
        
        print(f"Total Tests: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.ENDC}")
        print(f"{Colors.RED}Failed: {failed}{Colors.ENDC}")
        print(f"Duration: {duration:.1f} seconds")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"Success Rate: {success_rate:.1f}%")
            
            if success_rate == 100:
                print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! 🎉{Colors.ENDC}")
            elif success_rate >= 80:
                print(f"\n{Colors.YELLOW}⚠️ Most tests passed, but some issues remain{Colors.ENDC}")
            else:
                print(f"\n{Colors.RED}❌ Significant test failures detected{Colors.ENDC}")
        
        # Show coverage summary
        if self.results['coverage']:
            print(f"\n{Colors.BOLD}FEATURE COVERAGE:{Colors.ENDC}")
            for key, value in self.results['coverage'].items():
                if isinstance(value, dict):
                    print(f"  {key}: {value.get('available', 0)}/{value.get('total', 0)} available")
                else:
                    print(f"  {key}: {value}")


def main():
    """Run all tests"""
    runner = MasterTestRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()