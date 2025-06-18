#!/usr/bin/env python3
"""
ASCII Flowchart Generator
Analyzes documents and creates ASCII flowcharts with LLM assistance
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Try to import required libraries
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class NodeType(Enum):
    """Types of flowchart nodes"""
    START = "start"
    END = "end"
    PROCESS = "process"
    DECISION = "decision"
    INPUT_OUTPUT = "io"
    SUBPROCESS = "subprocess"

@dataclass
class FlowNode:
    """Represents a node in the flowchart"""
    id: str
    type: NodeType
    text: str
    connections: List[str] = field(default_factory=list)
    position: Tuple[int, int] = (0, 0)
    width: int = 0
    height: int = 0

@dataclass
class FlowchartData:
    """Contains all flowchart information"""
    title: str
    nodes: Dict[str, FlowNode]
    start_node: str
    description: str = ""

class FlowchartAnalyzer:
    """Analyzes documents to extract flowchart structure"""
    
    def __init__(self, llm_provider: str = "local", api_key: str = None):
        self.llm_provider = llm_provider
        self.api_key = api_key
        
    def analyze_document(self, content: str) -> FlowchartData:
        """
        Analyze document content and extract flowchart structure
        
        Args:
            content: Document text content
            
        Returns:
            FlowchartData with extracted structure
        """
        # For now, use pattern matching and heuristics
        # In full implementation, this would call LLM API
        
        # Extract title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Process Flow"
        
        # Extract steps using various patterns
        steps = self._extract_steps(content)
        decisions = self._extract_decisions(content)
        
        # Build flowchart data
        nodes = {}
        connections = []
        
        # Add start node
        start_id = "start"
        nodes[start_id] = FlowNode(
            id=start_id,
            type=NodeType.START,
            text="Start"
        )
        
        # Process extracted steps
        prev_id = start_id
        for i, step in enumerate(steps):
            node_id = f"step_{i}"
            
            # Determine node type
            if any(word in step.lower() for word in ['if', 'whether', 'decide', 'check']):
                node_type = NodeType.DECISION
            elif any(word in step.lower() for word in ['input', 'output', 'read', 'write']):
                node_type = NodeType.INPUT_OUTPUT
            else:
                node_type = NodeType.PROCESS
            
            nodes[node_id] = FlowNode(
                id=node_id,
                type=node_type,
                text=self._wrap_text(step, 20)
            )
            
            # Connect to previous node
            nodes[prev_id].connections.append(node_id)
            prev_id = node_id
        
        # Add end node
        end_id = "end"
        nodes[end_id] = FlowNode(
            id=end_id,
            type=NodeType.END,
            text="End"
        )
        nodes[prev_id].connections.append(end_id)
        
        # Create description
        description = f"This flowchart represents the process described in '{title}'. "
        description += f"It contains {len(steps)} main steps"
        if decisions:
            description += f" including {len(decisions)} decision points"
        description += "."
        
        return FlowchartData(
            title=title,
            nodes=nodes,
            start_node=start_id,
            description=description
        )
    
    def _extract_steps(self, content: str) -> List[str]:
        """Extract process steps from content"""
        steps = []
        
        # Look for numbered lists
        numbered_pattern = r'^\d+\.\s+(.+)$'
        steps.extend(re.findall(numbered_pattern, content, re.MULTILINE))
        
        # Look for bullet points
        bullet_pattern = r'^[\*\-]\s+(.+)$'
        steps.extend(re.findall(bullet_pattern, content, re.MULTILINE))
        
        # Look for action words at start of sentences
        action_pattern = r'^((?:First|Then|Next|Finally|Now),?\s+.+?)(?:\.|$)'
        steps.extend(re.findall(action_pattern, content, re.MULTILINE))
        
        # Clean and deduplicate
        steps = [s.strip() for s in steps if len(s.strip()) > 10]
        seen = set()
        unique_steps = []
        for step in steps:
            if step.lower() not in seen:
                seen.add(step.lower())
                unique_steps.append(step)
        
        return unique_steps[:10]  # Limit to 10 steps for readability
    
    def _extract_decisions(self, content: str) -> List[str]:
        """Extract decision points from content"""
        decisions = []
        
        # Look for if/then patterns
        if_pattern = r'[Ii]f\s+(.+?)\s*,?\s*then'
        decisions.extend(re.findall(if_pattern, content))
        
        # Look for question patterns
        question_pattern = r'([A-Z][^.?]+\?)'
        decisions.extend(re.findall(question_pattern, content))
        
        return decisions
    
    def _wrap_text(self, text: str, max_width: int) -> str:
        """Wrap text to fit within max width"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_width:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)

class ASCIIFlowchartBuilder:
    """Builds ASCII flowchart from flowchart data"""
    
    # ASCII art components
    COMPONENTS = {
        'box_horizontal': '─',
        'box_vertical': '│',
        'box_top_left': '┌',
        'box_top_right': '┐',
        'box_bottom_left': '└',
        'box_bottom_right': '┘',
        'arrow_down': '↓',
        'arrow_right': '→',
        'arrow_left': '←',
        'arrow_up': '↑',
        'connector_cross': '┼',
        'connector_t_down': '┬',
        'connector_t_up': '┴',
        'connector_t_right': '├',
        'connector_t_left': '┤',
        'diamond_top': '◇',
        'diamond_left': '◇',
        'diamond_right': '◇',
        'diamond_bottom': '◇',
    }
    
    def __init__(self, style: str = "simple"):
        self.style = style
        self.canvas = []
        self.width = 80
        self.height = 50
        
    def build(self, flowchart_data: FlowchartData) -> str:
        """
        Build ASCII flowchart from data
        
        Args:
            flowchart_data: The flowchart structure
            
        Returns:
            ASCII art representation
        """
        # Initialize canvas
        self._init_canvas()
        
        # Calculate positions
        self._calculate_positions(flowchart_data)
        
        # Draw nodes
        for node in flowchart_data.nodes.values():
            self._draw_node(node)
        
        # Draw connections
        for node in flowchart_data.nodes.values():
            for target_id in node.connections:
                if target_id in flowchart_data.nodes:
                    target = flowchart_data.nodes[target_id]
                    self._draw_connection(node, target)
        
        # Convert canvas to string
        return self._canvas_to_string()
    
    def _init_canvas(self):
        """Initialize empty canvas"""
        self.canvas = [[' ' for _ in range(self.width)] for _ in range(self.height)]
    
    def _calculate_positions(self, flowchart_data: FlowchartData):
        """Calculate node positions using simple vertical layout"""
        y_offset = 2
        x_center = self.width // 2
        
        # Simple vertical layout
        current_y = y_offset
        
        # Position nodes in order
        visited = set()
        queue = [flowchart_data.start_node]
        
        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            
            visited.add(node_id)
            node = flowchart_data.nodes[node_id]
            
            # Calculate node dimensions
            lines = node.text.split('\n')
            node.width = max(len(line) for line in lines) + 4
            node.height = len(lines) + 2
            
            # Position node
            node.position = (x_center - node.width // 2, current_y)
            current_y += node.height + 3
            
            # Add connections to queue
            queue.extend(node.connections)
    
    def _draw_node(self, node: FlowNode):
        """Draw a node on the canvas"""
        x, y = node.position
        
        if node.type in [NodeType.START, NodeType.END]:
            # Rounded box for start/end
            self._draw_rounded_box(x, y, node.width, node.height, node.text)
        elif node.type == NodeType.DECISION:
            # Diamond for decisions
            self._draw_diamond(x, y, node.width, node.height, node.text)
        elif node.type == NodeType.INPUT_OUTPUT:
            # Parallelogram for I/O
            self._draw_parallelogram(x, y, node.width, node.height, node.text)
        else:
            # Regular box for process
            self._draw_box(x, y, node.width, node.height, node.text)
    
    def _draw_box(self, x: int, y: int, width: int, height: int, text: str):
        """Draw a rectangular box"""
        # Top border
        self._draw_char(x, y, self.COMPONENTS['box_top_left'])
        for i in range(1, width - 1):
            self._draw_char(x + i, y, self.COMPONENTS['box_horizontal'])
        self._draw_char(x + width - 1, y, self.COMPONENTS['box_top_right'])
        
        # Sides and text
        lines = text.split('\n')
        for i in range(1, height - 1):
            self._draw_char(x, y + i, self.COMPONENTS['box_vertical'])
            self._draw_char(x + width - 1, y + i, self.COMPONENTS['box_vertical'])
            
            # Draw text centered
            if i - 1 < len(lines):
                line = lines[i - 1]
                text_x = x + (width - len(line)) // 2
                self._draw_text(text_x, y + i, line)
        
        # Bottom border
        self._draw_char(x, y + height - 1, self.COMPONENTS['box_bottom_left'])
        for i in range(1, width - 1):
            self._draw_char(x + i, y + height - 1, self.COMPONENTS['box_horizontal'])
        self._draw_char(x + width - 1, y + height - 1, self.COMPONENTS['box_bottom_right'])
    
    def _draw_rounded_box(self, x: int, y: int, width: int, height: int, text: str):
        """Draw a rounded box for start/end nodes"""
        # Top border
        self._draw_text(x + 1, y, '╭' + '─' * (width - 4) + '╮')
        
        # Sides and text
        lines = text.split('\n')
        for i in range(1, height - 1):
            self._draw_char(x, y + i, '│')
            self._draw_char(x + width - 1, y + i, '│')
            
            # Draw text centered
            if i - 1 < len(lines):
                line = lines[i - 1]
                text_x = x + (width - len(line)) // 2
                self._draw_text(text_x, y + i, line)
        
        # Bottom border
        self._draw_text(x + 1, y + height - 1, '╰' + '─' * (width - 4) + '╯')
    
    def _draw_diamond(self, x: int, y: int, width: int, height: int, text: str):
        """Draw a diamond shape for decision nodes"""
        # Simplified diamond using angle brackets
        mid_y = y + height // 2
        
        # Top point
        self._draw_char(x + width // 2, y, '◇')
        
        # Middle with text
        lines = text.split('\n')
        for i, line in enumerate(lines):
            text_y = mid_y - len(lines) // 2 + i
            text_x = x + (width - len(line)) // 2
            self._draw_text(text_x - 2, text_y, '< ' + line + ' >')
        
        # Bottom point
        self._draw_char(x + width // 2, y + height - 1, '◇')
    
    def _draw_parallelogram(self, x: int, y: int, width: int, height: int, text: str):
        """Draw a parallelogram for I/O nodes"""
        # Simplified using slashes
        for i in range(height):
            if i == 0:
                self._draw_text(x + 2, y + i, '/' + '─' * (width - 4) + '\\')
            elif i == height - 1:
                self._draw_text(x, y + i, '\\' + '─' * (width - 4) + '/')
            else:
                self._draw_char(x + 2 - i, y + i, '/')
                self._draw_char(x + width - 3 + i, y + i, '\\')
                
                # Draw text
                lines = text.split('\n')
                if i - 1 < len(lines):
                    line = lines[i - 1]
                    text_x = x + (width - len(line)) // 2
                    self._draw_text(text_x, y + i, line)
    
    def _draw_connection(self, from_node: FlowNode, to_node: FlowNode):
        """Draw connection between nodes"""
        # Simple vertical connection
        from_x = from_node.position[0] + from_node.width // 2
        from_y = from_node.position[1] + from_node.height
        
        to_x = to_node.position[0] + to_node.width // 2
        to_y = to_node.position[1] - 1
        
        # Draw vertical line
        for y in range(from_y, to_y):
            self._draw_char(from_x, y, '│')
        
        # Draw arrow
        self._draw_char(from_x, to_y, '↓')
    
    def _draw_char(self, x: int, y: int, char: str):
        """Draw a single character on canvas"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.canvas[y][x] = char
    
    def _draw_text(self, x: int, y: int, text: str):
        """Draw text on canvas"""
        for i, char in enumerate(text):
            self._draw_char(x + i, y, char)
    
    def _canvas_to_string(self) -> str:
        """Convert canvas to string"""
        # Trim empty rows
        first_non_empty = 0
        last_non_empty = len(self.canvas) - 1
        
        for i, row in enumerate(self.canvas):
            if any(c != ' ' for c in row):
                first_non_empty = i
                break
        
        for i in range(len(self.canvas) - 1, -1, -1):
            if any(c != ' ' for c in self.canvas[i]):
                last_non_empty = i
                break
        
        # Build string
        lines = []
        for i in range(first_non_empty, last_non_empty + 1):
            line = ''.join(self.canvas[i]).rstrip()
            lines.append(line)
        
        return '\n'.join(lines)

def generate_flowchart_from_file(file_path: str, output_path: str = None) -> str:
    """
    Generate ASCII flowchart from a markdown file
    
    Args:
        file_path: Path to input markdown file
        output_path: Optional path to save output
        
    Returns:
        Generated markdown content with flowchart
    """
    # Read input file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Analyze document
    analyzer = FlowchartAnalyzer()
    flowchart_data = analyzer.analyze_document(content)
    
    # Build ASCII flowchart
    builder = ASCIIFlowchartBuilder()
    ascii_flowchart = builder.build(flowchart_data)
    
    # Create output markdown
    output = f"# {flowchart_data.title} - Flowchart\n\n"
    output += "```\n"
    output += ascii_flowchart
    output += "\n```\n\n"
    output += f"## Description\n\n{flowchart_data.description}\n\n"
    output += "## Process Steps\n\n"
    
    # List steps
    step_num = 1
    for node_id, node in flowchart_data.nodes.items():
        if node.type not in [NodeType.START, NodeType.END]:
            clean_text = node.text.replace('\n', ' ')
            output += f"{step_num}. **{clean_text}**"
            if node.type == NodeType.DECISION:
                output += " (Decision Point)"
            elif node.type == NodeType.INPUT_OUTPUT:
                output += " (Input/Output)"
            output += "\n"
            step_num += 1
    
    output += f"\n---\n*Generated from: {Path(file_path).name}*\n"
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Flowchart saved to: {output_path}")
    
    return output

def main():
    """CLI interface for flowchart generator"""
    if len(sys.argv) < 2:
        print("ASCII Flowchart Generator")
        print("\nUsage:")
        print("  python ascii_flowchart_generator.py <input.md> [output.md]")
        print("\nExample:")
        print("  python ascii_flowchart_generator.py process.md process_flowchart.md")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Generate flowchart
    result = generate_flowchart_from_file(input_file, output_file)
    
    if not output_file:
        print(result)

if __name__ == '__main__':
    main()