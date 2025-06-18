# ASCII Flowchart Generator Guide

## Overview

The ASCII Flowchart Generator analyzes markdown documents and automatically creates ASCII art flowcharts representing the process or workflow described in the document.

## Features

- **Automatic Process Analysis**: Extracts steps, decisions, and flow from natural language
- **Multiple Node Types**: Support for start/end, process, decision, and I/O nodes
- **ASCII Art Output**: Creates beautiful flowcharts using ASCII characters
- **Markdown Integration**: Outputs complete markdown files with flowchart and description
- **Interactive Interface**: Available through the Enhanced Vault Manager

## Usage

### Via Enhanced Vault Manager (Recommended)

1. Launch the enhanced vault manager:
   ```bash
   ./obsidian_manager_enhanced
   ```

2. Navigate to: **ASCII Art Tools** → **Generate flowchart from document**

3. Select a markdown file from your vault

4. Choose output filename

5. The flowchart will be generated and saved to your vault

### Via Command Line

```bash
# Generate flowchart from a document
python3 ascii_flowchart_generator.py input.md output.md

# Generate and display without saving
python3 ascii_flowchart_generator.py input.md
```

## Supported Document Patterns

The analyzer recognizes various patterns in your markdown documents:

### Numbered Lists
```markdown
1. First, gather requirements
2. Then, design the system
3. Next, implement the code
4. Finally, deploy to production
```

### Bullet Points
```markdown
- Analyze the problem
- Create solution design
- Implement and test
- Deploy and monitor
```

### Sequential Language
```markdown
First, we collect the data.
Then, we process the information.
Next, we generate the report.
Finally, we deliver to the client.
```

### Decision Points
The generator automatically detects decision language:
- "If X, then Y"
- Questions ending with "?"
- Words like "decide", "check", "whether"

## Flowchart Elements

### Start/End Nodes
```
╭─────╮
│Start│
╰─────╯
```

### Process Nodes
```
┌─────────────┐
│   Process   │
│Description  │
└─────────────┘
```

### Decision Nodes
```
     ◇
< Decision >
<  Point   >
     ◇
```

### Input/Output Nodes
```
/─────────────\
\  Input/Output /
/─────────────\
```

### Connections
```
│  (vertical)
→  (horizontal right)
↓  (arrow down)
```

## Example Input/Output

### Input Document (`process.md`):
```markdown
# User Registration Process

Our user registration follows these steps:

1. User enters email and password
2. System validates email format
3. If email is valid, create account
4. Send confirmation email
5. User clicks confirmation link
6. Account becomes active
```

### Generated Flowchart:
```
                     ╭─────╮
                    │Start│
                     ╰─────╯
                        │
                        ↓
          ┌─────────────────────────────┐
          │ User enters email and       │
          │        password             │
          └─────────────────────────────┘
                        │
                        ↓
                        ◇
                < System validates >
                <  email format   >
                        ◇
                        │
                        ↓
          ┌─────────────────────────────┐
          │ If email is valid, create   │
          │         account             │
          └─────────────────────────────┘
                        │
                        ↓
          ┌─────────────────────────────┐
          │   Send confirmation email   │
          └─────────────────────────────┘
                        │
                        ↓
          ┌─────────────────────────────┐
          │ User clicks confirmation    │
          │          link               │
          └─────────────────────────────┘
                        │
                        ↓
          ┌─────────────────────────────┐
          │    Account becomes active   │
          └─────────────────────────────┘
                        │
                        ↓
                     ╭─────╮
                    │ End │
                     ╰─────╯
```

## Best Practices

### Writing Documents for Better Flowcharts

1. **Use Sequential Language**: "First", "Then", "Next", "Finally"
2. **Number Your Steps**: Use numbered lists for main processes
3. **Clear Decision Points**: Use "If...then" or questions
4. **Separate Concerns**: Break complex processes into smaller steps
5. **Use Action Verbs**: Start steps with clear action words

### Example of Well-Structured Process:
```markdown
# Order Processing Workflow

1. Customer submits order online
2. System validates payment information
3. Is payment valid?
   - If yes, proceed to fulfillment
   - If no, notify customer of error
4. Generate picking list for warehouse
5. Warehouse staff picks items
6. Package items for shipping
7. Generate shipping label
8. Ship package to customer
9. Send tracking information to customer
```

## Customization

### Adding Custom Templates

You can extend the generator by modifying the ASCII components in `ascii_flowchart_generator.py`:

```python
COMPONENTS = {
    'box_horizontal': '─',
    'box_vertical': '│',
    'arrow_down': '↓',
    # Add your custom characters
}
```

### Extending Node Types

Add new node types by extending the `NodeType` enum and implementing drawing methods.

## Troubleshooting

### Common Issues

**Flowchart looks empty or minimal:**
- Ensure your document has numbered lists or sequential language
- Use clear action verbs
- Break down complex sentences

**Text is cut off in boxes:**
- The generator automatically wraps text
- Very long phrases may be truncated
- Consider shorter, more concise descriptions

**Missing decision points:**
- Use explicit "If...then" language
- Include questions with "?"
- Use decision keywords like "check", "validate", "decide"

### Getting Better Results

1. **Structure your document clearly** with numbered steps
2. **Use consistent language patterns** throughout
3. **Keep individual steps concise** (under 50 characters)
4. **Separate different processes** into different documents
5. **Include clear start and end points**

## Integration with Obsidian

### Linking Flowcharts

Generated flowcharts are saved as markdown files in your vault and can be:
- Linked from other notes: `[[process_flowchart]]`
- Embedded in other documents: `![[process_flowchart]]`
- Tagged and categorized like any other note

### Batch Processing

Process multiple documents by running the CLI tool in a loop:
```bash
for file in *.md; do
    python3 ascii_flowchart_generator.py "$file" "${file%.md}_flowchart.md"
done
```

## Future Enhancements

Planned features include:
- LLM integration for better process analysis
- Multiple flowchart styles (UML, BPMN-style)
- Horizontal flowchart layouts
- Color coding for different node types
- Export to other formats (SVG, PNG)

---

*Happy flowcharting! 📊🎨*