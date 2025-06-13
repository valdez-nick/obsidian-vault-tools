#!/bin/bash

echo "🤖 Obsidian Librarian Agent Monitor"
echo "=================================="
echo ""

agents=("tag-models-v2" "tag-core-v2" "dir-models-v2" "dir-core-v2" "tag-cli-v2")
worktrees=("worktrees/tag-models" "worktrees/tag-core-service" "worktrees/directory-models" "worktrees/directory-core-service" "worktrees/tag-cli-commands")

for i in "${!agents[@]}"; do
    agent="${agents[$i]}"
    worktree="${worktrees[$i]}"
    
    echo "📊 Agent: $agent"
    echo "📁 Worktree: $worktree"
    
    # Check if session exists
    if tmux has-session -t "$agent" 2>/dev/null; then
        echo "🔄 Status: ACTIVE"
        
        # Check git status
        if [ -d "$worktree" ]; then
            cd "$worktree"
            status=$(git status --porcelain | wc -l)
            if [ "$status" -gt 0 ]; then
                echo "📝 Changes: $status files modified"
                git status --porcelain | head -3
            else
                echo "📝 Changes: No changes yet"
            fi
            
            # Check recent commits
            commits=$(git log --oneline --since="10 minutes ago" | wc -l)
            if [ "$commits" -gt 0 ]; then
                echo "✅ Recent commits:"
                git log --oneline --since="10 minutes ago"
            fi
            cd - > /dev/null
        else
            echo "❌ Worktree not found"
        fi
        
        # Get last few lines of output
        echo "💬 Recent output:"
        tmux capture-pane -t "$agent" -p | tail -3 | sed 's/^/   /'
    else
        echo "❌ Status: INACTIVE"
    fi
    
    echo ""
    echo "---"
    echo ""
done

echo "🎯 Summary:"
echo "$(tmux list-sessions | grep -E "(tag-|dir-)" | wc -l) agents running"
echo "$(git worktree list | grep -E "(tag-|directory-)" | wc -l) worktrees active"