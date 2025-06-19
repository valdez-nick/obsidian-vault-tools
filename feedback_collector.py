#!/usr/bin/env python3
"""
Feedback Collector
Collects user feedback and query-response pairs for model improvement
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sqlite3
import logging

logger = logging.getLogger(__name__)

@dataclass
class QueryFeedback:
    """Represents feedback for a query-response pair"""
    query_id: str
    query: str
    intent: str
    response: str
    models_used: List[str]
    strategy: str
    confidence: float
    user_rating: Optional[int] = None  # 1-5 scale
    user_feedback: Optional[str] = None
    timestamp: str = ""
    vault_context: Optional[Dict] = None
    response_time: Optional[float] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class FeedbackCollector:
    """Collects and manages feedback for improving models"""
    
    def __init__(self, data_path: str = "./llm_feedback"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(exist_ok=True)
        self.db_path = self.data_path / "feedback.db"
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for feedback storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                query_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                intent TEXT,
                response TEXT,
                models_used TEXT,
                strategy TEXT,
                confidence REAL,
                user_rating INTEGER,
                user_feedback TEXT,
                timestamp TEXT,
                vault_context TEXT,
                response_time REAL
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_intent ON feedback(intent)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_rating ON feedback(user_rating)
        """)
        
        conn.commit()
        conn.close()
        
    async def add_feedback(self, feedback: QueryFeedback):
        """Add feedback to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO feedback 
                (query_id, query, intent, response, models_used, strategy, 
                 confidence, user_rating, user_feedback, timestamp, 
                 vault_context, response_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.query_id,
                feedback.query,
                feedback.intent,
                feedback.response,
                json.dumps(feedback.models_used),
                feedback.strategy,
                feedback.confidence,
                feedback.user_rating,
                feedback.user_feedback,
                feedback.timestamp,
                json.dumps(feedback.vault_context) if feedback.vault_context else None,
                feedback.response_time
            ))
            
            conn.commit()
            logger.info(f"Added feedback for query {feedback.query_id}")
            
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            raise
        finally:
            conn.close()
            
    async def update_rating(self, query_id: str, rating: int, feedback: Optional[str] = None):
        """Update user rating for a query"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if feedback:
                cursor.execute("""
                    UPDATE feedback 
                    SET user_rating = ?, user_feedback = ?
                    WHERE query_id = ?
                """, (rating, feedback, query_id))
            else:
                cursor.execute("""
                    UPDATE feedback 
                    SET user_rating = ?
                    WHERE query_id = ?
                """, (rating, query_id))
                
            conn.commit()
            logger.info(f"Updated rating for query {query_id}")
            
        except Exception as e:
            logger.error(f"Failed to update rating: {e}")
            raise
        finally:
            conn.close()
            
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about collected feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Total queries
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_queries = cursor.fetchone()[0]
            
            # Queries with ratings
            cursor.execute("SELECT COUNT(*) FROM feedback WHERE user_rating IS NOT NULL")
            rated_queries = cursor.fetchone()[0]
            
            # Average rating
            cursor.execute("SELECT AVG(user_rating) FROM feedback WHERE user_rating IS NOT NULL")
            avg_rating = cursor.fetchone()[0] or 0
            
            # Rating distribution
            cursor.execute("""
                SELECT user_rating, COUNT(*) 
                FROM feedback 
                WHERE user_rating IS NOT NULL 
                GROUP BY user_rating
            """)
            rating_dist = dict(cursor.fetchall())
            
            # Intent distribution
            cursor.execute("""
                SELECT intent, COUNT(*) 
                FROM feedback 
                GROUP BY intent
            """)
            intent_dist = dict(cursor.fetchall())
            
            # Model usage
            cursor.execute("SELECT models_used FROM feedback")
            all_models = []
            for row in cursor.fetchall():
                models = json.loads(row[0])
                all_models.extend(models)
            
            model_usage = {}
            for model in set(all_models):
                model_usage[model] = all_models.count(model)
                
            # Strategy distribution
            cursor.execute("""
                SELECT strategy, COUNT(*) 
                FROM feedback 
                GROUP BY strategy
            """)
            strategy_dist = dict(cursor.fetchall())
            
            # Average response time
            cursor.execute("""
                SELECT AVG(response_time) 
                FROM feedback 
                WHERE response_time IS NOT NULL
            """)
            avg_response_time = cursor.fetchone()[0] or 0
            
            return {
                "total_queries": total_queries,
                "rated_queries": rated_queries,
                "average_rating": round(avg_rating, 2),
                "rating_distribution": rating_dist,
                "intent_distribution": intent_dist,
                "model_usage": model_usage,
                "strategy_distribution": strategy_dist,
                "average_response_time": round(avg_response_time, 2)
            }
            
        finally:
            conn.close()
            
    async def get_low_rated_queries(self, threshold: int = 3) -> List[Dict]:
        """Get queries with low ratings for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT query_id, query, intent, response, user_rating, user_feedback
                FROM feedback
                WHERE user_rating < ?
                ORDER BY timestamp DESC
                LIMIT 50
            """, (threshold,))
            
            rows = cursor.fetchall()
            return [
                {
                    "query_id": row[0],
                    "query": row[1],
                    "intent": row[2],
                    "response": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                    "rating": row[4],
                    "feedback": row[5]
                }
                for row in rows
            ]
            
        finally:
            conn.close()
            
    async def export_training_data(self, min_rating: int = 4) -> Dict[str, List]:
        """Export high-quality query-response pairs for training"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get high-rated queries
            cursor.execute("""
                SELECT query, intent, response, models_used, vault_context
                FROM feedback
                WHERE user_rating >= ?
            """, (min_rating,))
            
            training_data = {
                "classification": [],
                "generation": [],
                "extraction": []
            }
            
            for row in cursor.fetchall():
                query = row[0]
                intent = row[1]
                response = row[2]
                models_used = json.loads(row[3])
                vault_context = json.loads(row[4]) if row[4] else None
                
                # Classification examples (query -> intent)
                training_data["classification"].append({
                    "text": query,
                    "label": intent
                })
                
                # Generation examples (query + context -> response)
                if vault_context:
                    context_str = json.dumps(vault_context, indent=2)
                    prompt = f"Context:\n{context_str}\n\nQuery: {query}"
                else:
                    prompt = f"Query: {query}"
                    
                training_data["generation"].append({
                    "prompt": prompt,
                    "response": response,
                    "models": models_used
                })
                
                # Entity extraction examples
                if intent == "extract":
                    training_data["extraction"].append({
                        "text": query,
                        "response": response
                    })
                    
            return training_data
            
        finally:
            conn.close()
            
    async def get_model_performance(self) -> Dict[str, Dict]:
        """Analyze performance by model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all feedback with ratings
            cursor.execute("""
                SELECT models_used, user_rating, response_time
                FROM feedback
                WHERE user_rating IS NOT NULL
            """)
            
            model_stats = {}
            
            for row in cursor.fetchall():
                models = json.loads(row[0])
                rating = row[1]
                response_time = row[2]
                
                for model in models:
                    if model not in model_stats:
                        model_stats[model] = {
                            "ratings": [],
                            "response_times": []
                        }
                    
                    model_stats[model]["ratings"].append(rating)
                    if response_time:
                        model_stats[model]["response_times"].append(response_time)
                        
            # Calculate averages
            model_performance = {}
            for model, stats in model_stats.items():
                ratings = stats["ratings"]
                times = stats["response_times"]
                
                model_performance[model] = {
                    "average_rating": sum(ratings) / len(ratings) if ratings else 0,
                    "total_queries": len(ratings),
                    "average_response_time": sum(times) / len(times) if times else 0,
                    "rating_distribution": {
                        i: ratings.count(i) for i in range(1, 6)
                    }
                }
                
            return model_performance
            
        finally:
            conn.close()
            
    async def generate_improvement_report(self) -> str:
        """Generate a report with improvement recommendations"""
        stats = await self.get_feedback_stats()
        low_rated = await self.get_low_rated_queries()
        model_perf = await self.get_model_performance()
        
        report = []
        report.append("# 📊 Model Performance & Improvement Report")
        report.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
        
        # Overall statistics
        report.append("## 📈 Overall Statistics")
        report.append(f"- **Total Queries:** {stats['total_queries']}")
        report.append(f"- **Rated Queries:** {stats['rated_queries']}")
        report.append(f"- **Average Rating:** {stats['average_rating']}/5")
        report.append(f"- **Average Response Time:** {stats['average_response_time']}s\n")
        
        # Rating distribution
        report.append("## ⭐ Rating Distribution")
        for rating in range(5, 0, -1):
            count = stats['rating_distribution'].get(rating, 0)
            percentage = (count / stats['rated_queries'] * 100) if stats['rated_queries'] > 0 else 0
            bar = "█" * int(percentage / 5)
            report.append(f"{rating}★: {bar} {count} ({percentage:.1f}%)")
        report.append("")
        
        # Model performance
        report.append("## 🤖 Model Performance")
        sorted_models = sorted(model_perf.items(), 
                             key=lambda x: x[1]['average_rating'], 
                             reverse=True)
        
        for model, perf in sorted_models:
            report.append(f"\n### {model}")
            report.append(f"- Average Rating: {perf['average_rating']:.2f}/5")
            report.append(f"- Total Queries: {perf['total_queries']}")
            report.append(f"- Avg Response Time: {perf['average_response_time']:.2f}s")
            
        # Intent distribution
        report.append("\n## 🎯 Query Intent Distribution")
        for intent, count in sorted(stats['intent_distribution'].items(), 
                                  key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_queries'] * 100)
            report.append(f"- **{intent}:** {count} ({percentage:.1f}%)")
            
        # Improvement recommendations
        report.append("\n## 💡 Improvement Recommendations")
        
        # Based on low ratings
        if low_rated:
            report.append("\n### Areas Needing Improvement")
            
            # Analyze common patterns in low-rated queries
            low_rated_intents = {}
            for query in low_rated:
                intent = query.get('intent', 'unknown')
                low_rated_intents[intent] = low_rated_intents.get(intent, 0) + 1
                
            report.append("\nLow-rated query types:")
            for intent, count in sorted(low_rated_intents.items(), 
                                      key=lambda x: x[1], reverse=True):
                report.append(f"- {intent}: {count} queries")
                
        # Model-specific recommendations
        report.append("\n### Model Recommendations")
        for model, perf in model_perf.items():
            if perf['average_rating'] < 3.5:
                report.append(f"- **{model}**: Consider fine-tuning or replacing (avg rating: {perf['average_rating']:.2f})")
            elif perf['average_response_time'] > 10:
                report.append(f"- **{model}**: Optimize for speed (avg time: {perf['average_response_time']:.1f}s)")
                
        # Strategy recommendations
        report.append("\n### Strategy Optimization")
        for strategy, count in stats['strategy_distribution'].items():
            percentage = (count / stats['total_queries'] * 100)
            report.append(f"- **{strategy}**: Used in {percentage:.1f}% of queries")
            
        return "\n".join(report)
        
    async def cleanup_old_feedback(self, days: int = 90):
        """Remove feedback older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                DELETE FROM feedback
                WHERE timestamp < ?
            """, (cutoff_date,))
            
            deleted = cursor.rowcount
            conn.commit()
            
            logger.info(f"Cleaned up {deleted} old feedback entries")
            return deleted
            
        finally:
            conn.close()

# Example usage
async def test_feedback_collector():
    """Test feedback collection functionality"""
    collector = FeedbackCollector()
    
    print("📝 Testing Feedback Collector")
    
    # Add some test feedback
    import uuid
    
    test_feedbacks = [
        QueryFeedback(
            query_id=str(uuid.uuid4()),
            query="How many meeting notes do I have?",
            intent="count",
            response="You have 42 meeting notes in your vault.",
            models_used=["general_qa"],
            strategy="single",
            confidence=0.95,
            user_rating=5,
            response_time=1.2
        ),
        QueryFeedback(
            query_id=str(uuid.uuid4()),
            query="Summarize project Alpha",
            intent="summarize",
            response="Project Alpha is a web application...",
            models_used=["summarizer", "general_qa"],
            strategy="ensemble",
            confidence=0.85,
            user_rating=4,
            response_time=3.5
        ),
        QueryFeedback(
            query_id=str(uuid.uuid4()),
            query="Find all bugs",
            intent="search",
            response="I couldn't find specific bug reports...",
            models_used=["general_qa"],
            strategy="single",
            confidence=0.6,
            user_rating=2,
            user_feedback="Not helpful, missed obvious bug reports",
            response_time=2.1
        )
    ]
    
    # Add feedback
    print("\n✅ Adding test feedback...")
    for feedback in test_feedbacks:
        await collector.add_feedback(feedback)
        
    # Get statistics
    print("\n📊 Feedback Statistics:")
    stats = await collector.get_feedback_stats()
    print(json.dumps(stats, indent=2))
    
    # Get low-rated queries
    print("\n⚠️ Low-rated queries:")
    low_rated = await collector.get_low_rated_queries()
    for query in low_rated:
        print(f"- {query['query']} (rating: {query['rating']})")
        
    # Export training data
    print("\n📦 Exporting training data...")
    training_data = await collector.export_training_data(min_rating=4)
    print(f"Classification examples: {len(training_data['classification'])}")
    print(f"Generation examples: {len(training_data['generation'])}")
    
    # Generate improvement report
    print("\n📄 Generating improvement report...")
    report = await collector.generate_improvement_report()
    print(report[:500] + "..." if len(report) > 500 else report)

if __name__ == "__main__":
    from datetime import timedelta
    asyncio.run(test_feedback_collector())