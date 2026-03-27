"""
Results Database for RLQAS Experiment Management.

This module provides a simple database for storing and querying
experimental results.
"""

import os
import json
import sqlite3
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from contextlib import contextmanager


class ResultsDatabase:
    """Database for storing and querying experiment results.

    This class provides a SQLite-based storage system for experiment
    results with support for:
    - Storing experiment metadata and results
    - Querying by experiment type, status, date range
    - Retrieving comparison metrics
    - Exporting results

    Args:
        db_path: Path to SQLite database file
    """

    def __init__(self, db_path: str = "results/experiments.db"):
        """Initialize results database."""
        self.db_path = db_path

        # Create directory if needed
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

        # Initialize database
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Get database connection context manager.

        Yields:
            SQLite connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT DEFAULT 'created',
                    created_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    config TEXT,
                    results TEXT
                )
            """)

            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    agent_name TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_type ON experiments(type)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON metrics(experiment_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name)"
            )

            conn.commit()

    def store_experiment(
        self,
        experiment_id: str,
        name: str,
        experiment_type: str,
        config: Dict[str, Any],
        results: Optional[Dict[str, Any]] = None,
        status: str = "created",
    ) -> bool:
        """Store experiment in database.

        Args:
            experiment_id: Experiment ID
            name: Experiment name
            experiment_type: Experiment type
            config: Experiment configuration
            results: Optional results dictionary
            status: Experiment status

        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO experiments
                (id, name, type, status, created_at, config, results)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                name,
                experiment_type,
                status,
                now,
                json.dumps(config),
                json.dumps(results) if results else None,
            ))

            conn.commit()
            return True

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: New status
            results: Optional results to store

        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            if status == "completed":
                cursor.execute("""
                    UPDATE experiments
                    SET status = ?, completed_at = ?, results = ?
                    WHERE id = ?
                """, (status, now, json.dumps(results) if results else None, experiment_id))
            elif status == "running":
                cursor.execute("""
                    UPDATE experiments
                    SET status = ?, started_at = ?
                    WHERE id = ?
                """, (status, now, experiment_id))
            else:
                cursor.execute("""
                    UPDATE experiments
                    SET status = ?
                    WHERE id = ?
                """, (status, experiment_id))

            conn.commit()
            return True

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment dictionary or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

    def get_experiments(
        self,
        experiment_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get experiments with optional filtering.

        Args:
            experiment_type: Filter by type
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of experiment dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM experiments WHERE 1=1"
            params = []

            if experiment_type:
                query += " AND type = ?"
                params.append(experiment_type)

            if status:
                query += " AND status = ?"
                params.append(status)

            query += " ORDER BY created_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def store_metric(
        self,
        experiment_id: str,
        metric_name: str,
        metric_value: float,
        agent_name: Optional[str] = None,
    ) -> bool:
        """Store a metric value.

        Args:
            experiment_id: Experiment ID
            metric_name: Metric name
            metric_value: Metric value
            agent_name: Optional agent name

        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (experiment_id, metric_name, metric_value, agent_name)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, metric_name, metric_value, agent_name))
            conn.commit()
            return True

    def get_metrics(
        self,
        experiment_id: str,
        metric_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get metrics for an experiment.

        Args:
            experiment_id: Experiment ID
            metric_name: Optional metric name filter

        Returns:
            List of metric dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM metrics WHERE experiment_id = ?"
            params = [experiment_id]

            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_comparison(
        self,
        experiment_type: str,
        metric_name: str,
    ) -> List[Dict[str, Any]]:
        """Get comparison metrics across experiments.

        Args:
            experiment_type: Type of experiments to compare
            metric_name: Metric to compare

        Returns:
            List of comparison data
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT e.id, e.name, e.created_at, m.metric_value, m.agent_name
                FROM experiments e
                JOIN metrics m ON e.id = m.experiment_id
                WHERE e.type = ? AND m.metric_name = ?
                ORDER BY m.metric_value DESC
            """, (experiment_type, metric_name))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its metrics.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if successful
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Delete metrics first
            cursor.execute("DELETE FROM metrics WHERE experiment_id = ?", (experiment_id,))

            # Delete experiment
            cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))

            conn.commit()
            return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Statistics dictionary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            # Total experiments
            cursor.execute("SELECT COUNT(*) as count FROM experiments")
            stats["total_experiments"] = cursor.fetchone()["count"]

            # By status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM experiments
                GROUP BY status
            """)
            stats["by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}

            # By type
            cursor.execute("""
                SELECT type, COUNT(*) as count
                FROM experiments
                GROUP BY type
            """)
            stats["by_type"] = {row["type"]: row["count"] for row in cursor.fetchall()}

            return stats

    def export_results(
        self,
        experiment_id: Optional[str] = None,
        output_path: str = "results_export.json",
    ) -> str:
        """Export results to JSON file.

        Args:
            experiment_id: Optional specific experiment to export
            output_path: Path to output file

        Returns:
            Path to exported file
        """
        if experiment_id:
            experiments = [self.get_experiment(experiment_id)]
        else:
            experiments = self.get_experiments()

        # Convert to serializable format
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "experiments": experiments,
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return output_path
