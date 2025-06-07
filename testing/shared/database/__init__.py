"""
Database module for testing framework
Modular database operations with clean separation of concerns
"""
from .connection import DatabaseConnection
from .schema_manager import SchemaManager
from .result_writer import ResultWriter

# Singleton instances for easy import
db_connection = DatabaseConnection()
schema_manager = SchemaManager(db_connection)
result_writer = ResultWriter(db_connection)

__all__ = [
    'DatabaseConnection',
    'SchemaManager', 
    'ResultWriter',
    'db_connection',
    'schema_manager',
    'result_writer'
]