"""
MongoDB Tools for Agentic Workflows with Compensation

This module provides MongoDB operations that can be used in agentic workflows
with automatic compensation/rollback capabilities.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from langchain_core.tools import tool

load_dotenv()

logger = logging.getLogger(__name__)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://admin:password@localhost:27017/test_lang_comp?authSource=admin")
DB_NAME = "test_lang_comp"

# Global client connection
_client: Optional[MongoClient] = None
_db = None


def get_db():
    """Get MongoDB database connection."""
    global _client, _db
    if _client is None:
        try:
            _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            _db = _client[DB_NAME]
            # Test connection
            _client.server_info()
            logger.info(f"Connected to MongoDB: {DB_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    return _db


# Track original states for compensation
_original_states = {}


@tool
def create_user(user_id: str, name: str, email: str, role: str = "user") -> str:
    """
    Create a new user in the database.
    
    Args:
        user_id: Unique identifier for the user
        name: User's full name
        email: User's email address
        role: User's role (default: "user")
    
    Returns:
        JSON string with user_id and status
    """
    try:
        db = get_db()
        users_collection = db["users"]
        
        # Check if user already exists
        if users_collection.find_one({"user_id": user_id}):
            return json.dumps({
                "status": "error",
                "message": f"User {user_id} already exists"
            })
        
        user_doc = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "role": role,
            "created_at": None  # Will be set by MongoDB
        }
        
        result = users_collection.insert_one(user_doc)
        
        # Store original state (empty, since this is a new user)
        _original_states[f"user_{user_id}"] = None
        
        logger.info(f"Created user: {user_id}")
        return json.dumps({
            "status": "success",
            "user_id": user_id,
            "inserted_id": str(result.inserted_id)
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error creating user: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Database error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def update_user_profile(user_id: str, updates: Dict[str, Any]) -> str:
    """
    Update a user's profile information.
    
    Args:
        user_id: User identifier
        updates: Dictionary of fields to update (e.g., {"name": "New Name", "email": "new@email.com"})
    
    Returns:
        JSON string with status and updated fields
    """
    try:
        db = get_db()
        users_collection = db["users"]
        
        # Get current state for compensation
        current_user = users_collection.find_one({"user_id": user_id})
        if not current_user:
            return json.dumps({
                "status": "error",
                "message": f"User {user_id} not found"
            })
        
        # Store original state for compensation
        state_key = f"profile_{user_id}"
        if state_key not in _original_states:
            _original_states[state_key] = {
                k: v for k, v in current_user.items() 
                if k in updates and k not in ["_id", "created_at"]
            }
        
        # Update user
        result = users_collection.update_one(
            {"user_id": user_id},
            {"$set": updates}
        )
        
        if result.matched_count == 0:
            return json.dumps({
                "status": "error",
                "message": f"User {user_id} not found"
            })
        
        logger.info(f"Updated profile for user: {user_id}")
        return json.dumps({
            "status": "success",
            "user_id": user_id,
            "updated_fields": list(updates.keys()),
            "modified_count": result.modified_count
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error updating profile: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Database error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def add_user_preferences(user_id: str, preferences: Dict[str, Any]) -> str:
    """
    Add preferences to a user's profile.
    
    Args:
        user_id: User identifier
        preferences: Dictionary of preferences (e.g., {"theme": "dark", "notifications": True})
    
    Returns:
        JSON string with status
    """
    try:
        db = get_db()
        users_collection = db["users"]
        
        # Check if user exists
        current_user = users_collection.find_one({"user_id": user_id})
        if not current_user:
            return json.dumps({
                "status": "error",
                "message": f"User {user_id} not found"
            })
        
        # Store original preferences for compensation
        state_key = f"preferences_{user_id}"
        original_prefs = current_user.get("preferences", {})
        if state_key not in _original_states:
            _original_states[state_key] = original_prefs.copy() if original_prefs else {}
        
        # Add preferences
        result = users_collection.update_one(
            {"user_id": user_id},
            {"$set": {"preferences": preferences}}
        )
        
        logger.info(f"Added preferences for user: {user_id}")
        return json.dumps({
            "status": "success",
            "user_id": user_id,
            "preferences": preferences
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error adding preferences: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Database error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error adding preferences: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def create_user_session(user_id: str, session_data: Dict[str, Any]) -> str:
    """
    Create a session for a user. This operation will fail if user has more than 5 active sessions.
    
    Args:
        user_id: User identifier
        session_data: Session information (e.g., {"device": "mobile", "ip": "192.168.1.1"})
    
    Returns:
        JSON string with status and session_id
    """
    try:
        db = get_db()
        sessions_collection = db["sessions"]
        
        # Check if user exists
        users_collection = db["users"]
        if not users_collection.find_one({"user_id": user_id}):
            return json.dumps({
                "status": "error",
                "message": f"User {user_id} not found"
            })
        
        # Check session limit (this will fail if > 5 sessions)
        active_sessions = sessions_collection.count_documents({
            "user_id": user_id,
            "status": "active"
        })
        
        if active_sessions >= 5:
            return json.dumps({
                "status": "error",
                "message": f"User {user_id} has reached maximum session limit (5 active sessions)"
            })
        
        # Create session
        session_doc = {
            "user_id": user_id,
            "session_id": f"session_{user_id}_{len(session_data)}",
            "status": "active",
            **session_data
        }
        
        result = sessions_collection.insert_one(session_doc)
        
        # Store original state (no sessions before)
        state_key = f"session_{result.inserted_id}"
        _original_states[state_key] = None
        
        logger.info(f"Created session for user: {user_id}")
        return json.dumps({
            "status": "success",
            "user_id": user_id,
            "session_id": session_doc["session_id"],
            "inserted_id": str(result.inserted_id)
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error creating session: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Database error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


# Compensation tools
@tool
def delete_user(user_id: str) -> str:
    """
    Delete a user from the database (compensation for create_user).
    
    Args:
        user_id: User identifier to delete
    
    Returns:
        JSON string with status
    """
    try:
        db = get_db()
        users_collection = db["users"]
        sessions_collection = db["sessions"]
        
        # Delete user's sessions first
        sessions_collection.delete_many({"user_id": user_id})
        
        # Delete user
        result = users_collection.delete_one({"user_id": user_id})
        
        if result.deleted_count == 0:
            return json.dumps({
                "status": "error",
                "message": f"User {user_id} not found"
            })
        
        logger.info(f"Deleted user: {user_id}")
        return json.dumps({
            "status": "success",
            "user_id": user_id,
            "message": "User deleted"
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error deleting user: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Database error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def revert_user_profile(user_id: str, original_state: Dict[str, Any]) -> str:
    """
    Revert a user's profile to its original state (compensation for update_user_profile).
    
    Args:
        user_id: User identifier
        original_state: Original state to restore
    
    Returns:
        JSON string with status
    """
    try:
        db = get_db()
        users_collection = db["users"]
        
        if not original_state:
            return json.dumps({
                "status": "success",
                "message": "No original state to restore"
            })
        
        # Revert to original state
        result = users_collection.update_one(
            {"user_id": user_id},
            {"$set": original_state}
        )
        
        logger.info(f"Reverted profile for user: {user_id}")
        return json.dumps({
            "status": "success",
            "user_id": user_id,
            "message": "Profile reverted"
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error reverting profile: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Database error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error reverting profile: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def remove_user_preferences(user_id: str) -> str:
    """
    Remove user preferences (compensation for add_user_preferences).
    
    Args:
        user_id: User identifier
    
    Returns:
        JSON string with status
    """
    try:
        db = get_db()
        users_collection = db["users"]
        
        # Get original preferences from stored state
        state_key = f"preferences_{user_id}"
        original_prefs = _original_states.get(state_key, {})
        
        # Restore original preferences or remove if none
        if original_prefs:
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"preferences": original_prefs}}
            )
        else:
            users_collection.update_one(
                {"user_id": user_id},
                {"$unset": {"preferences": ""}}
            )
        
        logger.info(f"Removed preferences for user: {user_id}")
        return json.dumps({
            "status": "success",
            "user_id": user_id,
            "message": "Preferences removed"
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error removing preferences: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Database error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error removing preferences: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def delete_user_session(session_id: str) -> str:
    """
    Delete a user session (compensation for create_user_session).
    
    Args:
        session_id: Session identifier to delete
    
    Returns:
        JSON string with status
    """
    try:
        db = get_db()
        sessions_collection = db["sessions"]
        
        result = sessions_collection.delete_one({"session_id": session_id})
        
        if result.deleted_count == 0:
            return json.dumps({
                "status": "error",
                "message": f"Session {session_id} not found"
            })
        
        logger.info(f"Deleted session: {session_id}")
        return json.dumps({
            "status": "success",
            "session_id": session_id,
            "message": "Session deleted"
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error deleting session: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Database error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def get_user_info(user_id: str) -> str:
    """
    Get user information from the database.
    
    Args:
        user_id: User identifier
    
    Returns:
        JSON string with user information
    """
    try:
        db = get_db()
        users_collection = db["users"]
        
        user = users_collection.find_one({"user_id": user_id}, {"_id": 0})
        
        if not user:
            return json.dumps({
                "status": "error",
                "message": f"User {user_id} not found"
            })
        
        return json.dumps({
            "status": "success",
            "user": user
        })
    except PyMongoError as e:
        logger.error(f"MongoDB error getting user info: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Database error: {str(e)}"
        })
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


# Helper function to get all MongoDB tools
def get_mongodb_tools():
    """Get all MongoDB tools for use in agents."""
    return [
        create_user,
        update_user_profile,
        add_user_preferences,
        create_user_session,
        delete_user,
        revert_user_profile,
        remove_user_preferences,
        delete_user_session,
        get_user_info,
    ]

