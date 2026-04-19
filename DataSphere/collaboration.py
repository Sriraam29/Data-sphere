import streamlit as st
import pandas as pd
import json
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple

class Collaboration:
    def __init__(self):
        """Initialize the Collaboration module"""
        # Initialize session state for collaboration features
        if "workspaces" not in st.session_state:
            st.session_state.workspaces = self._get_default_workspaces()
        
        if "comments" not in st.session_state:
            st.session_state.comments = {}
        
        if "versions" not in st.session_state:
            st.session_state.versions = {}
        
        if "notifications" not in st.session_state:
            st.session_state.notifications = []
        
        if "current_workspace" not in st.session_state:
            st.session_state.current_workspace = "Default"
    
    def collaboration_ui(self):
        """UI for collaboration features"""
        st.subheader("Collaboration Hub")
        
        # Check if user is logged in
        if not st.session_state.get("current_user"):
            st.warning("Please log in to use collaboration features.")
            return
        
        # Create tabs for different collaboration features
        tab1, tab2, tab3, tab4 = st.tabs([
            "Workspaces", 
            "Comments & Annotations", 
            "Version Control", 
            "Notifications"
        ])
        
        with tab1:
            self._workspaces_ui()
        
        with tab2:
            self._comments_ui()
        
        with tab3:
            self._version_control_ui()
        
        with tab4:
            self._notifications_ui()
    
    def _workspaces_ui(self):
        """UI for managing shared workspaces"""
        st.subheader("Shared Workspaces")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display current workspace
            current_ws = st.session_state.workspaces.get(st.session_state.current_workspace, {})
            st.write(f"**Current Workspace:** {st.session_state.current_workspace}")
            
            if current_ws:
                st.write(f"**Description:** {current_ws.get('description', 'No description')}")
                st.write(f"**Created by:** {current_ws.get('created_by', 'Unknown')}")
                st.write(f"**Created on:** {current_ws.get('created_at', 'Unknown')}")
                
                # Display members
                st.write("**Members:**")
                members = current_ws.get("members", [])
                for member in members:
                    st.write(f"- {member}")
            
            # Display workspace content
            st.subheader("Workspace Content")
            
            # Get content from the current workspace
            queries = current_ws.get("queries", {})
            models = current_ws.get("models", {})
            dashboards = current_ws.get("dashboards", {})
            
            if not queries and not models and not dashboards:
                st.info("This workspace is empty. Add content using the sidebar options.")
            else:
                # Create tabs for different content types
                content_tab1, content_tab2, content_tab3 = st.tabs(["Queries", "Models", "Dashboards"])
                
                with content_tab1:
                    if not queries:
                        st.info("No queries in this workspace.")
                    else:
                        for query_id, query_data in queries.items():
                            with st.expander(f"{query_data['name']} (by {query_data['created_by']})"):
                                st.write(f"**Description:** {query_data.get('description', 'No description')}")
                                st.write(f"**Created:** {query_data.get('created_at', 'Unknown')}")
                                st.write(f"**Last modified:** {query_data.get('modified_at', 'Never')}")
                                
                                # Display SQL
                                st.code(query_data.get("sql", ""), language="sql")
                                
                                # Action buttons
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("Open in Query Editor", key=f"open_query_{query_id}"):
                                        # Set the query in the query editor
                                        st.session_state.current_query = query_data.get("sql", "")
                                        st.session_state.query_name = query_data.get("name", "")
                                        st.session_state.query_description = query_data.get("description", "")
                                        st.success(f"Query '{query_data['name']}' loaded in Query Editor.")
                                
                                with col2:
                                    if st.button("Add Comment", key=f"comment_query_{query_id}"):
                                        st.session_state.commenting_on = {
                                            "type": "query",
                                            "id": query_id,
                                            "name": query_data['name']
                                        }
                                
                                with col3:
                                    if st.button("View History", key=f"history_query_{query_id}"):
                                        st.session_state.viewing_history = {
                                            "type": "query",
                                            "id": query_id,
                                            "name": query_data['name']
                                        }
                
                with content_tab2:
                    if not models:
                        st.info("No semantic models in this workspace.")
                    else:
                        for model_id, model_data in models.items():
                            with st.expander(f"{model_data['name']} (by {model_data['created_by']})"):
                                st.write(f"**Description:** {model_data.get('description', 'No description')}")
                                st.write(f"**Created:** {model_data.get('created_at', 'Unknown')}")
                                st.write(f"**Last modified:** {model_data.get('modified_at', 'Never')}")
                                
                                # Display entities
                                st.write(f"**Entities:** {', '.join(model_data.get('entities', {}).keys())}")
                                
                                # Action buttons
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("Open in Semantic Layer", key=f"open_model_{model_id}"):
                                        # Set the model in the semantic layer
                                        st.success(f"Model '{model_data['name']}' loaded in Semantic Layer.")
                                
                                with col2:
                                    if st.button("Add Comment", key=f"comment_model_{model_id}"):
                                        st.session_state.commenting_on = {
                                            "type": "model",
                                            "id": model_id,
                                            "name": model_data['name']
                                        }
                                
                                with col3:
                                    if st.button("View History", key=f"history_model_{model_id}"):
                                        st.session_state.viewing_history = {
                                            "type": "model",
                                            "id": model_id,
                                            "name": model_data['name']
                                        }
                
                with content_tab3:
                    if not dashboards:
                        st.info("No dashboards in this workspace.")
                    else:
                        for dashboard_id, dashboard_data in dashboards.items():
                            with st.expander(f"{dashboard_data['name']} (by {dashboard_data['created_by']})"):
                                st.write(f"**Description:** {dashboard_data.get('description', 'No description')}")
                                st.write(f"**Created:** {dashboard_data.get('created_at', 'Unknown')}")
                                st.write(f"**Last modified:** {dashboard_data.get('modified_at', 'Never')}")
                                
                                # Display visualizations
                                st.write(f"**Visualizations:** {len(dashboard_data.get('visualizations', []))}")
                                
                                # Action buttons
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("Open Dashboard", key=f"open_dashboard_{dashboard_id}"):
                                        st.success(f"Dashboard '{dashboard_data['name']}' opened.")
                                
                                with col2:
                                    if st.button("Add Comment", key=f"comment_dashboard_{dashboard_id}"):
                                        st.session_state.commenting_on = {
                                            "type": "dashboard",
                                            "id": dashboard_id,
                                            "name": dashboard_data['name']
                                        }
                                
                                with col3:
                                    if st.button("View History", key=f"history_dashboard_{dashboard_id}"):
                                        st.session_state.viewing_history = {
                                            "type": "dashboard",
                                            "id": dashboard_id,
                                            "name": dashboard_data['name']
                                        }
        
        with col2:
            # Workspace selector and management
            st.subheader("Workspaces")
            
            # List available workspaces
            workspace_options = list(st.session_state.workspaces.keys())
            selected_workspace = st.selectbox(
                "Select Workspace",
                options=workspace_options,
                index=workspace_options.index(st.session_state.current_workspace),
                key="workspace_selector"
            )
            
            # Switch workspace button
            if st.button("Switch Workspace", key="switch_workspace_btn"):
                st.session_state.current_workspace = selected_workspace
                st.success(f"Switched to workspace: {selected_workspace}")
                st.rerun()
            
            # Create new workspace
            st.subheader("Create New Workspace")
            
            with st.form("create_workspace_form"):
                workspace_name = st.text_input("Workspace Name", key="new_workspace_name")
                workspace_description = st.text_area("Description", key="new_workspace_description")
                
                # Member selection (from users)
                if "users" in st.session_state:
                    user_options = list(st.session_state.users.keys())
                    selected_members = st.multiselect(
                        "Members",
                        options=user_options,
                        default=[st.session_state.current_user],
                        key="new_workspace_members"
                    )
                else:
                    selected_members = [st.session_state.current_user]
                
                submitted = st.form_submit_button("Create Workspace")
                if submitted and workspace_name:
                    if workspace_name in st.session_state.workspaces:
                        st.error(f"Workspace '{workspace_name}' already exists.")
                    else:
                        # Create new workspace
                        st.session_state.workspaces[workspace_name] = {
                            "description": workspace_description,
                            "created_by": st.session_state.current_user,
                            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "members": selected_members,
                            "queries": {},
                            "models": {},
                            "dashboards": {}
                        }
                        
                        # Switch to the new workspace
                        st.session_state.current_workspace = workspace_name
                        
                        # Add notification
                        self._add_notification(
                            f"New workspace '{workspace_name}' created by {st.session_state.current_user}",
                            "workspace_created",
                            workspace_name,
                            selected_members
                        )
                        
                        st.success(f"Workspace '{workspace_name}' created successfully.")
                        st.rerun()
    
    def _comments_ui(self):
        """UI for comments and annotations"""
        st.subheader("Comments & Annotations")
        
        # Check if user is adding a comment
        if hasattr(st.session_state, "commenting_on"):
            item = st.session_state.commenting_on
            st.write(f"Adding comment to: **{item['name']}** ({item['type']})")
            
            with st.form("add_comment_form"):
                comment_text = st.text_area("Comment", key="new_comment_text")
                is_private = st.checkbox("Private (only visible to you)", key="new_comment_private")
                
                submitted = st.form_submit_button("Add Comment")
                if submitted and comment_text:
                    # Create comment ID
                    comment_id = str(uuid.uuid4())
                    
                    # Initialize comments for this item if not exists
                    item_key = f"{item['type']}_{item['id']}"
                    if item_key not in st.session_state.comments:
                        st.session_state.comments[item_key] = []
                    
                    # Add the comment
                    st.session_state.comments[item_key].append({
                        "id": comment_id,
                        "text": comment_text,
                        "created_by": st.session_state.current_user,
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "private": is_private
                    })
                    
                    # Add notification if not private
                    if not is_private:
                        # Get workspace members
                        workspace = st.session_state.workspaces.get(st.session_state.current_workspace, {})
                        members = workspace.get("members", [])
                        
                        # Add notification
                        self._add_notification(
                            f"New comment on {item['type']} '{item['name']}' by {st.session_state.current_user}",
                            "new_comment",
                            item_key,
                            members
                        )
                    
                    st.success("Comment added successfully.")
                    
                    # Clear the commenting state
                    del st.session_state.commenting_on
                    st.rerun()
            
            # Cancel button
            if st.button("Cancel", key="cancel_comment_btn"):
                del st.session_state.commenting_on
                st.rerun()
        
        # Display comments for items in the current workspace
        st.subheader("Recent Comments")
        
        # Get current workspace content
        workspace = st.session_state.workspaces.get(st.session_state.current_workspace, {})
        
        # Collect all items in the workspace
        all_items = []
        
        # Add queries
        for query_id, query_data in workspace.get("queries", {}).items():
            all_items.append({
                "type": "query",
                "id": query_id,
                "name": query_data["name"],
                "key": f"query_{query_id}"
            })
        
        # Add models
        for model_id, model_data in workspace.get("models", {}).items():
            all_items.append({
                "type": "model",
                "id": model_id,
                "name": model_data["name"],
                "key": f"model_{model_id}"
            })
        
        # Add dashboards
        for dashboard_id, dashboard_data in workspace.get("dashboards", {}).items():
            all_items.append({
                "type": "dashboard",
                "id": dashboard_id,
                "name": dashboard_data["name"],
                "key": f"dashboard_{dashboard_id}"
            })
        
        # Display comments for each item
        if not all_items:
            st.info("No items with comments in this workspace.")
        else:
            # Filter for items with comments
            items_with_comments = []
            for item in all_items:
                if item["key"] in st.session_state.comments:
                    # Filter comments visible to current user
                    visible_comments = [
                        c for c in st.session_state.comments[item["key"]]
                        if not c["private"] or c["created_by"] == st.session_state.current_user
                    ]
                    
                    if visible_comments:
                        items_with_comments.append({
                            **item,
                            "comments": visible_comments
                        })
            
            if not items_with_comments:
                st.info("No comments found in this workspace.")
            else:
                # Sort items by most recent comment
                items_with_comments.sort(
                    key=lambda x: max([c["created_at"] for c in x["comments"]]),
                    reverse=True
                )
                
                # Display comments
                for item in items_with_comments:
                    with st.expander(f"{item['name']} ({item['type']})"):
                        # Sort comments by creation time (newest first)
                        sorted_comments = sorted(
                            item["comments"],
                            key=lambda x: x["created_at"],
                            reverse=True
                        )
                        
                        for comment in sorted_comments:
                            st.markdown(
                                f"""
                                <div style="border-left: 3px solid #ccc; padding-left: 10px; margin-bottom: 10px;">
                                    <p><strong>{comment['created_by']}</strong> • {comment['created_at']} 
                                    {' • <em>Private</em>' if comment['private'] else ''}</p>
                                    <p>{comment['text']}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Delete comment option (only for own comments)
                            if comment["created_by"] == st.session_state.current_user:
                                if st.button("Delete", key=f"delete_comment_{comment['id']}"):
                                    # Remove the comment
                                    st.session_state.comments[item["key"]].remove(comment)
                                    st.success("Comment deleted.")
                                    st.rerun()
    
    def _version_control_ui(self):
        """UI for version control of queries and models"""
        st.subheader("Version Control")
        
        # Check if user is viewing history
        if hasattr(st.session_state, "viewing_history"):
            item = st.session_state.viewing_history
            st.write(f"Version history for: **{item['name']}** ({item['type']})")
            
            # Get versions for this item
            item_key = f"{item['type']}_{item['id']}"
            versions = st.session_state.versions.get(item_key, [])
            
            if not versions:
                st.info("No version history available for this item.")
            else:
                # Sort versions by creation time (newest first)
                sorted_versions = sorted(
                    versions,
                    key=lambda x: x["created_at"],
                    reverse=True
                )
                
                # Display versions
                for i, version in enumerate(sorted_versions):
                    with st.expander(f"Version {len(sorted_versions) - i} • {version['created_at']} by {version['created_by']}"):
                        st.write(f"**Change message:** {version.get('message', 'No message')}")
                        
                        # Display content based on type
                        if item['type'] == "query":
                            st.code(version.get("content", {}).get("sql", ""), language="sql")
                        elif item['type'] == "model":
                            st.write(f"**Entities:** {', '.join(version.get('content', {}).get('entities', {}).keys())}")
                        elif item['type'] == "dashboard":
                            st.write(f"**Visualizations:** {len(version.get('content', {}).get('visualizations', []))}")
                        
                        # Restore version button
                        if i > 0:  # Only allow restoring older versions
                            if st.button("Restore this version", key=f"restore_version_{version['id']}"):
                                # Implement restore logic here
                                st.success(f"Version restored. The {item['type']} has been updated.")
                                
                                # Add notification
                                workspace = st.session_state.workspaces.get(st.session_state.current_workspace, {})
                                members = workspace.get("members", [])
                                
                                self._add_notification(
                                    f"{item['type'].capitalize()} '{item['name']}' restored to previous version by {st.session_state.current_user}",
                                    "version_restored",
                                    item_key,
                                    members
                                )
            
            # Back button
            if st.button("Back", key="back_from_history_btn"):
                del st.session_state.viewing_history
                st.rerun()
        
        else:
            # Display version control options
            st.write("Select an item from the Workspaces tab to view its version history.")
            
            # Version control settings
            st.subheader("Version Control Settings")
            
            auto_versioning = st.checkbox(
                "Enable automatic versioning",
                value=True,
                key="auto_versioning_checkbox",
                help="Automatically create a new version when an item is modified"
            )
            
            max_versions = st.slider(
                "Maximum versions to keep per item",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                key="max_versions_slider",
                help="Older versions will be automatically pruned"
            )
            
            # Save settings button
            if st.button("Save Settings", key="save_version_settings_btn"):
                st.session_state.version_control_settings = {
                    "auto_versioning": auto_versioning,
                    "max_versions": max_versions
                }
                st.success("Version control settings saved.")
    
    def _notifications_ui(self):
        """UI for notifications"""
        st.subheader("Notifications")
        
        # Filter options
        col1, col2 = st.columns([3, 1])
        
        with col1:
            filter_type = st.multiselect(
                "Filter by type",
                options=["All", "Comments", "Versions", "Workspaces", "Shares"],
                default=["All"],
                key="notification_filter_type"
            )
        
        with col2:
            mark_all_read = st.button("Mark all as read", key="mark_all_read_btn")
            if mark_all_read:
                # Mark all notifications as read
                for notification in st.session_state.notifications:
                    if notification["for_user"] == st.session_state.current_user and not notification["read"]:
                        notification["read"] = True
                st.success("All notifications marked as read.")
                st.rerun()
        
        # Get notifications for current user
        user_notifications = [
            n for n in st.session_state.notifications
            if n["for_user"] == st.session_state.current_user
        ]
        
        # Apply filters
        if "All" not in filter_type:
            filtered_notifications = []
            for n in user_notifications:
                if "Comments" in filter_type and n["type"] == "new_comment":
                    filtered_notifications.append(n)
                elif "Versions" in filter_type and n["type"] in ["version_created", "version_restored"]:
                    filtered_notifications.append(n)
                elif "Workspaces" in filter_type and n["type"] in ["workspace_created", "workspace_updated"]:
                    filtered_notifications.append(n)
                elif "Shares" in filter_type and n["type"] in ["item_shared"]:
                    filtered_notifications.append(n)
        else:
            filtered_notifications = user_notifications
        
        # Sort notifications by time (newest first)
        sorted_notifications = sorted(
            filtered_notifications,
            key=lambda x: x["created_at"],
            reverse=True
        )
        
        # Display notifications
        if not sorted_notifications:
            st.info("No notifications to display.")
        else:
            for notification in sorted_notifications:
                # Create a background color based on read status
                bg_color = "#f0f0f0" if notification["read"] else "#e6f3ff"
                
                # Create a notification card
                st.markdown(
                    f"""
                    <div style="background-color: {bg_color}; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                        <p><strong>{notification['message']}</strong></p>
                        <p style="color: #666; font-size: 0.8em;">{notification['created_at']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Mark as read button (if unread)
                if not notification["read"]:
                    if st.button("Mark as read", key=f"mark_read_{notification['id']}"):
                        notification["read"] = True
                        st.success("Notification marked as read.")
                        st.rerun()
    
    def add_to_workspace(self, item_type: str, item_id: str, item_data: Dict[str, Any]) -> bool:
        """Add an item to the current workspace
        
        Args:
            item_type: Type of item ('query', 'model', or 'dashboard')
            item_id: Unique identifier for the item
            item_data: Item data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not st.session_state.current_workspace:
            return False
        
        # Get current workspace
        workspace = st.session_state.workspaces.get(st.session_state.current_workspace)
        if not workspace:
            return False
        
        # Add item to workspace
        if item_type == "query":
            workspace["queries"][item_id] = item_data
        elif item_type == "model":
            workspace["models"][item_id] = item_data
        elif item_type == "dashboard":
            workspace["dashboards"][item_id] = item_data
        else:
            return False
        
        # Create initial version
        self.create_version(
            item_type, 
            item_id, 
            item_data, 
            "Initial version"
        )
        
        # Add notification
        members = workspace.get("members", [])
        self._add_notification(
            f"New {item_type} '{item_data['name']}' added to workspace '{st.session_state.current_workspace}' by {st.session_state.current_user}",
            "item_added",
            f"{item_type}_{item_id}",
            members
        )
        
        return True
    
    def create_version(self, item_type: str, item_id: str, content: Dict[str, Any], message: str) -> str:
        """Create a new version of an item
        
        Args:
            item_type: Type of item ('query', 'model', or 'dashboard')
            item_id: Unique identifier for the item
            content: Item content to version
            message: Version message
            
        Returns:
            str: Version ID
        """
        # Create version ID
        version_id = str(uuid.uuid4())
        
        # Initialize versions for this item if not exists
        item_key = f"{item_type}_{item_id}"
        if item_key not in st.session_state.versions:
            st.session_state.versions[item_key] = []
        
        # Add the version
        st.session_state.versions[item_key].append({
            "id": version_id,
            "created_by": st.session_state.current_user,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
            "content": content
        })
        
        # Prune old versions if needed
        max_versions = st.session_state.get("version_control_settings", {}).get("max_versions", 20)
        if len(st.session_state.versions[item_key]) > max_versions:
            # Sort by creation time (oldest first)
            sorted_versions = sorted(
                st.session_state.versions[item_key],
                key=lambda x: x["created_at"]
            )
            
            # Remove oldest versions
            st.session_state.versions[item_key] = sorted_versions[-(max_versions):]
        
        return version_id
    
    def _add_notification(self, message: str, notification_type: str, item_key: str, users: List[str]) -> str:
        """Add a notification for users
        
        Args:
            message: Notification message
            notification_type: Type of notification
            item_key: Key of the related item
            users: List of users to notify
            
        Returns:
            str: Notification ID
        """
        # Create notification ID
        notification_id = str(uuid.uuid4())
        
        # Add notification for each user
        for user in users:
            # Skip the current user (who triggered the notification)
            if user == st.session_state.current_user:
                continue
                
            st.session_state.notifications.append({
                "id": notification_id,
                "message": message,
                "type": notification_type,
                "item_key": item_key,
                "for_user": user,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "created_by": st.session_state.current_user,
                "read": False
            })
        
        return notification_id
    
    def _get_default_workspaces(self) -> Dict[str, Any]:
        """Get default workspaces
        
        Returns:
            Dict[str, Any]: Default workspaces
        """
        return {
            "Default": {
                "description": "Default workspace for all users",
                "created_by": "system",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "members": ["admin"],
                "queries": {
                    "sample_query_1": {
                        "name": "Sample Customer Query",
                        "description": "Query to analyze customer data",
                        "created_by": "admin",
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "modified_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "sql": "SELECT customer_id, name, email, created_at, last_order_date\nFROM customers\nWHERE status = 'active'\nORDER BY last_order_date DESC\nLIMIT 100;"
                    }
                },
                "models": {},
                "dashboards": {}
            },
            "Sales Team": {
                "description": "Workspace for the sales team",
                "created_by": "admin",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "members": ["admin"],
                "queries": {
                    "sales_query_1": {
                        "name": "Monthly Sales Report",
                        "description": "Query to generate monthly sales report",
                        "created_by": "admin",
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "modified_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "sql": "SELECT\n    DATE_TRUNC('month', order_date) AS month,\n    COUNT(*) AS order_count,\n    SUM(total_amount) AS total_sales,\n    AVG(total_amount) AS avg_order_value\nFROM orders\nWHERE order_date >= DATE_TRUNC('year', CURRENT_DATE)\nGROUP BY DATE_TRUNC('month', order_date)\nORDER BY month;"
                    }
                },
                "models": {},
                "dashboards": {}
            }
        }