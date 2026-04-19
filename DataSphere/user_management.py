import streamlit as st
import pandas as pd
import hashlib
import hmac
import os
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

class UserManagement:
    def __init__(self):
        """Initialize the User Management module"""
        # Initialize session state for users and roles
        if "users" not in st.session_state:
            st.session_state.users = {
                "admin": {
                    "password_hash": self._hash_password("admin123"),
                    "full_name": "Administrator",
                    "email": "admin@example.com",
                    "role": "admin",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_login": None
                }
            }
        
        if "roles" not in st.session_state:
            st.session_state.roles = {
                "admin": {
                    "name": "Administrator",
                    "description": "Full access to all features",
                    "permissions": {
                        "database_connections": ["view", "create", "edit", "delete"],
                        "query_generator": ["view", "create", "edit", "delete", "execute"],
                        "schema_visualization": ["view"],
                        "query_optimization": ["view", "create", "edit", "execute"],
                        "schema_advisor": ["view", "create", "execute"],
                        "semantic_layer": ["view", "create", "edit", "delete"],
                        "advanced_visualization": ["view", "create", "edit", "delete"],
                        "enterprise_integration": ["view", "create", "edit", "delete"],
                        "cloud_storage": ["view", "create", "edit", "delete"],
                        "user_management": ["view", "create", "edit", "delete"]
                    }
                },
                "analyst": {
                    "name": "Data Analyst",
                    "description": "Can query data and create visualizations",
                    "permissions": {
                        "database_connections": ["view"],
                        "query_generator": ["view", "create", "edit", "execute"],
                        "schema_visualization": ["view"],
                        "query_optimization": ["view", "create", "execute"],
                        "schema_advisor": ["view"],
                        "semantic_layer": ["view"],
                        "advanced_visualization": ["view", "create", "edit"],
                        "enterprise_integration": ["view"],
                        "cloud_storage": ["view"],
                        "user_management": []
                    }
                },
                "viewer": {
                    "name": "Viewer",
                    "description": "Read-only access to dashboards and reports",
                    "permissions": {
                        "database_connections": ["view"],
                        "query_generator": ["view"],
                        "schema_visualization": ["view"],
                        "query_optimization": ["view"],
                        "schema_advisor": ["view"],
                        "semantic_layer": ["view"],
                        "advanced_visualization": ["view"],
                        "enterprise_integration": ["view"],
                        "cloud_storage": ["view"],
                        "user_management": []
                    }
                }
            }
        
        # Current user
        if "current_user" not in st.session_state:
            st.session_state.current_user = None
        
        # Authentication token
        if "auth_token" not in st.session_state:
            st.session_state.auth_token = None
        
        # Token expiration
        if "token_expiry" not in st.session_state:
            st.session_state.token_expiry = None
    
    def user_management_ui(self):
        """UI for user management and permissions"""
        st.subheader("User Management & Permissions")
        
        # Check if user is logged in and has admin permissions
        if not self._is_authenticated():
            self._login_ui()
            return
        
        # Check if current user has permission to access user management
        if not self._has_permission("user_management", "view"):
            st.error("You don't have permission to access User Management.")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs(["Users", "Roles & Permissions", "Audit Log", "Settings"])
        
        with tab1:
            self._users_ui()
        
        with tab2:
            self._roles_ui()
        
        with tab3:
            self._audit_log_ui()
        
        with tab4:
            self._settings_ui()
    
    def _login_ui(self):
        """UI for user login"""
        st.subheader("Login")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                submitted = st.form_submit_button("Login")
                if submitted and username and password:
                    if self._authenticate(username, password):
                        st.success(f"Welcome, {st.session_state.users[username]['full_name']}!")
                        
                        # Update last login time
                        st.session_state.users[username]["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Add to audit log
                        self._add_audit_log(username, "login", "User logged in")
                        
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
        
        with col2:
            st.info(
                "Use your administrator credentials to log in. "
                "If this is a fresh install, contact your system administrator for access."
            )
    
    def _users_ui(self):
        """UI for managing users"""
        st.subheader("Manage Users")
        
        # Create new user section
        with st.expander("Create New User", expanded=False):
            if not self._has_permission("user_management", "create"):
                st.warning("You don't have permission to create users.")
            else:
                with st.form("create_user_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    full_name = st.text_input("Full Name")
                    email = st.text_input("Email")
                    
                    # Role selection
                    role_options = list(st.session_state.roles.keys())
                    role = st.selectbox("Role", options=role_options)
                    
                    submitted = st.form_submit_button("Create User")
                    if submitted:
                        if not username or not password or not full_name or not email:
                            st.error("All fields are required.")
                        elif password != confirm_password:
                            st.error("Passwords do not match.")
                        elif username in st.session_state.users:
                            st.error(f"Username '{username}' already exists.")
                        else:
                            # Create new user
                            st.session_state.users[username] = {
                                "password_hash": self._hash_password(password),
                                "full_name": full_name,
                                "email": email,
                                "role": role,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "last_login": None
                            }
                            
                            st.success(f"User '{username}' created successfully.")
                            
                            # Add to audit log
                            self._add_audit_log(
                                st.session_state.current_user,
                                "create_user",
                                f"Created user: {username}"
                            )
                            
                            st.rerun()
        
        # List existing users
        st.subheader("Existing Users")
        
        if not st.session_state.users:
            st.info("No users found.")
        else:
            # Convert users to DataFrame for display
            users_data = []
            for username, user_data in st.session_state.users.items():
                users_data.append({
                    "Username": username,
                    "Full Name": user_data["full_name"],
                    "Email": user_data["email"],
                    "Role": user_data["role"],
                    "Created": user_data["created_at"],
                    "Last Login": user_data["last_login"] or "Never"
                })
            
            users_df = pd.DataFrame(users_data)
            st.dataframe(users_df)
            
            # User details and actions
            selected_user = st.selectbox(
                "Select User to Manage",
                options=list(st.session_state.users.keys()),
                key="selected_user"
            )
            
            if selected_user:
                user_data = st.session_state.users[selected_user]
                
                st.subheader(f"User Details: {selected_user}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Full Name:** {user_data['full_name']}")
                    st.write(f"**Email:** {user_data['email']}")
                    st.write(f"**Role:** {user_data['role']}")
                
                with col2:
                    st.write(f"**Created:** {user_data['created_at']}")
                    st.write(f"**Last Login:** {user_data['last_login'] or 'Never'}")
                
                # Actions
                st.subheader("Actions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Edit user
                    if self._has_permission("user_management", "edit"):
                        if st.button("Edit User", key=f"edit_{selected_user}"):
                            st.session_state.edit_user = selected_user
                            st.rerun()
                
                with col2:
                    # Reset password
                    if self._has_permission("user_management", "edit"):
                        if st.button("Reset Password", key=f"reset_{selected_user}"):
                            st.session_state.reset_password_user = selected_user
                            st.rerun()
                
                with col3:
                    # Delete user
                    if self._has_permission("user_management", "delete"):
                        if selected_user != st.session_state.current_user:  # Prevent deleting yourself
                            if st.button("Delete User", key=f"delete_{selected_user}"):
                                if st.session_state.current_user == "admin" or selected_user != "admin":  # Only admin can delete admin
                                    del st.session_state.users[selected_user]
                                    
                                    st.success(f"User '{selected_user}' deleted successfully.")
                                    
                                    # Add to audit log
                                    self._add_audit_log(
                                        st.session_state.current_user,
                                        "delete_user",
                                        f"Deleted user: {selected_user}"
                                    )
                                    
                                    st.rerun()
                                else:
                                    st.error("You cannot delete the admin user.")
                        else:
                            st.warning("You cannot delete your own account.")
                
                # Edit user form
                if hasattr(st.session_state, "edit_user") and st.session_state.edit_user == selected_user:
                    st.subheader(f"Edit User: {selected_user}")
                    
                    with st.form("edit_user_form"):
                        full_name = st.text_input("Full Name", value=user_data["full_name"])
                        email = st.text_input("Email", value=user_data["email"])
                        
                        # Role selection
                        role_options = list(st.session_state.roles.keys())
                        role = st.selectbox(
                            "Role",
                            options=role_options,
                            index=role_options.index(user_data["role"])
                        )
                        
                        submitted = st.form_submit_button("Save Changes")
                        if submitted:
                            if not full_name or not email:
                                st.error("All fields are required.")
                            else:
                                # Update user
                                st.session_state.users[selected_user]["full_name"] = full_name
                                st.session_state.users[selected_user]["email"] = email
                                st.session_state.users[selected_user]["role"] = role
                                
                                st.success(f"User '{selected_user}' updated successfully.")
                                
                                # Add to audit log
                                self._add_audit_log(
                                    st.session_state.current_user,
                                    "edit_user",
                                    f"Updated user: {selected_user}"
                                )
                                
                                # Clear edit state
                                del st.session_state.edit_user
                                
                                st.rerun()
                    
                    if st.button("Cancel Edit"):
                        del st.session_state.edit_user
                        st.rerun()
                
                # Reset password form
                if hasattr(st.session_state, "reset_password_user") and st.session_state.reset_password_user == selected_user:
                    st.subheader(f"Reset Password: {selected_user}")
                    
                    with st.form("reset_password_form"):
                        new_password = st.text_input("New Password", type="password")
                        confirm_password = st.text_input("Confirm Password", type="password")
                        
                        submitted = st.form_submit_button("Reset Password")
                        if submitted:
                            if not new_password:
                                st.error("Password is required.")
                            elif new_password != confirm_password:
                                st.error("Passwords do not match.")
                            else:
                                # Update password
                                st.session_state.users[selected_user]["password_hash"] = self._hash_password(new_password)
                                
                                st.success(f"Password for '{selected_user}' reset successfully.")
                                
                                # Add to audit log
                                self._add_audit_log(
                                    st.session_state.current_user,
                                    "reset_password",
                                    f"Reset password for user: {selected_user}"
                                )
                                
                                # Clear reset state
                                del st.session_state.reset_password_user
                                
                                st.rerun()
                    
                    if st.button("Cancel Reset"):
                        del st.session_state.reset_password_user
                        st.rerun()
    
    def _roles_ui(self):
        """UI for managing roles and permissions"""
        st.subheader("Roles & Permissions")
        
        # Create new role section
        with st.expander("Create New Role", expanded=False):
            if not self._has_permission("user_management", "create"):
                st.warning("You don't have permission to create roles.")
            else:
                with st.form("create_role_form"):
                    role_id = st.text_input("Role ID", placeholder="e.g., data_scientist")
                    role_name = st.text_input("Role Name", placeholder="e.g., Data Scientist")
                    role_description = st.text_area("Description", placeholder="Role description...")
                    
                    # Permission sections
                    st.subheader("Permissions")
                    
                    # Define modules and their possible permissions
                    modules = {
                        "database_connections": "Database Connections",
                        "query_generator": "Query Generator",
                        "schema_visualization": "Schema Visualization",
                        "query_optimization": "Query Optimization",
                        "schema_advisor": "Schema Advisor",
                        "semantic_layer": "Semantic Layer",
                        "advanced_visualization": "Advanced Visualization",
                        "enterprise_integration": "Enterprise Integration",
                        "cloud_storage": "Cloud Storage",
                        "user_management": "User Management"
                    }
                    
                    permissions = ["view", "create", "edit", "delete", "execute"]
                    
                    # Initialize permissions dictionary
                    role_permissions = {}
                    
                    # Create permission checkboxes for each module
                    for module_id, module_name in modules.items():
                        st.write(f"**{module_name}**")
                        
                        # Create a row of checkboxes for each permission type
                        cols = st.columns(len(permissions))
                        
                        module_perms = []
                        for i, perm in enumerate(permissions):
                            with cols[i]:
                                if st.checkbox(perm.capitalize(), key=f"{module_id}_{perm}"):
                                    module_perms.append(perm)
                        
                        role_permissions[module_id] = module_perms
                    
                    submitted = st.form_submit_button("Create Role")
                    if submitted:
                        if not role_id or not role_name:
                            st.error("Role ID and Name are required.")
                        elif role_id in st.session_state.roles:
                            st.error(f"Role ID '{role_id}' already exists.")
                        else:
                            # Create new role
                            st.session_state.roles[role_id] = {
                                "name": role_name,
                                "description": role_description,
                                "permissions": role_permissions
                            }
                            
                            st.success(f"Role '{role_name}' created successfully.")
                            
                            # Add to audit log
                            self._add_audit_log(
                                st.session_state.current_user,
                                "create_role",
                                f"Created role: {role_id}"
                            )
                            
                            st.rerun()
        
        # List existing roles
        st.subheader("Existing Roles")
        
        if not st.session_state.roles:
            st.info("No roles found.")
        else:
            # Convert roles to DataFrame for display
            roles_data = []
            for role_id, role_data in st.session_state.roles.items():
                # Count total permissions
                total_perms = sum(len(perms) for perms in role_data["permissions"].values())
                
                roles_data.append({
                    "Role ID": role_id,
                    "Name": role_data["name"],
                    "Description": role_data["description"],
                    "Total Permissions": total_perms
                })
            
            roles_df = pd.DataFrame(roles_data)
            st.dataframe(roles_df)
            
            # Role details and actions
            selected_role = st.selectbox(
                "Select Role to Manage",
                options=list(st.session_state.roles.keys()),
                key="selected_role"
            )
            
            if selected_role:
                role_data = st.session_state.roles[selected_role]
                
                st.subheader(f"Role Details: {role_data['name']}")
                st.write(f"**Description:** {role_data['description']}")
                
                # Display permissions
                st.subheader("Permissions")
                
                # Define modules and their possible permissions
                modules = {
                    "database_connections": "Database Connections",
                    "query_generator": "Query Generator",
                    "schema_visualization": "Schema Visualization",
                    "query_optimization": "Query Optimization",
                    "schema_advisor": "Schema Advisor",
                    "semantic_layer": "Semantic Layer",
                    "advanced_visualization": "Advanced Visualization",
                    "enterprise_integration": "Enterprise Integration",
                    "cloud_storage": "Cloud Storage",
                    "user_management": "User Management"
                }
                
                permissions = ["view", "create", "edit", "delete", "execute"]
                
                # Create a table-like display of permissions
                header_cols = st.columns([3] + [1] * len(permissions))
                with header_cols[0]:
                    st.write("**Module**")
                
                for i, perm in enumerate(permissions):
                    with header_cols[i + 1]:
                        st.write(f"**{perm.capitalize()}**")
                
                for module_id, module_name in modules.items():
                    row_cols = st.columns([3] + [1] * len(permissions))
                    
                    with row_cols[0]:
                        st.write(module_name)
                    
                    module_perms = role_data["permissions"].get(module_id, [])
                    
                    for i, perm in enumerate(permissions):
                        with row_cols[i + 1]:
                            if perm in module_perms:
                                st.write("✅")
                            else:
                                st.write("❌")
                
                # Actions
                st.subheader("Actions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Edit role
                    if self._has_permission("user_management", "edit"):
                        # Add a timestamp to make the key unique
                        edit_key = f"edit_{selected_role}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                        if st.button("Edit Role", key=edit_key):
                            st.session_state.edit_role = selected_role
                            st.rerun()
                
                with col2:
                    # Delete role
                    if self._has_permission("user_management", "delete"):
                        if selected_role != "admin":  # Prevent deleting admin role
                            # Add a timestamp to make the key unique
                            delete_key = f"delete_{selected_role}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                            if st.button("Delete Role", key=delete_key):
                                # Check if any users have this role
                                users_with_role = [username for username, user_data in st.session_state.users.items() 
                                                if user_data["role"] == selected_role]
                                
                                if users_with_role:
                                    st.error(f"Cannot delete role '{selected_role}' because it is assigned to {len(users_with_role)} users.")
                                else:
                                    del st.session_state.roles[selected_role]
                                    
                                    st.success(f"Role '{selected_role}' deleted successfully.")
                                    
                                    # Add to audit log
                                    self._add_audit_log(
                                        st.session_state.current_user,
                                        "delete_role",
                                        f"Deleted role: {selected_role}"
                                    )
                                    
                                    st.rerun()
                        else:
                            st.warning("You cannot delete the admin role.")
                
                # Edit role form
                if hasattr(st.session_state, "edit_role") and st.session_state.edit_role == selected_role:
                    st.subheader(f"Edit Role: {role_data['name']}")
                    
                    with st.form("edit_role_form"):
                        role_name = st.text_input("Role Name", value=role_data["name"])
                        role_description = st.text_area("Description", value=role_data["description"])
                        
                        # Permission sections
                        st.subheader("Permissions")
                        
                        # Define modules and their possible permissions
                        modules = {
                            "database_connections": "Database Connections",
                            "query_generator": "Query Generator",
                            "schema_visualization": "Schema Visualization",
                            "query_optimization": "Query Optimization",
                            "schema_advisor": "Schema Advisor",
                            "semantic_layer": "Semantic Layer",
                            "advanced_visualization": "Advanced Visualization",
                            "enterprise_integration": "Enterprise Integration",
                            "cloud_storage": "Cloud Storage",
                            "user_management": "User Management"
                        }
                        
                        permissions = ["view", "create", "edit", "delete", "execute"]
                        
                        # Initialize permissions dictionary
                        role_permissions = {}
                        
                        # Create permission checkboxes for each module
                        for module_id, module_name in modules.items():
                            st.write(f"**{module_name}**")
                            
                            # Create a row of checkboxes for each permission type
                            cols = st.columns(len(permissions))
                            
                            module_perms = []
                            current_perms = role_data["permissions"].get(module_id, [])
                            
                            for i, perm in enumerate(permissions):
                                with cols[i]:
                                    if st.checkbox(
                                        perm.capitalize(),
                                        value=perm in current_perms,
                                        key=f"edit_{module_id}_{perm}"
                                    ):
                                        module_perms.append(perm)
                            
                            role_permissions[module_id] = module_perms
                        
                        submitted = st.form_submit_button("Save Changes")
                        if submitted:
                            if not role_name:
                                st.error("Role Name is required.")
                            else:
                                # Update role
                                st.session_state.roles[selected_role]["name"] = role_name
                                st.session_state.roles[selected_role]["description"] = role_description
                                st.session_state.roles[selected_role]["permissions"] = role_permissions
                                
                                st.success(f"Role '{role_name}' updated successfully.")
                                
                                # Add to audit log
                                self._add_audit_log(
                                    st.session_state.current_user,
                                    "edit_role",
                                    f"Updated role: {selected_role}"
                                )
                                
                                # Clear edit state
                                del st.session_state.edit_role
                                
                                st.rerun()
                    
                    # Add a timestamp to make the key unique
                    cancel_key = f"cancel_edit_{selected_role}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
                    if st.button("Cancel Edit", key=cancel_key):
                        del st.session_state.edit_role
                        st.rerun()
    
    def _audit_log_ui(self):
        """UI for viewing audit logs"""
        st.subheader("Audit Log")
        
        # Initialize audit log if not exists
        if "audit_log" not in st.session_state:
            st.session_state.audit_log = []
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filter by user
            user_options = ["All Users"] + list(st.session_state.users.keys())
            filter_user = st.selectbox("Filter by User", options=user_options, key="filter_audit_user")
        
        with col2:
            # Filter by action
            action_options = ["All Actions"]
            if st.session_state.audit_log:
                action_options += list(set(log["action"] for log in st.session_state.audit_log))
            
            filter_action = st.selectbox("Filter by Action", options=action_options, key="filter_audit_action")
        
        with col3:
            # Filter by date
            date_options = ["All Dates", "Today", "Yesterday", "Last 7 Days", "Last 30 Days"]
            filter_date = st.selectbox("Filter by Date", options=date_options, key="filter_audit_date")
        
        # Apply filters
        filtered_logs = st.session_state.audit_log.copy()
        
        if filter_user != "All Users":
            filtered_logs = [log for log in filtered_logs if log["username"] == filter_user]
        
        if filter_action != "All Actions":
            filtered_logs = [log for log in filtered_logs if log["action"] == filter_action]
        
        if filter_date != "All Dates":
            now = datetime.now()
            
            if filter_date == "Today":
                start_date = datetime(now.year, now.month, now.day)
            elif filter_date == "Yesterday":
                start_date = datetime(now.year, now.month, now.day) - timedelta(days=1)
                end_date = datetime(now.year, now.month, now.day)
            elif filter_date == "Last 7 Days":
                start_date = now - timedelta(days=7)
            elif filter_date == "Last 30 Days":
                start_date = now - timedelta(days=30)
            
            if filter_date == "Yesterday":
                filtered_logs = [
                    log for log in filtered_logs 
                    if start_date <= datetime.strptime(log["timestamp"], "%Y-%m-%d %H:%M:%S") < end_date
                ]
            else:
                filtered_logs = [
                    log for log in filtered_logs 
                    if datetime.strptime(log["timestamp"], "%Y-%m-%d %H:%M:%S") >= start_date
                ]
        
        # Sort by timestamp (newest first)
        filtered_logs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Display logs
        if not filtered_logs:
            st.info("No audit logs found matching the filters.")
        else:
            # Convert to DataFrame for display
            logs_data = []
            for log in filtered_logs:
                logs_data.append({
                    "Timestamp": log["timestamp"],
                    "User": log["username"],
                    "Action": log["action"],
                    "Details": log["details"],
                    "IP Address": log.get("ip_address", "N/A")
                })
            
            logs_df = pd.DataFrame(logs_data)
            st.dataframe(logs_df)
            
            # Export options
            if st.button("Export Audit Log"):
                csv = logs_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def _settings_ui(self):
        """UI for user management settings"""
        st.subheader("User Management Settings")
        
        # Password policy
        st.write("**Password Policy**")
        
        # Initialize password policy if not exists
        if "password_policy" not in st.session_state:
            st.session_state.password_policy = {
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True,
                "max_age_days": 90,
                "history_count": 3
            }
        
        policy = st.session_state.password_policy
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_length = st.number_input("Minimum Password Length", min_value=6, max_value=20, value=policy["min_length"])
            require_uppercase = st.checkbox("Require Uppercase Letters", value=policy["require_uppercase"])
            require_lowercase = st.checkbox("Require Lowercase Letters", value=policy["require_lowercase"])
        
        with col2:
            require_numbers = st.checkbox("Require Numbers", value=policy["require_numbers"])
            require_special = st.checkbox("Require Special Characters", value=policy["require_special"])
            max_age_days = st.number_input("Password Expiry (days)", min_value=0, max_value=365, value=policy["max_age_days"])
            history_count = st.number_input("Password History Count", min_value=0, max_value=10, value=policy["history_count"])
        
        if st.button("Save Password Policy"):
            # Update password policy
            st.session_state.password_policy = {
                "min_length": min_length,
                "require_uppercase": require_uppercase,
                "require_lowercase": require_lowercase,
                "require_numbers": require_numbers,
                "require_special": require_special,
                "max_age_days": max_age_days,
                "history_count": history_count
            }
            
            st.success("Password policy updated successfully.")
            
            # Add to audit log
            self._add_audit_log(
                st.session_state.current_user,
                "update_password_policy",
                "Updated password policy settings"
            )
        
        # Session settings
        st.write("**Session Settings**")
        
        # Initialize session settings if not exists
        if "session_settings" not in st.session_state:
            st.session_state.session_settings = {
                "session_timeout_minutes": 30,
                "max_failed_attempts": 5,
                "lockout_duration_minutes": 15
            }
        
        session = st.session_state.session_settings
        
        col1, col2 = st.columns(2)
        
        with col1:
            session_timeout = st.number_input("Session Timeout (minutes)", min_value=5, max_value=240, value=session["session_timeout_minutes"])
        
        with col2:
            max_failed_attempts = st.number_input("Max Failed Login Attempts", min_value=1, max_value=10, value=session["max_failed_attempts"])
            lockout_duration = st.number_input("Account Lockout Duration (minutes)", min_value=5, max_value=1440, value=session["lockout_duration_minutes"])
        
        if st.button("Save Session Settings"):
            # Update session settings
            st.session_state.session_settings = {
                "session_timeout_minutes": session_timeout,
                "max_failed_attempts": max_failed_attempts,
                "lockout_duration_minutes": lockout_duration
            }
            
            st.success("Session settings updated successfully.")
            
            # Add to audit log
            self._add_audit_log(
                st.session_state.current_user,
                "update_session_settings",
                "Updated session settings"
            )
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using scrypt with a random salt.
        Format: hex(salt):hex(hash)  — salt is stored alongside the hash.
        scrypt is a memory-hard KDF that defeats GPU/rainbow-table attacks.
        """
        salt = os.urandom(32)
        hash_bytes = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=16384,  # CPU/memory cost — NIST recommended minimum
            r=8,
            p=1
        )
        return salt.hex() + ":" + hash_bytes.hex()

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against a stored scrypt hash.
        Also handles legacy plain SHA-256 hashes (no colon separator) so
        existing accounts keep working — they will be re-hashed on next login.
        """
        if ":" not in stored_hash:
            # Legacy SHA-256 hash — verify then trigger re-hash on next login
            return hmac.compare_digest(
                hashlib.sha256(password.encode()).hexdigest(),
                stored_hash
            )
        salt_hex, hash_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=16384,
            r=8,
            p=1
        )
        return hmac.compare_digest(actual, expected)

    def _authenticate(self, username: str, password: str) -> bool:
        """Authenticate a user"""
        if username not in st.session_state.users:
            return False

        user = st.session_state.users[username]

        if not self._verify_password(password, user["password_hash"]):
            return False

        # Upgrade legacy SHA-256 hash to scrypt transparently on login
        if ":" not in user["password_hash"]:
            st.session_state.users[username]["password_hash"] = self._hash_password(password)

        if True:  # password verified
            # Set current user
            st.session_state.current_user = username
            
            # Generate auth token
            st.session_state.auth_token = str(uuid.uuid4())
            
            # Set token expiry
            session_timeout = st.session_state.get("session_settings", {}).get("session_timeout_minutes", 30)
            st.session_state.token_expiry = datetime.now() + timedelta(minutes=session_timeout)
            
            return True
        
        return False
    
    def _is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        if not st.session_state.current_user or not st.session_state.auth_token:
            return False
        
        # Check token expiry
        if not st.session_state.token_expiry or datetime.now() > st.session_state.token_expiry:
            # Token expired, log out user
            st.session_state.current_user = None
            st.session_state.auth_token = None
            st.session_state.token_expiry = None
            return False
        
        # Extend token expiry
        session_timeout = st.session_state.get("session_settings", {}).get("session_timeout_minutes", 30)
        st.session_state.token_expiry = datetime.now() + timedelta(minutes=session_timeout)
        
        return True
    
    def _has_permission(self, module: str, permission: str) -> bool:
        """Check if current user has a specific permission"""
        if not st.session_state.current_user:
            return False
        
        # Admin has all permissions
        if st.session_state.current_user == "admin":
            return True
        
        user = st.session_state.users[st.session_state.current_user]
        role = user["role"]
        
        if role not in st.session_state.roles:
            return False
        
        role_permissions = st.session_state.roles[role]["permissions"]
        
        if module not in role_permissions:
            return False
        
        return permission in role_permissions[module]
    
    def _add_audit_log(self, username: str, action: str, details: str) -> None:
        """Add an entry to the audit log"""
        if "audit_log" not in st.session_state:
            st.session_state.audit_log = []
        
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "username": username,
            "action": action,
            "details": details,
            "ip_address": "127.0.0.1"  # In a real app, would get the actual IP
        }
        
        st.session_state.audit_log.append(log_entry)
    
    def logout(self) -> None:
        """Log out the current user"""
        if st.session_state.current_user:
            # Add to audit log
            self._add_audit_log(
                st.session_state.current_user,
                "logout",
                "User logged out"
            )
            
            # Clear user session
            st.session_state.current_user = None
            st.session_state.auth_token = None
            st.session_state.token_expiry = None
    
    def get_current_user(self) -> Optional[str]:
        """Get the current logged-in user"""
        if self._is_authenticated():
            return st.session_state.current_user
        return None
    
    def check_permission(self, module: str, permission: str) -> bool:
        """Public method to check permissions from other modules"""
        return self._has_permission(module, permission)