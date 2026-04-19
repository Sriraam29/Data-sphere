import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import json
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

class DataStorytelling:
    def __init__(self):
        """Initialize the Data Storytelling module"""
        # Initialize session state for data stories
        if "data_stories" not in st.session_state:
            st.session_state.data_stories = self._get_default_stories()
        
        if "current_story" not in st.session_state:
            st.session_state.current_story = None
        
        if "story_edit_mode" not in st.session_state:
            st.session_state.story_edit_mode = False
        
        if "presentation_mode" not in st.session_state:
            st.session_state.presentation_mode = False
        
        if "current_slide" not in st.session_state:
            st.session_state.current_slide = 0
    
    def storytelling_ui(self, db_manager, advanced_visualization):
        """UI for data storytelling features"""
        st.subheader("Advanced Data Storytelling")
        
        # Check if user is logged in
        if not st.session_state.get("current_user"):
            st.warning("Please log in to use data storytelling features.")
            return
        
        # Check if in presentation mode
        if st.session_state.presentation_mode:
            self._presentation_mode_ui()
            return
        
        # Create tabs for different storytelling features
        tab1, tab2, tab3, tab4 = st.tabs([
            "My Stories", 
            "Create Story", 
            "Story Library", 
            "Settings"
        ])
        
        with tab1:
            self._my_stories_ui()
        
        with tab2:
            self._create_story_ui(db_manager, advanced_visualization)
        
        with tab3:
            self._story_library_ui()
        
        with tab4:
            self._storytelling_settings_ui()
    
    def _my_stories_ui(self):
        """UI for managing user's data stories"""
        st.subheader("My Data Stories")
        
        # Filter stories by current user
        current_user = st.session_state.get("current_user")
        user_stories = {
            story_id: story for story_id, story in st.session_state.data_stories.items()
            if story.get("created_by") == current_user
        }
        
        if not user_stories:
            st.info("You haven't created any data stories yet. Go to the 'Create Story' tab to get started.")
        else:
            # Display user's stories
            for story_id, story in user_stories.items():
                with st.expander(f"{story['title']} ({len(story['slides'])} slides)"):
                    st.write(f"**Description:** {story.get('description', 'No description')}")
                    st.write(f"**Created:** {story.get('created_at', 'Unknown')}")
                    st.write(f"**Last modified:** {story.get('modified_at', 'Never')}")
                    
                    # Display story tags
                    if story.get("tags"):
                        st.write("**Tags:** " + ", ".join(story["tags"]))
                    
                    # Display story thumbnail if available
                    if story.get("thumbnail"):
                        st.image(story["thumbnail"], width=300)
                    
                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("Edit", key=f"edit_story_{story_id}"):
                            st.session_state.current_story = story_id
                            st.session_state.story_edit_mode = True
                            st.rerun()
                    
                    with col2:
                        if st.button("Present", key=f"present_story_{story_id}"):
                            st.session_state.current_story = story_id
                            st.session_state.presentation_mode = True
                            st.session_state.current_slide = 0
                            st.rerun()
                    
                    with col3:
                        if st.button("Share", key=f"share_story_{story_id}"):
                            # Check if collaboration module is available
                            if "collaboration" in st.session_state:
                                # Add to workspace
                                if st.session_state.collaboration.add_to_workspace("story", story_id, story):
                                    st.success(f"Story '{story['title']}' shared to workspace '{st.session_state.current_workspace}'.")
                                else:
                                    st.error("Failed to share story to workspace.")
                            else:
                                st.error("Collaboration module not available.")
                    
                    with col4:
                        if st.button("Delete", key=f"delete_story_{story_id}"):
                            # Confirm deletion
                            if st.session_state.get("confirm_delete_story") == story_id:
                                del st.session_state.data_stories[story_id]
                                st.success(f"Story '{story['title']}' deleted.")
                                st.rerun()
                            else:
                                st.session_state.confirm_delete_story = story_id
                                st.warning(f"Are you sure you want to delete '{story['title']}'? Click Delete again to confirm.")
    
    def _create_story_ui(self, db_manager, advanced_visualization):
        """UI for creating and editing data stories"""
        # Check if editing an existing story
        if st.session_state.story_edit_mode and st.session_state.current_story:
            story = st.session_state.data_stories[st.session_state.current_story]
            st.subheader(f"Edit Story: {story['title']}")
            
            # Story metadata
            with st.expander("Story Details", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    story_title = st.text_input(
                        "Story Title",
                        value=story.get("title", ""),
                        key="edit_story_title"
                    )
                
                with col2:
                    story_tags = st.text_input(
                        "Tags (comma separated)",
                        value=",".join(story.get("tags", [])),
                        key="edit_story_tags"
                    )
                
                story_description = st.text_area(
                    "Description",
                    value=story.get("description", ""),
                    key="edit_story_description"
                )
                
                # Update story metadata
                if st.button("Update Story Details"):
                    story["title"] = story_title
                    story["description"] = story_description
                    story["tags"] = [tag.strip() for tag in story_tags.split(",") if tag.strip()]
                    story["modified_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("Story details updated.")
            
            # Slide management
            st.subheader("Slides")
            
            # Add new slide button
            if st.button("Add New Slide"):
                new_slide_id = str(uuid.uuid4())
                story["slides"][new_slide_id] = {
                    "title": f"Slide {len(story['slides']) + 1}",
                    "content": "",
                    "layout": "text",
                    "order": len(story["slides"]),
                    "annotations": [],
                    "visualizations": [],
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                story["modified_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("New slide added.")
                st.rerun()
            
            # Sort slides by order
            sorted_slides = sorted(
                story["slides"].items(),
                key=lambda x: x[1].get("order", 0)
            )
            
            # Display slides
            for slide_id, slide in sorted_slides:
                with st.expander(f"Slide {slide.get('order', 0) + 1}: {slide.get('title', 'Untitled')}"):
                    # Slide title
                    slide_title = st.text_input(
                        "Slide Title",
                        value=slide.get("title", ""),
                        key=f"slide_title_{slide_id}"
                    )
                    
                    # Slide layout
                    layout_options = [
                        "text", "text_image", "image_text", "text_chart",
                        "chart_text", "full_image", "full_chart", "two_charts"
                    ]
                    slide_layout = st.selectbox(
                        "Layout",
                        options=layout_options,
                        index=layout_options.index(slide.get("layout", "text")),
                        key=f"slide_layout_{slide_id}"
                    )
                    
                    # Slide content
                    slide_content = st.text_area(
                        "Content (Markdown supported)",
                        value=slide.get("content", ""),
                        height=150,
                        key=f"slide_content_{slide_id}"
                    )
                    
                    # Visualizations
                    if slide_layout in ["text_image", "image_text", "full_image"]:
                        st.subheader("Image")
                        image_url = st.text_input(
                            "Image URL",
                            value=slide.get("image_url", ""),
                            key=f"slide_image_url_{slide_id}"
                        )
                        
                        if image_url:
                            st.image(image_url, caption="Preview", width=300)
                    
                    if slide_layout in ["text_chart", "chart_text", "full_chart", "two_charts"]:
                        st.subheader("Chart")
                        
                        # Query selection for chart data
                        if "query_results" in st.session_state and isinstance(st.session_state.query_results, pd.DataFrame):
                            st.write("Using current query results for visualization")
                            df = st.session_state.query_results
                            
                            # Chart type
                            chart_type = st.selectbox(
                                "Chart Type",
                                options=["bar", "line", "scatter", "pie", "area", "heatmap"],
                                index=0,
                                key=f"chart_type_{slide_id}"
                            )
                            
                            # Chart configuration
                            if chart_type in ["bar", "line", "scatter", "area"]:
                                x_col = st.selectbox(
                                    "X-axis",
                                    options=df.columns.tolist(),
                                    index=0,
                                    key=f"x_col_{slide_id}"
                                )
                                
                                y_col = st.selectbox(
                                    "Y-axis",
                                    options=df.columns.tolist(),
                                    index=min(1, len(df.columns) - 1),
                                    key=f"y_col_{slide_id}"
                                )
                                
                                # Generate chart
                                if chart_type == "bar":
                                    chart = alt.Chart(df).mark_bar().encode(
                                        x=x_col,
                                        y=y_col,
                                        tooltip=[x_col, y_col]
                                    ).properties(width=600, height=400)
                                elif chart_type == "line":
                                    chart = alt.Chart(df).mark_line().encode(
                                        x=x_col,
                                        y=y_col,
                                        tooltip=[x_col, y_col]
                                    ).properties(width=600, height=400)
                                elif chart_type == "scatter":
                                    chart = alt.Chart(df).mark_circle().encode(
                                        x=x_col,
                                        y=y_col,
                                        tooltip=[x_col, y_col]
                                    ).properties(width=600, height=400)
                                elif chart_type == "area":
                                    chart = alt.Chart(df).mark_area().encode(
                                        x=x_col,
                                        y=y_col,
                                        tooltip=[x_col, y_col]
                                    ).properties(width=600, height=400)
                                
                                st.altair_chart(chart, use_container_width=True)
                                
                                # Save chart configuration
                                slide["chart_config"] = {
                                    "type": chart_type,
                                    "x_col": x_col,
                                    "y_col": y_col,
                                    "data_source": "current_query"
                                }
                            
                            elif chart_type == "pie":
                                label_col = st.selectbox(
                                    "Labels",
                                    options=df.columns.tolist(),
                                    index=0,
                                    key=f"label_col_{slide_id}"
                                )
                                
                                value_col = st.selectbox(
                                    "Values",
                                    options=df.columns.tolist(),
                                    index=min(1, len(df.columns) - 1),
                                    key=f"value_col_{slide_id}"
                                )
                                
                                # Create pie chart using matplotlib
                                fig, ax = plt.subplots(figsize=(10, 6))
                                df.plot.pie(y=value_col, labels=df[label_col], ax=ax, autopct='%1.1f%%')
                                st.pyplot(fig)
                                
                                # Save chart configuration
                                slide["chart_config"] = {
                                    "type": chart_type,
                                    "label_col": label_col,
                                    "value_col": value_col,
                                    "data_source": "current_query"
                                }
                            
                            elif chart_type == "heatmap":
                                x_col = st.selectbox(
                                    "X-axis",
                                    options=df.columns.tolist(),
                                    index=0,
                                    key=f"heatmap_x_col_{slide_id}"
                                )
                                
                                y_col = st.selectbox(
                                    "Y-axis",
                                    options=df.columns.tolist(),
                                    index=min(1, len(df.columns) - 1),
                                    key=f"heatmap_y_col_{slide_id}"
                                )
                                
                                value_col = st.selectbox(
                                    "Value",
                                    options=df.columns.tolist(),
                                    index=min(2, len(df.columns) - 1),
                                    key=f"heatmap_value_col_{slide_id}"
                                )
                                
                                # Create pivot table for heatmap
                                pivot_df = df.pivot_table(
                                    index=y_col,
                                    columns=x_col,
                                    values=value_col,
                                    aggfunc='mean'
                                )
                                
                                # Create heatmap using seaborn
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", ax=ax)
                                st.pyplot(fig)
                                
                                # Save chart configuration
                                slide["chart_config"] = {
                                    "type": chart_type,
                                    "x_col": x_col,
                                    "y_col": y_col,
                                    "value_col": value_col,
                                    "data_source": "current_query"
                                }
                        else:
                            st.info("Run a query first to use its results for visualization.")
                    
                    # Annotations
                    st.subheader("Annotations")
                    
                    # Add annotation
                    new_annotation = st.text_area(
                        "Add Annotation",
                        key=f"new_annotation_{slide_id}",
                        placeholder="Add insights or explanations about this slide..."
                    )
                    
                    if st.button("Add Annotation", key=f"add_annotation_btn_{slide_id}") and new_annotation:
                        if "annotations" not in slide:
                            slide["annotations"] = []
                        
                        slide["annotations"].append({
                            "id": str(uuid.uuid4()),
                            "text": new_annotation,
                            "created_by": st.session_state.current_user,
                            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                        story["modified_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success("Annotation added.")
                        st.rerun()
                    
                    # Display existing annotations
                    if slide.get("annotations"):
                        for i, annotation in enumerate(slide["annotations"]):
                            st.markdown(
                                f"""
                                <div style="border-left: 3px solid #ccc; padding-left: 10px; margin-bottom: 10px;">
                                    <p><strong>{annotation.get('created_by', 'Unknown')}</strong> • {annotation.get('created_at', '')}</p>
                                    <p>{annotation.get('text', '')}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Delete annotation button
                            if st.button("Delete", key=f"delete_annotation_{slide_id}_{i}"):
                                slide["annotations"].pop(i)
                                story["modified_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.success("Annotation deleted.")
                                st.rerun()
                    
                    # Update slide button
                    if st.button("Update Slide", key=f"update_slide_{slide_id}"):
                        slide["title"] = slide_title
                        slide["content"] = slide_content
                        slide["layout"] = slide_layout
                        if slide_layout in ["text_image", "image_text", "full_image"]:
                            slide["image_url"] = image_url
                        
                        story["modified_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success("Slide updated.")
                    
                    # Slide order controls
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if slide.get("order", 0) > 0 and st.button("Move Up", key=f"move_up_{slide_id}"):
                            # Find the slide with the previous order
                            for other_id, other_slide in story["slides"].items():
                                if other_slide.get("order", 0) == slide.get("order", 0) - 1:
                                    # Swap orders
                                    other_slide["order"] = slide.get("order", 0)
                                    slide["order"] = slide.get("order", 0) - 1
                                    story["modified_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success("Slide moved up.")
                                    st.rerun()
                                    break
                    
                    with col2:
                        if slide.get("order", 0) < len(story["slides"]) - 1 and st.button("Move Down", key=f"move_down_{slide_id}"):
                            # Find the slide with the next order
                            for other_id, other_slide in story["slides"].items():
                                if other_slide.get("order", 0) == slide.get("order", 0) + 1:
                                    # Swap orders
                                    other_slide["order"] = slide.get("order", 0)
                                    slide["order"] = slide.get("order", 0) + 1
                                    story["modified_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success("Slide moved down.")
                                    st.rerun()
                                    break
                    
                    # Delete slide button
                    if st.button("Delete Slide", key=f"delete_slide_{slide_id}"):
                        # Confirm deletion
                        if st.session_state.get("confirm_delete_slide") == slide_id:
                            # Delete the slide
                            del story["slides"][slide_id]
                            
                            # Reorder remaining slides
                            for i, (other_id, other_slide) in enumerate(sorted(
                                story["slides"].items(),
                                key=lambda x: x[1].get("order", 0)
                            )):
                                other_slide["order"] = i
                            
                            story["modified_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.success("Slide deleted.")
                            st.rerun()
                        else:
                            st.session_state.confirm_delete_slide = slide_id
                            st.warning("Are you sure you want to delete this slide? Click Delete Slide again to confirm.")
            
            # Exit edit mode button
            if st.button("Exit Edit Mode"):
                st.session_state.story_edit_mode = False
                st.session_state.current_story = None
                st.rerun()
        
        else:
            # Create new story
            st.subheader("Create New Data Story")
            
            with st.form("create_story_form"):
                story_title = st.text_input("Story Title", key="new_story_title")
                story_description = st.text_area("Description", key="new_story_description")
                story_tags = st.text_input("Tags (comma separated)", key="new_story_tags")
                
                submitted = st.form_submit_button("Create Story")
                if submitted:
                    if not story_title:
                        st.error("Please enter a story title.")
                    else:
                        # Create new story
                        story_id = str(uuid.uuid4())
                        
                        # Parse tags
                        tags = [tag.strip() for tag in story_tags.split(",") if tag.strip()]
                        
                        # Create story
                        st.session_state.data_stories[story_id] = {
                            "title": story_title,
                            "description": story_description,
                            "tags": tags,
                            "created_by": st.session_state.current_user,
                            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "modified_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "slides": {
                                str(uuid.uuid4()): {
                                    "title": "Introduction",
                                    "content": "Welcome to your new data story. Edit this slide to get started.",
                                    "layout": "text",
                                    "order": 0,
                                    "annotations": [],
                                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                            },
                            "thumbnail": None
                        }
                        
                        st.success(f"Story '{story_title}' created successfully.")
                        
                        # Enter edit mode
                        st.session_state.current_story = story_id
                        st.session_state.story_edit_mode = True
                        st.rerun()
    
    def _story_library_ui(self):
        """UI for browsing shared data stories"""
        st.subheader("Story Library")
        
        # Get all stories shared in workspaces
        shared_stories = {}
        
        # Check if collaboration module is available
        if "collaboration" in st.session_state:
            # Get current workspace
            workspace = st.session_state.workspaces.get(st.session_state.current_workspace, {})
            
            # Get stories from workspace
            for story_id, story in workspace.get("stories", {}).items():
                shared_stories[story_id] = story
        
        # Also include public stories from all users
        for story_id, story in st.session_state.data_stories.items():
            if story.get("public", False) and story_id not in shared_stories:
                shared_stories[story_id] = story
        
        if not shared_stories:
            st.info("No shared stories available in the library.")
        else:
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                filter_tag = st.selectbox(
                    "Filter by Tag",
                    options=["All"] + list(set(
                        tag for story in shared_stories.values()
                        for tag in story.get("tags", [])
                    )),
                    key="library_filter_tag"
                )
            
            with col2:
                filter_author = st.selectbox(
                    "Filter by Author",
                    options=["All"] + list(set(
                        story.get("created_by", "Unknown")
                        for story in shared_stories.values()
                    )),
                    key="library_filter_author"
                )
            
            # Apply filters
            filtered_stories = {}
            for story_id, story in shared_stories.items():
                if (filter_tag == "All" or filter_tag in story.get("tags", [])) and \
                   (filter_author == "All" or filter_author == story.get("created_by", "Unknown")):
                    filtered_stories[story_id] = story
            
            # Display filtered stories
            if not filtered_stories:
                st.info("No stories match the selected filters.")
            else:
                # Create a grid layout
                cols = st.columns(3)
                
                for i, (story_id, story) in enumerate(filtered_stories.items()):
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                                <h3>{story.get('title', 'Untitled')}</h3>
                                <p><em>By {story.get('created_by', 'Unknown')}</em></p>
                                <p>{story.get('description', 'No description')[:100]}...</p>
                                <p><strong>Slides:</strong> {len(story.get('slides', {}))}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Display story thumbnail if available
                        if story.get("thumbnail"):
                            st.image(story["thumbnail"], width=200)
                        
                        # View button
                        if st.button("View Story", key=f"view_library_story_{story_id}"):
                            st.session_state.current_story = story_id
                            st.session_state.presentation_mode = True
                            st.session_state.current_slide = 0
                            st.rerun()
    
    def _storytelling_settings_ui(self):
        """UI for data storytelling settings"""
        st.subheader("Storytelling Settings")
        
        # Presentation settings
        st.write("**Presentation Settings**")
        
        # Theme selection
        theme = st.selectbox(
            "Presentation Theme",
            options=["Default", "Dark", "Light", "Corporate", "Minimal"],
            key="presentation_theme"
        )
        
        # Transition effects
        transition = st.selectbox(
            "Slide Transition Effect",
            options=["None", "Fade", "Slide", "Zoom"],
            key="slide_transition"
        )
        
        # Auto-advance
        auto_advance = st.checkbox(
            "Enable auto-advance",
            value=False,
            key="auto_advance"
        )
        
        if auto_advance:
            advance_time = st.slider(
                "Seconds per slide",
                min_value=5,
                max_value=60,
                value=20,
                step=5,
                key="advance_time"
            )
        
        # Save settings button
        if st.button("Save Settings"):
            st.session_state.storytelling_settings = {
                "theme": theme,
                "transition": transition,
                "auto_advance": auto_advance,
                "advance_time": advance_time if auto_advance else 20
            }
            st.success("Settings saved.")
        
        # AI Assistance settings
        st.write("**AI Assistance**")
        
        enable_ai = st.checkbox(
            "Enable AI-generated insights",
            value=True,
            key="enable_ai_insights"
        )
        
        if enable_ai:
            insight_depth = st.select_slider(
                "Insight Depth",
                options=["Basic", "Moderate", "Detailed"],
                value="Moderate",
                key="insight_depth"
            )
            
            auto_annotations = st.checkbox(
                "Auto-generate annotations for charts",
                value=True,
                key="auto_annotations"
            )
        
        # Save AI settings button
        if st.button("Save AI Settings"):
            st.session_state.ai_storytelling_settings = {
                "enable_ai": enable_ai,
                "insight_depth": insight_depth if enable_ai else "Moderate",
                "auto_annotations": auto_annotations if enable_ai else False
            }
            st.success("AI settings saved.")
    
    def _presentation_mode_ui(self):
        """UI for presentation mode"""
        if not st.session_state.current_story:
            st.error("No story selected for presentation.")
            
            if st.button("Exit Presentation Mode"):
                st.session_state.presentation_mode = False
                st.rerun()
            return
        
        # Get the current story
        story = st.session_state.data_stories[st.session_state.current_story]
        
        # Get slides sorted by order
        sorted_slides = sorted(
            story["slides"].items(),
            key=lambda x: x[1].get("order", 0)
        )
        
        if not sorted_slides:
            st.error("This story has no slides.")
            
            if st.button("Exit Presentation Mode"):
                st.session_state.presentation_mode = False
                st.rerun()
            return
        
        # Get current slide
        current_slide_index = st.session_state.current_slide
        if current_slide_index >= len(sorted_slides):
            current_slide_index = 0
            st.session_state.current_slide = 0
        
        slide_id, slide = sorted_slides[current_slide_index]
        
        # Create a full-screen container
        st.markdown(
            """
            <style>
            .presentation-container {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: white;
                z-index: 999;
                padding: 20px;
                overflow-y: auto;
            }
            .slide-title {
                font-size: 2.5em;
                margin-bottom: 20px;
            }
            .slide-content {
                font-size: 1.5em;
                margin-bottom: 30px;
            }
            .navigation-controls {
                position: fixed;
                bottom: 20px;
                right: 20px;
                display: flex;
                gap: 10px;
            }
            .exit-button {
                position: fixed;
                top: 20px;
                right: 20px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Display slide content based on layout
        st.markdown(f"<h1 class='slide-title'>{slide.get('title', 'Untitled Slide')}</h1>", unsafe_allow_html=True)
        
        layout = slide.get("layout", "text")
        
        if layout == "text":
            st.markdown(f"<div class='slide-content'>{slide.get('content', '')}</div>", unsafe_allow_html=True)
        
        elif layout == "text_image":
            col1, col2 = st.columns([6, 4])
            with col1:
                st.markdown(f"<div class='slide-content'>{slide.get('content', '')}</div>", unsafe_allow_html=True)
            with col2:
                if slide.get("image_url"):
                    st.image(slide["image_url"])
        
        elif layout == "image_text":
            col1, col2 = st.columns([4, 6])
            with col1:
                if slide.get("image_url"):
                    st.image(slide["image_url"])
            with col2:
                st.markdown(f"<div class='slide-content'>{slide.get('content', '')}</div>", unsafe_allow_html=True)
        
        elif layout == "full_image":
            if slide.get("image_url"):
                st.image(slide["image_url"], use_column_width=True)
            st.markdown(f"<div class='slide-content'>{slide.get('content', '')}</div>", unsafe_allow_html=True)
        
        elif layout in ["text_chart", "chart_text", "full_chart", "two_charts"]:
            # Display chart based on configuration
            if slide.get("chart_config"):
                chart_config = slide["chart_config"]
                
                if layout == "text_chart":
                    col1, col2 = st.columns([6, 4])
                    with col1:
                        st.markdown(f"<div class='slide-content'>{slide.get('content', '')}</div>", unsafe_allow_html=True)
                    with col2:
                        self._render_chart(chart_config)
                
                elif layout == "chart_text":
                    col1, col2 = st.columns([4, 6])
                    with col1:
                        self._render_chart(chart_config)
                    with col2:
                        st.markdown(f"<div class='slide-content'>{slide.get('content', '')}</div>", unsafe_allow_html=True)
                
                elif layout == "full_chart":
                    self._render_chart(chart_config)
                    st.markdown(f"<div class='slide-content'>{slide.get('content', '')}</div>", unsafe_allow_html=True)
                
                elif layout == "two_charts":
                    col1, col2 = st.columns(2)
                    with col1:
                        if slide.get("chart_config"):
                            self._render_chart(chart_config)
                    with col2:
                        if slide.get("chart_config2"):
                            self._render_chart(slide["chart_config2"])
                    st.markdown(f"<div class='slide-content'>{slide.get('content', '')}</div>", unsafe_allow_html=True)
            else:
                st.warning("Chart configuration not found.")
                st.markdown(f"<div class='slide-content'>{slide.get('content', '')}</div>", unsafe_allow_html=True)
        
        # Display annotations if available
        if slide.get("annotations"):
            with st.expander("Annotations & Insights", expanded=False):
                for annotation in slide["annotations"]:
                    st.markdown(
                        f"""
                        <div style="border-left: 3px solid #ccc; padding-left: 10px; margin-bottom: 10px;">
                            <p><strong>{annotation.get('created_by', 'Unknown')}</strong> • {annotation.get('created_at', '')}</p>
                            <p>{annotation.get('text', '')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # Navigation controls
        col1, col2, col3 = st.columns([1, 10, 1])
        
        with col1:
            if current_slide_index > 0:
                if st.button("◀ Previous"):
                    st.session_state.current_slide = current_slide_index - 1
                    st.rerun()
        
        with col3:
            if current_slide_index < len(sorted_slides) - 1:
                if st.button("Next ▶"):
                    st.session_state.current_slide = current_slide_index + 1
                    st.rerun()
        
        # Exit presentation mode button
        if st.button("Exit Presentation"):
            st.session_state.presentation_mode = False
            st.rerun()
        
        # Slide counter
        st.markdown(
            f"""
            <div style="position: fixed; bottom: 20px; left: 20px; font-size: 0.8em; color: #666;">
                Slide {current_slide_index + 1} of {len(sorted_slides)}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _render_chart(self, chart_config):
        """Render a chart based on configuration"""
        if not chart_config:
            st.warning("No chart configuration available.")
            return
        
        # Get data source
        if chart_config.get("data_source") == "current_query" and \
           "query_results" in st.session_state and \
           isinstance(st.session_state.query_results, pd.DataFrame):
            df = st.session_state.query_results
        else:
            st.warning("Chart data not available.")
            return
        
        # Render chart based on type
        chart_type = chart_config.get("type", "bar")
        
        if chart_type == "bar":
            x_col = chart_config.get("x_col")
            y_col = chart_config.get("y_col")
            
            if x_col and y_col:
                chart = alt.Chart(df).mark_bar().encode(
                    x=x_col,
                    y=y_col,
                    tooltip=[x_col, y_col]
                ).properties(width=600, height=400)
                
                st.altair_chart(chart, use_container_width=True)
        
        elif chart_type == "line":
            x_col = chart_config.get("x_col")
            y_col = chart_config.get("y_col")
            
            if x_col and y_col:
                chart = alt.Chart(df).mark_line().encode(
                    x=x_col,
                    y=y_col,
                    tooltip=[x_col, y_col]
                ).properties(width=600, height=400)
                
                st.altair_chart(chart, use_container_width=True)
        
        elif chart_type == "scatter":
            x_col = chart_config.get("x_col")
            y_col = chart_config.get("y_col")
            
            if x_col and y_col:
                chart = alt.Chart(df).mark_circle().encode(
                    x=x_col,
                    y=y_col,
                    tooltip=[x_col, y_col]
                ).properties(width=600, height=400)
                
                st.altair_chart(chart, use_container_width=True)
        
        elif chart_type == "pie":
            label_col = chart_config.get("label_col")
            value_col = chart_config.get("value_col")
            
            if label_col and value_col:
                fig, ax = plt.subplots(figsize=(10, 6))
                df.plot.pie(y=value_col, labels=df[label_col], ax=ax, autopct='%1.1f%%')
                st.pyplot(fig)
        
        elif chart_type == "area":
            x_col = chart_config.get("x_col")
            y_col = chart_config.get("y_col")
            
            if x_col and y_col:
                chart = alt.Chart(df).mark_area().encode(
                    x=x_col,
                    y=y_col,
                    tooltip=[x_col, y_col]
                ).properties(width=600, height=400)
                
                st.altair_chart(chart, use_container_width=True)
        
        elif chart_type == "heatmap":
            x_col = chart_config.get("x_col")
            y_col = chart_config.get("y_col")
            value_col = chart_config.get("value_col")
            
            if x_col and y_col and value_col:
                # Create pivot table for heatmap
                pivot_df = df.pivot_table(
                    index=y_col,
                    columns=x_col,
                    values=value_col,
                    aggfunc='mean'
                )
                
                # Create heatmap using seaborn
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", ax=ax)
                st.pyplot(fig)
    
    def _get_default_stories(self) -> Dict[str, Any]:
        """Get default data stories
        
        Returns:
            Dict[str, Any]: Default data stories
        """
        return {
            "sample_story_1": {
                "title": "Introduction to Data Storytelling",
                "description": "A sample story demonstrating the features of the Data Storytelling module",
                "tags": ["sample", "tutorial", "introduction"],
                "created_by": "admin",
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "modified_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "public": True,
                "slides": {
                    "slide_1": {
                        "title": "Welcome to Data Storytelling",
                        "content": """
                        # Welcome to Data Storytelling
                        
                        Data storytelling is the practice of building a narrative around a set of data and its accompanying visualizations to help convey the meaning of that data in a powerful and compelling fashion.
                        
                        With our Data Storytelling module, you can:
                        - Create interactive presentations
                        - Add context to your visualizations
                        - Share insights with your team
                        - Build branching narratives
                        """,
                        "layout": "text",
                        "order": 0,
                        "annotations": [],
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "slide_2": {
                        "title": "Why Data Storytelling Matters",
                        "content": """
                        ## Why Data Storytelling Matters
                        
                        Data without context is just numbers. Storytelling helps you:
                        
                        - Make data more accessible to non-technical stakeholders
                        - Highlight key insights that might be missed
                        - Drive decision-making with compelling narratives
                        - Create memorable presentations that stick with your audience
                        """,
                        "layout": "text_image",
                        "image_url": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80",
                        "order": 1,
                        "annotations": [
                            {
                                "id": "annotation_1",
                                "text": "Research shows that stories are 22 times more memorable than facts alone.",
                                "created_by": "admin",
                                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        ],
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "slide_3": {
                        "title": "Getting Started",
                        "content": """
                        ## Getting Started
                        
                        To create your first data story:
                        
                        1. Go to the "Create Story" tab
                        2. Enter a title and description
                        3. Add slides with different layouts
                        4. Include visualizations from your queries
                        5. Add annotations and insights
                        6. Present your story to stakeholders
                        """,
                        "layout": "text",
                        "order": 2,
                        "annotations": [],
                        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                },
                "thumbnail": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80"
            }
        }