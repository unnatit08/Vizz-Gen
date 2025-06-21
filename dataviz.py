import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import re
import datetime
import os
import matplotlib.pyplot as plt

# Initialize session state
if 'selected_suggestion' not in st.session_state:
    st.session_state.selected_suggestion = ""
if 'plot_type' not in st.session_state:
    st.session_state.plot_type = "Line Plot"
if 'x_column' not in st.session_state:
    st.session_state.x_column = None
if 'y_column' not in st.session_state:
    st.session_state.y_column = None
if 'hist_column' not in st.session_state:
    st.session_state.hist_column = None
if 'label_column' not in st.session_state:
    st.session_state.label_column = None
if 'value_column' not in st.session_state:
    st.session_state.value_column = None
if 'saved_plots' not in st.session_state:
    st.session_state.saved_plots = []
if 'multiple_plots' not in st.session_state:
    st.session_state.multiple_plots = []
if 'use_plotly' not in st.session_state:
    st.session_state.use_plotly = True

# --- Streamlit App Title (Centered with Icon) ---
st.markdown("<h1 style='text-align: center;'>📊 Data Visualization Generator</h1>", unsafe_allow_html=True)

# --- File Uploader with Icon ---
uploaded_file = st.file_uploader("📂 Choose a CSV file", type=["csv", "xlsx"])

# --- Cached function to load data ---
@st.cache_data
def load_data(file):
    """Caches data loading to improve performance on reruns."""
    if file is not None:
        try:
            if file.name.endswith('.csv'):
                data = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                data = pd.read_excel(file)
            return data
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
    return None

# Load data using the cached function
data = load_data(uploaded_file)

# Main app logic when data is loaded
if data is not None:
    st.write("Data Preview:")
    st.write(data.head())

    # --- Customization Options ---
    st.sidebar.header("⚙️ Plot Customization")
    
    # Input for plot title
    plot_title = st.sidebar.text_input("Plot Title", "My Plot")

    column_options = data.columns.tolist()

    # --- Plot Library Selection ---
    plot_library = "Plotly" if st.session_state.use_plotly else "Matplotlib"

    # --- Advanced Customization Options ---
    with st.sidebar.expander("🎨 Advanced Settings"):
        template = st.selectbox(
            "Select Template", 
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
            disabled=not st.session_state.use_plotly
        )
        
        if st.session_state.use_plotly:
            color_sequence = st.selectbox(
                "Color Sequence", 
                ["Plotly", "D3", "Viridis", "Rainbow"]
            )
        
        st.markdown("**Common Settings**")
        opacity = st.slider("Opacity", 0.1, 1.0, 0.8)
        
        if not st.session_state.use_plotly:
            figsize_x = st.slider("Figure Width", 5, 15, 10)
            figsize_y = st.slider("Figure Height", 3, 10, 6)

    # --- Structured Visualization Suggestions ---
    st.sidebar.subheader("💡 Suggested Visualizations")
    
    # Get column types
    numeric_cols_suggest = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols_suggest = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # First dropdown - select visualization type
    visualization_type = st.sidebar.selectbox(
        "Select Visualization Type",
        ["", "Line Plot", "Scatter Plot", "Bar Chart", "Pie Chart", "Doughnut Chart", "Histogram"],
        key='visualization_type_select'
    )

    # Generate suggestions based on selected visualization type
    if visualization_type:
        suggestions = []
        
        if visualization_type in ["Line Plot", "Scatter Plot"]:
            if len(numeric_cols_suggest) >= 2:
                # Generate all numeric column pairs
                pairs = [(numeric_cols_suggest[i], numeric_cols_suggest[j]) 
                        for i in range(len(numeric_cols_suggest)) 
                        for j in range(i+1, len(numeric_cols_suggest))]
                suggestions = [f"'{x}' vs '{y}'" for x, y in pairs]
                
        elif visualization_type == "Bar Chart":
            if len(categorical_cols_suggest) >= 1 and len(numeric_cols_suggest) >= 1:
                # Generate all category vs numeric combinations
                suggestions = [f"'{cat}' vs '{num}'" 
                             for cat in categorical_cols_suggest 
                             for num in numeric_cols_suggest]
                
        elif visualization_type in ["Pie Chart", "Doughnut Chart"]:
            if len(categorical_cols_suggest) >= 1 and len(numeric_cols_suggest) >= 1:
                # Generate all label vs value combinations
                suggestions = [f"'{cat}' values from '{num}'" 
                             for cat in categorical_cols_suggest 
                             for num in numeric_cols_suggest]
                
        elif visualization_type == "Histogram":
            if len(numeric_cols_suggest) >= 1:
                suggestions = [f"Distribution of '{num}'" 
                             for num in numeric_cols_suggest]

        # Display column combination dropdown if there are suggestions
        if suggestions:
            selected_option = st.sidebar.selectbox(
                f"Select {visualization_type} Option", 
                [""] + suggestions,
                key='column_combination_select'
            )
            
            if selected_option:
                # Parse the selected option to get columns
                if visualization_type in ["Line Plot", "Scatter Plot", "Bar Chart"]:
                    cols = re.findall(r"'(.*?)'", selected_option)
                    if len(cols) >= 2:
                        st.session_state.update({
                            "plot_type": visualization_type,
                            "x_column": cols[0],
                            "y_column": cols[1] if visualization_type != "Histogram" else None,
                            "selected_suggestion": f"{visualization_type}: {selected_option}"
                        })
                        
                elif visualization_type in ["Pie Chart", "Doughnut Chart"]:
                    cols = re.findall(r"'(.*?)'", selected_option)
                    if len(cols) >= 2:
                        st.session_state.update({
                            "plot_type": visualization_type,
                            "label_column": cols[0],
                            "value_column": cols[1],
                            "selected_suggestion": f"{visualization_type}: {selected_option}"
                        })
                        
                elif visualization_type == "Histogram":
                    col = re.findall(r"'(.*?)'", selected_option)[0]
                    st.session_state.update({
                        "plot_type": visualization_type,
                        "hist_column": col,
                        "selected_suggestion": f"{visualization_type}: {selected_option}"
                    })

    # --- Manual Plot Configuration ---
    plot_type = st.sidebar.selectbox(
        "Select Plot Type",
        ["Line Plot", "Scatter Plot", "Bar Chart", "Histogram", "Pie Chart", "Doughnut Chart"],
        index=["Line Plot", "Scatter Plot", "Bar Chart", "Histogram", "Pie Chart", "Doughnut Chart"].index(st.session_state.plot_type)
    )

    # Update plot type in session state
    if plot_type != st.session_state.plot_type:
        st.session_state.update({
            "plot_type": plot_type,
            "selected_suggestion": ""
        })

    # Column selection based on plot type
    if plot_type in ['Line Plot', 'Scatter Plot']:
        st.sidebar.subheader("Select Columns")
        if len(numeric_cols_suggest) >= 2:
            x_col = st.sidebar.selectbox(
                "X-axis Column", 
                numeric_cols_suggest,
                index=numeric_cols_suggest.index(st.session_state.x_column) if st.session_state.x_column in numeric_cols_suggest else 0
            )
            y_options = [col for col in numeric_cols_suggest if col != x_col]
            y_col = st.sidebar.selectbox(
                "Y-axis Column", 
                y_options,
                index=y_options.index(st.session_state.y_column) if st.session_state.y_column in y_options else 0
            )
            st.session_state.update({"x_column": x_col, "y_column": y_col})
            
    elif plot_type == 'Bar Chart':
        st.sidebar.subheader("Select Columns")
        if len(categorical_cols_suggest) > 0 and len(numeric_cols_suggest) > 0:
            x_col = st.sidebar.selectbox(
                "Category Column", 
                categorical_cols_suggest,
                index=categorical_cols_suggest.index(st.session_state.x_column) if st.session_state.x_column in categorical_cols_suggest else 0
            )
            y_col = st.sidebar.selectbox(
                "Value Column", 
                numeric_cols_suggest,
                index=numeric_cols_suggest.index(st.session_state.y_column) if st.session_state.y_column in numeric_cols_suggest else 0
            )
            st.session_state.update({"x_column": x_col, "y_column": y_col})
            
    elif plot_type in ['Pie Chart', 'Doughnut Chart']:
        st.sidebar.subheader("Select Columns")
        if len(categorical_cols_suggest) > 0 and len(numeric_cols_suggest) > 0:
            label_col = st.sidebar.selectbox(
                "Label Column", 
                categorical_cols_suggest,
                index=categorical_cols_suggest.index(st.session_state.label_column) if st.session_state.label_column in categorical_cols_suggest else 0
            )
            value_col = st.sidebar.selectbox(
                "Value Column", 
                numeric_cols_suggest,
                index=numeric_cols_suggest.index(st.session_state.value_column) if st.session_state.value_column in numeric_cols_suggest else 0
            )
            st.session_state.update({"label_column": label_col, "value_column": value_col})
            
    elif plot_type == 'Histogram':
        st.sidebar.subheader("Select Column")
        if len(numeric_cols_suggest) > 0:
            hist_col = st.sidebar.selectbox(
                "Column", 
                numeric_cols_suggest,
                index=numeric_cols_suggest.index(st.session_state.hist_column) if st.session_state.hist_column in numeric_cols_suggest else 0
            )
            st.session_state.update({"hist_column": hist_col})

    # --- Data Filtering Section ---
    with st.sidebar.expander("🔍 Data Filters"):
        filtered_data = data.copy()

        # Determine columns involved in the current plot
        filter_columns = []
        if plot_type in ['Line Plot', 'Scatter Plot', 'Bar Chart']:
            filter_columns = [st.session_state.x_column, st.session_state.y_column]
        elif plot_type == 'Histogram':
            filter_columns = [st.session_state.hist_column]
        elif plot_type in ['Pie Chart', 'Doughnut Chart']:
            filter_columns = [st.session_state.label_column, st.session_state.value_column]

        filter_columns = [col for col in filter_columns if col is not None]

        # Apply filters only to relevant columns
        for col in filter_columns:
            if col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    min_val, max_val = float(data[col].min()), float(data[col].max())
                    if min_val != max_val:
                        selected_range = st.slider(
                            f"Filter {col}", min_val, max_val, (min_val, max_val)
                        )
                        filtered_data = filtered_data[
                            (filtered_data[col] >= selected_range[0]) &
                            (filtered_data[col] <= selected_range[1])
                        ]
                elif pd.api.types.is_string_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                    options = data[col].dropna().unique().tolist()
                    selected = st.multiselect(
                        f"Filter {col}", options, default=options
                    )
                    filtered_data = filtered_data[filtered_data[col].isin(selected)]

        # Update data with filtered version
        data = filtered_data

    # --- Generate Plot Button ---
    if st.button("Generate Plot"):
        if plot_type in ['Line Plot', 'Scatter Plot', 'Bar Chart', 'Histogram', 'Pie Chart', 'Doughnut Chart']:
            if st.session_state.use_plotly:
                fig = None
                if plot_type in ['Line Plot', 'Scatter Plot'] and st.session_state.x_column and st.session_state.y_column:
                    if plot_type == 'Line Plot':
                        fig = px.line(data, x=st.session_state.x_column, y=st.session_state.y_column, 
                                    title=plot_title, template=template)
                    else:  # Scatter Plot
                        fig = px.scatter(data, x=st.session_state.x_column, y=st.session_state.y_column, 
                                       title=plot_title, template=template,
                                       trendline="ols" if st.sidebar.checkbox("Show Trendline") else None)
                    
                elif plot_type == 'Bar Chart' and st.session_state.x_column and st.session_state.y_column:
                    fig = px.bar(data, x=st.session_state.x_column, y=st.session_state.y_column,
                                title=plot_title, template=template)
                    
                elif plot_type == 'Histogram' and st.session_state.hist_column:
                    fig = px.histogram(data, x=st.session_state.hist_column, 
                                     title=plot_title, template=template,
                                     nbins=st.sidebar.slider("Number of Bins", 5, 50, 20))
                    
                elif plot_type in ['Pie Chart', 'Doughnut Chart'] and st.session_state.label_column and st.session_state.value_column:
                    fig = px.pie(data, names=st.session_state.label_column, values=st.session_state.value_column,
                                title=plot_title, template=template,
                                hole=0.4 if plot_type == 'Doughnut Chart' else 0)
                    
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Matplotlib version
                plt.figure(figsize=(figsize_x, figsize_y))
                if plot_type in ['Line Plot', 'Scatter Plot'] and st.session_state.x_column and st.session_state.y_column:
                    if plot_type == 'Line Plot':
                        plt.plot(data[st.session_state.x_column], data[st.session_state.y_column])
                    else:  # Scatter Plot
                        plt.scatter(data[st.session_state.x_column], data[st.session_state.y_column])
                    plt.xlabel(st.session_state.x_column)
                    plt.ylabel(st.session_state.y_column)
                    
                elif plot_type == 'Bar Chart' and st.session_state.x_column and st.session_state.y_column:
                    plt.bar(data[st.session_state.x_column], data[st.session_state.y_column])
                    plt.xticks(rotation=45)
                    
                elif plot_type == 'Histogram' and st.session_state.hist_column:
                    plt.hist(data[st.session_state.hist_column], bins=20)
                    plt.xlabel(st.session_state.hist_column)
                    plt.ylabel("Frequency")
                    
                elif plot_type in ['Pie Chart', 'Doughnut Chart'] and st.session_state.label_column and st.session_state.value_column:
                    plt.pie(data[st.session_state.value_column], 
                           labels=data[st.session_state.label_column],
                           autopct='%1.1f%%')
                    plt.axis('equal')
                    
                plt.title(plot_title)
                plt.grid(True)
                st.pyplot(plt)

    # --- Export Section ---
    if st.checkbox("Show Export Options"):
        st.subheader("Export Visualization")
        export_format = st.selectbox("Select Format", ["PNG", "HTML", "CSV"])
        
        if st.button("Export"):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"export_{plot_title}_{timestamp}.{export_format.lower()}"
            
            if export_format == "PNG":
                if st.session_state.use_plotly and fig:
                    fig.write_image(filename)
                else:
                    plt.savefig(filename)
                st.success(f"Saved as {filename}")
                
            elif export_format == "HTML":
                if st.session_state.use_plotly and fig:
                    fig.write_html(filename)
                    st.success(f"Saved as {filename}")
                else:
                    st.warning("HTML export only available for Plotly visualizations")
                    
            elif export_format == "CSV":
                data.to_csv(filename, index=False)
                st.success(f"Data saved as {filename}")

# --- Features Section ---
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Features"):
    st.markdown("""
    **Key Features:**
    - Interactive Plotly visualizations
    - Data filtering options
    - Multiple export formats
    - Smart plot suggestions
    - Support for CSV/Excel files
    """)

# --- Run Instructions ---
st.sidebar.markdown("")
