import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

# Set page config
st.set_page_config(page_title="DBMEMS Analysis", layout="wide")

# Title
st.title("DBMEMS Analysis of Marine Data")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_data(uploaded_file):
        df = pd.read_csv(uploaded_file)
        # Convert timestamp columns to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    df = load_data(uploaded_file)

    def compute_dbmems(time_points, n_eigenvectors=10):
        """Compute DBMEMS using time-based distance matrix"""
        # Convert timestamps to seconds since start
        time_seconds = (time_points - time_points.min()).dt.total_seconds()
        
        # Compute distance matrix
        dist_matrix = squareform(pdist(time_seconds.values.reshape(-1, 1)))
        
        # Compute truncated distance matrix (threshold = max distance / 4)
        threshold = np.max(dist_matrix) / 4
        dist_matrix[dist_matrix > threshold] = 4 * threshold
        
        # Compute centered distance matrix
        n = len(time_points)
        centering_matrix = np.eye(n) - np.ones((n, n)) / n
        centered_dist_matrix = -0.5 * centering_matrix @ (dist_matrix ** 2) @ centering_matrix
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = eigh(centered_dist_matrix)
        
        # Sort by absolute value of eigenvalues
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Return first n_eigenvectors
        return eigenvectors[:, :n_eigenvectors], eigenvalues[:n_eigenvectors]

    # Sidebar for analysis options
    st.sidebar.header("Analysis Options")
    selected_variables = st.sidebar.multiselect(
        "Select Variables for Analysis",
        options=df.columns[1:],  # Exclude 'File' column
        default=['Temperature', 'Salinity', 'Oxygen (ml/l)']
    )

    n_eigenvectors = st.sidebar.slider(
        "Number of DBMEMS Eigenvectors",
        min_value=1,
        max_value=20,
        value=10
    )

    # DBMEMS Analysis
    st.header("DBMEMS Analysis")
    if len(selected_variables) > 0:
        # Compute DBMEMS
        dbmems, eigenvalues = compute_dbmems(df['Timestamp'], n_eigenvectors)
        
        # Plot eigenvalues
        fig = px.line(
            x=range(1, len(eigenvalues) + 1),
            y=eigenvalues,
            title="DBMEMS Eigenvalues",
            labels={'x': 'Eigenvector Number', 'y': 'Eigenvalue'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot first few eigenvectors
        st.subheader("DBMEMS Eigenvectors")
        for i in range(min(5, n_eigenvectors)):
            fig = px.line(
                x=df['Timestamp'],
                y=dbmems[:, i],
                title=f'DBMEMS Eigenvector {i+1}',
                labels={'x': 'Time', 'y': f'Eigenvector {i+1}'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation between DBMEMS and selected variables
        st.subheader("Correlation with DBMEMS")
        for var in selected_variables:
            if pd.api.types.is_numeric_dtype(df[var]):
                correlations = np.corrcoef(dbmems.T, df[var].fillna(0))[:-1, -1]
                fig = px.bar(
                    x=range(1, len(correlations) + 1),
                    y=correlations,
                    title=f'Correlation between DBMEMS and {var}',
                    labels={'x': 'Eigenvector Number', 'y': 'Correlation'}
                )
                st.plotly_chart(fig, use_container_width=True)

    # Time series decomposition
    st.header("Time Series Decomposition")
    if len(selected_variables) > 0:
        for var in selected_variables:
            if pd.api.types.is_numeric_dtype(df[var]):
                decomposition = seasonal_decompose(df[var], period=24)  # Assuming daily periodicity
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Timestamp'], y=decomposition.trend, name='Trend'))
                fig.add_trace(go.Scatter(x=df['Timestamp'], y=decomposition.seasonal, name='Seasonal'))
                fig.add_trace(go.Scatter(x=df['Timestamp'], y=decomposition.resid, name='Residual'))
                
                fig.update_layout(
                    title=f'Decomposition of {var}',
                    xaxis_title='Time',
                    yaxis_title='Value'
                )
                st.plotly_chart(fig, use_container_width=True)

    # Correlation Analysis
    st.header("Correlation Analysis")
    if len(selected_variables) > 1:
        corr_matrix = df[selected_variables].corr()
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Time Series Plots
    st.header("Time Series Analysis")
    for var in selected_variables:
        if pd.api.types.is_numeric_dtype(df[var]):
            fig = px.line(
                df,
                x='Timestamp',
                y=var,
                title=f'Time Series of {var}'
            )
            st.plotly_chart(fig, use_container_width=True)

    # Distribution Analysis
    st.header("Distribution Analysis")
    for var in selected_variables:
        if pd.api.types.is_numeric_dtype(df[var]):
            fig = px.histogram(
                df,
                x=var,
                title=f'Distribution of {var}',
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)

    # Box Plots
    st.header("Box Plots")
    if len(selected_variables) > 0:
        fig = px.box(
            df,
            y=selected_variables,
            title="Box Plots of Selected Variables"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Data Summary
    st.header("Data Summary")
    st.write(df[selected_variables].describe())
else:
    st.info("Please upload a CSV file to begin analysis.") 