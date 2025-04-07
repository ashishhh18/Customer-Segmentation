import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io

class CustomerSegmentation:
    def __init__(self):
        
        st.set_page_config(
            page_title="Customer Segmentation",
            page_icon=":bar_chart:",
            layout="wide"
        )

    def load_and_preprocess_data(self, uploaded_file):
        
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Please ensure your CSV has columns: {required_columns}")
            return None, None
        
        # Select features for clustering
        features = required_columns
        X = df[features]
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, df

    def find_optimal_clusters(self, X_scaled, max_clusters=10):
        
        # Calculate inertia for different numbers of clusters
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot the Elbow Curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), inertias, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        
        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return inertias, buf

    def perform_clustering(self, X_scaled, k):
        
        kmeans = KMeans(n_clusters=k, random_state=42)
        return kmeans.fit_predict(X_scaled)

    def visualize_clusters(self, X_scaled, cluster_labels):
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            X_scaled[:, 0], 
            X_scaled[:, 1], 
            c=cluster_labels, 
            cmap='viridis'
        )
        plt.title('Customer Segments')
        plt.xlabel('Standardized Annual Income')
        plt.ylabel('Standardized Spending Score')
        plt.colorbar(scatter, label='Cluster')
        
        # Save plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return buf

    def analyze_clusters(self, df, cluster_labels):
        
        # Add cluster labels to the dataframe
        df_clustered = df.copy()
        df_clustered['Cluster'] = cluster_labels
        
        # Calculate cluster summaries
        cluster_summary = df_clustered.groupby('Cluster')[
            ['Annual Income (k$)', 'Spending Score (1-100)']
        ].agg(['mean', 'min', 'max', 'count'])
        
        return cluster_summary

    def run(self):
        
        st.title("🛍️ Customer Segmentation Analysis")
        
        # Sidebar for file upload and configuration
        st.sidebar.header("Configuration")
        
        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Upload Customer Data CSV", 
            type=['csv']
        )
        
        # Number of clusters selector
        num_clusters = st.sidebar.slider(
            "Select Number of Clusters", 
            min_value=2, 
            max_value=10, 
            value=3
        )
        
        if uploaded_file is not None:
            # Load and preprocess data
            X_scaled, df = self.load_and_preprocess_data(uploaded_file)
            
            if X_scaled is not None:
                # Find optimal clusters (Elbow Method)
                inertias, elbow_plot = self.find_optimal_clusters(X_scaled)
                
                # Display Elbow Method Plot
                st.subheader("Elbow Method for Optimal Clusters")
                st.image(elbow_plot, caption="Elbow Curve for Cluster Selection")
                
                # Perform clustering
                cluster_labels = self.perform_clustering(X_scaled, num_clusters)
                
                # Visualize clusters
                cluster_plot = self.visualize_clusters(X_scaled, cluster_labels)
                
                # Display cluster visualization
                st.subheader("Customer Segments Visualization")
                st.image(cluster_plot, caption=f"Customer Segmentation into {num_clusters} Clusters")
                
                # Analyze and display cluster summary
                cluster_summary = self.analyze_clusters(df, cluster_labels)
                
                st.subheader("Cluster Summary Statistics")
                st.dataframe(cluster_summary)
                
                # Allow download of results
                st.sidebar.header("Download Results")
                
                # Download cluster labels
                if st.sidebar.download_button(
                    label="Download Cluster Labels",
                    data=pd.DataFrame({
                        'Cluster': cluster_labels, 
                        **df
                    }).to_csv(index=False),
                    file_name='customer_clusters.csv',
                    mime='text/csv'
                ):
                    st.sidebar.success("Cluster labels downloaded successfully!")

# Main
def main():
    app = CustomerSegmentation()
    app.run()

if __name__ == "__main__":
    main()

