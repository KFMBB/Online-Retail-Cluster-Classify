# Online Retail Cluster and Classify

## Project Overview
This project provides a comprehensive solution for customer segmentation and classification using the Online Retail dataset. The primary objective is to preprocess, cluster, and classify customers based on their purchasing behavior, enabling businesses to adopt targeted marketing strategies and optimize customer relationship management.

The application integrates clustering techniques to identify customer segments and uses pre-trained classification models to predict and label new customers. A user-friendly Streamlit web application offers interactive analysis and visualization of customer segments.

## Features
- **Data Upload**: Accepts customer purchase data in the Online Retail dataset format.
- **Automated Preprocessing**:
  - Cleans and prepares the uploaded data.
  - Extracts critical features: Monetary Value, Frequency, and Recency.
- **Clustering-Based Segmentation**:
  - Customer segments are derived through clustering (KMeans and other clustering techniques).
  - Segments include:
    - Win-Back Campaign
    - Minimal Effort Group
    - Loyal VIPs
    - Upsell Potential
    - High Value Retention Targets
    - Re-Engagement Campaigns
    - Dormant Customers
- **Classification**:
  - Uses 10 pre-trained classification models for predicting customer segments.
- **Visualization**:
  - Interactive pie chart for segment distribution.
  - 3D scatter plot for Monetary Value, Frequency, and Recency, with segments as labels.
- **Download Results**: Outputs segmentation results in CSV format.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required libraries (listed in `requirements.txt`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/KFMBB/Online-Retail-Cluster-Classify.git
   cd Online-Retail-Cluster-Classify
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage Guide
1. Open the app in your browser: [Streamlit App](https://online-retail-cluster-classify-mok.streamlit.app/).
2. Upload your dataset in the Online Retail format.
3. View the preprocessed metrics and choose a classification model.
4. Visualize customer segments using interactive charts.
5. Download the segmentation results as a CSV file.

## Implementation Details

### Project Structure
- **app.py**: Main Streamlit app script.
- **models/**: Directory containing pre-trained classification models in `.pkl` format.
- **data/**: Example datasets for testing.
- **notebook/**: Jupyter notebook with detailed preprocessing, clustering, and classification pipeline.

### Workflow

#### Preprocessing and Feature Engineering
- **Notebook Reference**: The clustering and feature engineering steps are detailed in the Jupyter notebook `Online_Retail_Detailed_Clustering_Classification.ipynb`.
- **Steps**:
  1. Compute **Monetary Value**: Total spending per customer (`Quantity` × `UnitPrice`).
  2. Calculate **Frequency**: Number of unique invoices per customer.
  3. Derive **Recency**: Days since the last purchase, based on the most recent invoice date.
  4. Scale features using `StandardScaler` for clustering and classification.

#### Clustering-Based Segmentation
- Clustering algorithms, including **KMeans**, are applied to identify customer segments based on scaled features (Monetary Value, Frequency, and Recency).
- Each segment is labeled descriptively (e.g., Loyal VIPs, Dormant Customers).
- Cluster labels are encoded in the dataset for subsequent classification tasks.

#### Classification
- Pre-trained models include Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes, MLP, KNN, and Stacking.
- Models are stored as `.pkl` files and dynamically loaded based on user selection.
- Predictions assign new customers to the most relevant segments based on their features.

#### Streamlit UI Components
- **File Uploader**: Users upload their datasets for processing.
- **Dropdown Menu**: Allows selection of a pre-trained classification model.
- **Interactive Charts**:
  - Pie chart: Displays segment distribution.
  - 3D scatter plot: Visualizes Monetary Value, Frequency, and Recency with customer segments.
- **Download Button**: Enables users to download segmented results.

## Technical Details
- **Programming Language**: Python
- **Libraries**:
  - Data Processing: `pandas`, `numpy`, `scikit-learn`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Web Application: `Streamlit`
- **Models**:
  - Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes, MLP, KNN, and Stacking.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Submit a pull request.

## License
This project is licensed under [MIT License](LICENSE).

## Repository
GitHub: [Online Retail Cluster and Classify](https://github.com/KFMBB/Online-Retail-Cluster-Classify)

## Hosted Application
Streamlit App: [https://online-retail-cluster-classify-mok.streamlit.app/](https://online-retail-cluster-classify-mok.streamlit.app/)
 

