
# Online Retail Cluster and Classify

## Project Overview
This project provides a comprehensive solution for customer segmentation and classification using the Online Retail dataset. The goal is to preprocess, cluster, and classify customers based on their purchasing behavior, allowing businesses to make data-driven marketing decisions.

The solution includes a user-friendly Streamlit web application for interactive customer analysis.

## Features
- **Data Upload**: Upload data in the original Online Retail dataset format.
- **Automated Preprocessing**: Automatically cleans and preprocesses the uploaded data, extracting key features:
  - **Monetary Value**: Total spending by each customer.
  - **Frequency**: Number of purchases made by each customer.
  - **Recency**: Days since the last purchase.
- **Model Selection**: Choose from 10 pre-trained classification models for customer segmentation.
- **Visualization**:
  - Interactive pie chart for segment distribution.
  - 3D scatter plot of customer segments based on Monetary Value, Frequency, and Recency.
- **Download Results**: Export segmentation results as a CSV file.
- **Segment Labels**:
  - Win-Back Campaign
  - Minimal Effort Group
  - Loyal VIPs
  - Upsell Potential
  - High Value Retention Targets
  - Re-Engagement Campaigns
  - Dormant Customers

## Setup Instructions
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
1. Open the app in your browser: [Streamlit App](https://online-retail-cluster-classify-mok.streamlit.app/)
2. Upload your dataset in the Online Retail format.
3. View the preprocessed metrics and choose a classification model.
4. Visualize customer segments with interactive charts.
5. Download the segmentation results as a CSV file.

## Implementation Details
### Project Structure
- **app.py**: Main Streamlit app script.
- **models/**: Directory containing pre-trained classification models in `.pkl` format.
- **data/**: Example datasets for testing.

### Workflow
1. **Data Preprocessing**:
   - Reads the uploaded dataset and cleans it.
   - Computes Monetary Value, Frequency, and Recency for each customer.
2. **Feature Engineering**:
   - Scales features using `StandardScaler`.
   - Extracts Recency from the last invoice date.
3. **Clustering and Classification**:
   - Allows selection of pre-trained models (e.g., Random Forest, Gradient Boosting).
   - Predicts customer clusters and assigns descriptive labels.
4. **Visualization**:
   - Displays customer segment distribution with pie charts.
   - Shows a 3D scatter plot for segment analysis.

### Streamlit UI Components
- **File Uploader**: Upload CSV files for processing.
- **Dropdown Menu**: Select a classification model.
- **Interactive Charts**: Visualize segment distributions and relationships.
- **Download Button**: Export results.

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
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m "Add your message"`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Submit a pull request.

## License
This project is licensed under [MIT License](LICENSE).

## Repository
GitHub: [Online Retail Cluster and Classify](https://github.com/KFMBB/Online-Retail-Cluster-Classify)

## Hosted Application
Streamlit App: [https://online-retail-cluster-classify-mok.streamlit.app/](https://online-retail-cluster-classify-mok.streamlit.app/)
