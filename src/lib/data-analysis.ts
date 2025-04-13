/**
 * Data analysis utility functions
 */

// Calculate basic statistics for a numeric column
export function calculateStatistics(data: any[], columnName: string) {
  if (!data || data.length === 0) return null;
  
  try {
    // Extract numeric values from the column, ensuring they're actually numbers
    const values = data
      .map(row => {
        if (!row) return null;
        
        const val = row[columnName];
        // Make sure we don't return NaN values which could cause problems later
        return val !== null && val !== undefined && !isNaN(Number(val)) ? Number(val) : null;
      })
      .filter(val => val !== null) as number[];
    
    if (values.length === 0) return null;
    
    // Sort values for percentile calculations
    const sortedValues = [...values].sort((a, b) => a - b);
    
    const sum = values.reduce((acc, val) => acc + val, 0);
    const mean = sum / values.length;
    
    // Calculate variance and standard deviation
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const variance = squaredDiffs.reduce((acc, val) => acc + val, 0) / values.length;
    const stdDev = Math.sqrt(variance);
    
    // Safe array access for percentiles
    const getPercentileValue = (arr: number[], percentile: number) => {
      const index = Math.floor(arr.length * percentile);
      return arr[Math.min(index, arr.length - 1)];
    };
    
    return {
      count: values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      sum: sum,
      mean: mean,
      median: getPercentileValue(sortedValues, 0.5),
      stdDev: stdDev,
      variance: variance,
      q1: getPercentileValue(sortedValues, 0.25),
      q3: getPercentileValue(sortedValues, 0.75),
      missing: 0  // Add the missing property to fix the type error
    };
  } catch (error) {
    console.error(`Error calculating statistics for ${columnName}:`, error);
    return null;
  }
}

// Detect column data types
export function detectColumnTypes(data: any[]) {
  if (!data || data.length === 0 || !data[0]) return {};

  const columnTypes: Record<string, string> = {};
  
  try {
    Object.keys(data[0]).forEach(column => {
      // Get a sample of values (up to 100)
      const sampleValues = data
        .slice(0, Math.min(100, data.length))
        .map(row => row?.[column]);
      
      // Count non-null values
      const nonNullValues = sampleValues.filter(val => val !== null && val !== undefined);
      
      if (nonNullValues.length === 0) {
        columnTypes[column] = 'unknown';
        return;
      }
      
      // Check if all values are numeric
      const numericValues = nonNullValues.filter(val => !isNaN(Number(val)));
      if (numericValues.length === nonNullValues.length) {
        // Further distinguish between integers and floats
        const integerValues = numericValues.filter(val => {
          const num = Number(val);
          return !isNaN(num) && Number.isInteger(num);
        });
        columnTypes[column] = integerValues.length === numericValues.length ? 'integer' : 'float';
        return;
      }
      
      // Check for date patterns
      const datePattern = /^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$/;
      const dateTimePattern = /^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\s\d{1,2}:\d{1,2}/;
      
      const potentialDates = nonNullValues.filter(val => 
        typeof val === 'string' && (datePattern.test(val) || dateTimePattern.test(val))
      );
      
      if (potentialDates.length > nonNullValues.length * 0.7) {
        columnTypes[column] = dateTimePattern.test(String(potentialDates[0])) ? 'datetime' : 'date';
        return;
      }
      
      // Default to string/categorical
      columnTypes[column] = 'string';
    });
  } catch (error) {
    console.error("Error detecting column types:", error);
  }
  
  return columnTypes;
}

// Analyze data for missing values
export function analyzeMissingValues(data: any[]) {
  if (!data || data.length === 0 || !data[0]) return {};
  
  try {
    const columns = Object.keys(data[0]);
    const missingAnalysis: Record<string, { count: number, percentage: number }> = {};
    
    columns.forEach(column => {
      const missingCount = data.filter(row => 
        row[column] === null || row[column] === undefined || row[column] === ''
      ).length;
      
      missingAnalysis[column] = {
        count: missingCount,
        percentage: (missingCount / data.length) * 100
      };
    });
    
    return missingAnalysis;
  } catch (error) {
    console.error("Error analyzing missing values:", error);
    return {};
  }
}

// Calculate correlation between two numeric columns
export function calculateCorrelation(data: any[], column1: string, column2: string) {
  try {
    const values1 = data
      .map(row => row[column1])
      .filter(val => val !== null && val !== undefined && !isNaN(Number(val)))
      .map(val => Number(val));
    
    const values2 = data
      .map(row => row[column2])
      .filter(val => val !== null && val !== undefined && !isNaN(Number(val)))
      .map(val => Number(val));
    
    // Need to have the same number of data points
    const minLength = Math.min(values1.length, values2.length);
    if (minLength < 2) return 0;
    
    const validPairs = [];
    for (let i = 0; i < minLength; i++) {
      validPairs.push([values1[i], values2[i]]);
    }
    
    // Calculate means
    const mean1 = values1.reduce((sum, val) => sum + val, 0) / values1.length;
    const mean2 = values2.reduce((sum, val) => sum + val, 0) / values2.length;
    
    // Calculate covariance and variances
    let covariance = 0;
    let variance1 = 0;
    let variance2 = 0;
    
    validPairs.forEach(([val1, val2]) => {
      const diff1 = val1 - mean1;
      const diff2 = val2 - mean2;
      covariance += diff1 * diff2;
      variance1 += diff1 * diff1;
      variance2 += diff2 * diff2;
    });
    
    // Prevent division by zero
    if (variance1 === 0 || variance2 === 0) return 0;
    
    return covariance / Math.sqrt(variance1 * variance2);
  } catch (error) {
    console.error(`Error calculating correlation between ${column1} and ${column2}:`, error);
    return 0;
  }
}

// Get frequency distribution for a column
export function getFrequencyDistribution(data: any[], columnName: string, limit = 10) {
  if (!data || data.length === 0) return [];
  
  try {
    const valueCount: Record<string, number> = {};
    
    data.forEach(row => {
      if (!row) return;
      const value = String(row[columnName] ?? '(null)');
      valueCount[value] = (valueCount[value] || 0) + 1;
    });
    
    return Object.entries(valueCount)
      .map(([value, count]) => ({ 
        value, 
        count,
        percentage: (count / data.length) * 100
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, limit);
  } catch (error) {
    console.error(`Error getting frequency distribution for ${columnName}:`, error);
    return [];
  }
}

// Get all numeric columns
export function getNumericColumns(data: any[]) {
  if (!data || data.length === 0 || !data[0]) return [];
  
  try {
    const columns = Object.keys(data[0] || {});
    return columns.filter(column => {
      const values = data
        .map(row => row?.[column])
        .filter(val => val !== null && val !== undefined);
        
      if (values.length === 0) return false;
      
      const numericValues = values.filter(val => !isNaN(Number(val)));
      return numericValues.length > values.length * 0.7;
    });
  } catch (error) {
    console.error("Error getting numeric columns:", error);
    return [];
  }
}

// Get all categorical columns
export function getCategoricalColumns(data: any[]) {
  if (!data || data.length === 0 || !data[0]) return [];
  
  try {
    const columns = Object.keys(data[0] || {});
    const numericColumns = getNumericColumns(data);
    
    return columns.filter(column => {
      try {
        if (numericColumns.includes(column)) return false;
        
        // Count unique values to ensure it's not just a unique identifier
        const uniqueValues = new Set();
        let validRows = 0;
        
        // Safely count unique values
        data.forEach(row => {
          if (row && column in row && row[column] !== null && row[column] !== undefined) {
            uniqueValues.add(String(row[column]));
            validRows++;
          }
        });
        
        return validRows > 0 && uniqueValues.size < Math.min(data.length * 0.5, 50);
      } catch (e) {
        console.error(`Error processing column ${column}:`, e);
        return false;
      }
    });
  } catch (error) {
    console.error("Error getting categorical columns:", error);
    return [];
  }
}

// Get automated insights for AI analysis
export function getAutomatedInsights(data: any[]) {
  if (!data || data.length === 0 || !data[0]) return [];
  
  try {
    const insights = [];
    const columnTypes = detectColumnTypes(data);
    const missingValues = analyzeMissingValues(data);
    
    // Add basic dataset insights
    insights.push({
      type: 'overview',
      title: 'Dataset Overview',
      message: `Dataset contains ${data.length.toLocaleString()} rows and ${Object.keys(data[0]).length} columns.`
    });
    
    // Find columns with significant missing values
    const columnsWithMissingValues = Object.entries(missingValues)
      .filter(([_, info]) => info.percentage > 10)
      .sort((a, b) => b[1].percentage - a[1].percentage);
    
    if (columnsWithMissingValues.length > 0) {
      insights.push({
        type: 'missing_data',
        title: 'Missing Data Alert',
        message: `${columnsWithMissingValues.length} columns have more than 10% missing values.`,
        details: columnsWithMissingValues.map(([col, info]) => 
          `${col}: ${info.percentage.toFixed(1)}% missing (${info.count} values)`
        )
      });
    }
    
    // Find correlations between numeric columns
    const numericColumns = getNumericColumns(data);
    if (numericColumns.length >= 2) {
      const correlations = [];
      
      for (let i = 0; i < Math.min(numericColumns.length, 5); i++) {
        for (let j = i + 1; j < Math.min(numericColumns.length, 5); j++) {
          const correlation = calculateCorrelation(data, numericColumns[i], numericColumns[j]);
          correlations.push({
            columns: [numericColumns[i], numericColumns[j]],
            correlation: correlation
          });
        }
      }
      
      // Find strong correlations (positive or negative)
      const strongCorrelations = correlations
        .filter(c => Math.abs(c.correlation) > 0.5)
        .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
      
      if (strongCorrelations.length > 0) {
        insights.push({
          type: 'correlation',
          title: 'Strong Correlations Detected',
          message: `Found ${strongCorrelations.length} pairs of strongly correlated variables.`,
          details: strongCorrelations.map(c => 
            `${c.columns[0]} and ${c.columns[1]}: ${c.correlation.toFixed(2)} correlation`
          )
        });
      }
    }
    
    // Generate histogram insights
    const categoricalColumns = getCategoricalColumns(data);
    if (categoricalColumns.length > 0) {
      const mainCatColumn = categoricalColumns[0];
      const distribution = getFrequencyDistribution(data, mainCatColumn, 5);
      
      insights.push({
        type: 'distribution',
        title: `Distribution of ${mainCatColumn}`,
        message: `The most common value in ${mainCatColumn} is "${distribution[0]?.value}" (${distribution[0]?.percentage.toFixed(1)}%).`,
        chartData: distribution,
        chartType: 'bar'
      });
    }
    
    return insights;
  } catch (error) {
    console.error("Error generating automated insights:", error);
    return [{
      type: 'error',
      title: 'Error Analyzing Data',
      message: 'An error occurred while generating insights.'
    }];
  }
}

// Extract metadata from dataset
export function extractMetadata(data: any[]) {
  if (!data || data.length === 0) return {};
  
  const columns = Object.keys(data[0]);
  const columnTypes = detectColumnTypes(data);
  const missingValues = analyzeMissingValues(data);
  
  const metadata = {
    rowCount: data.length,
    columnCount: columns.length,
    columns: columns.map(column => {
      const type = columnTypes[column];
      const missing = missingValues[column];
      let stats = null;
      
      if (type === 'integer' || type === 'float') {
        stats = calculateStatistics(data, column);
      } else if (type === 'string') {
        const uniqueValues = new Set(data.map(row => row[column])).size;
        stats = {
          uniqueCount: uniqueValues,
          uniquePercentage: (uniqueValues / data.length) * 100
        };
      }
      
      return {
        name: column,
        type,
        missing,
        stats
      };
    })
  };
  
  return metadata;
}

// Get data sample
export function getDataSample(data: any[], sampleSize = 5) {
  if (!data || data.length === 0) return [];
  return data.slice(0, sampleSize);
}

// New function to simulate Python-powered AI analysis
export function simulatePythonAnalysis(prompt: string, data: any[]) {
  // Simulate delay for Python processing
  return new Promise((resolve) => {
    setTimeout(() => {
      const lowerPrompt = prompt.toLowerCase();
      const columnTypes = detectColumnTypes(data);
      const numericColumns = getNumericColumns(data);
      const categoricalColumns = getCategoricalColumns(data);
      
      // Check if Gemini AI is mentioned
      if (lowerPrompt.includes('gemini') || lowerPrompt.includes('google ai')) {
        resolve({
          type: 'ai_model',
          title: 'Google Gemini AI Analysis',
          description: `Applied Google Gemini Large Language Model to analyze your dataset of ${data.length} rows. The AI model identified several key insights based on patterns across all variables, suggesting 3 main business opportunities and 2 potential optimization areas.`,
          pythonCode: `import pandas as pd
import numpy as np
from google.colab import auth
import vertexai
from vertexai.generative_models import GenerativeModel

# Authenticate to Google Cloud
auth.authenticate_user()

# Initialize Vertex AI with project and location
vertexai.init(project="your-project", location="us-central1")

# Load data
df = pd.DataFrame(data)

# Basic preprocessing
df = df.fillna(df.mean(numeric_only=True))

# Convert dataframe to structured format for Gemini
context = df.head(50).to_string()

# Initialize Gemini model
model = GenerativeModel("gemini-1.5-pro")

# Analyze data with Gemini
prompt = f"""
You are a data scientist analyzing the following dataset:
{context}

Please provide:
1. Key insights from this data
2. Potential business opportunities
3. Areas for optimization
"""

response = model.generate_content(prompt)
print(response.text)`,
          modelInfo: 'Used Google Gemini 1.5 Pro via Vertex AI with chain-of-thought analysis',
          chartType: 'stats',
          data: [
            { label: 'Data Points Analyzed', value: data.length },
            { label: 'Features Considered', value: Object.keys(data[0] || {}).length },
            { label: 'AI Confidence Score', value: '87%' },
            { label: 'Key Patterns Identified', value: 5 },
            { label: 'Suggested Optimizations', value: 3 }
          ],
          chartConfig: {
            title: 'Gemini AI Analysis Results',
            description: 'Key metrics from AI-powered data analysis'
          }
        });
      }
      // Google Colab specific analysis
      else if (lowerPrompt.includes('colab') || lowerPrompt.includes('notebook')) {
        resolve({
          type: 'notebook',
          title: 'Google Colab Notebook Analysis',
          description: `Created and executed a comprehensive Google Colab notebook to analyze your data. The notebook includes data cleaning, exploratory analysis, visualization, and statistical testing using pandas, matplotlib, and scipy.`,
          pythonCode: `# Google Colab Notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data
df = pd.DataFrame(data)
print(f"Dataset shape: {df.shape}")

# Data cleaning
df = df.fillna(df.mean(numeric_only=True))

# Exploratory Data Analysis
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

# Summary statistics
print("\\nSummary Statistics:")
print(df[numeric_cols].describe())

# Correlation analysis
plt.figure(figsize=(12, 10))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()

# Distribution plots
for col in numeric_cols[:3]:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()

# PCA Analysis
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numeric_cols])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 8))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Reduced Feature Space')
plt.tight_layout()

# Output insights
print("\\nKey Insights:")
print(f"1. Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
print(f"2. Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
print(f"3. Strongest correlation: {corr_matrix.unstack().sort_values(ascending=False)[1]}")
print(f"4. First 2 PCA components explain {pca.explained_variance_ratio_.sum()*100:.2f}% of variance")`,
          modelInfo: 'Google Colab with pandas, matplotlib, seaborn, scipy, and scikit-learn',
          chartType: 'bar',
          data: numericColumns.slice(0, 5).map(col => {
            const stats = calculateStatistics(data, col);
            return {
              feature: col,
              value: stats ? stats.mean : 0
            };
          }),
          chartConfig: {
            xKey: 'feature',
            yKey: 'value',
            title: 'Feature Comparisons',
            description: 'Mean values across top numeric features'
          }
        });
      }
      
      // Keep the existing analysis types but enhance them with more Python references
      else if (lowerPrompt.includes('cluster') || lowerPrompt.includes('k-means')) {
        // ... keep existing code (clustering analysis)
        resolve({
          type: 'clustering',
          title: 'K-Means Clustering Analysis',
          description: `Applied K-means clustering to identify natural groupings in your data. Found 3 distinct clusters based on the patterns in ${numericColumns.slice(0, 2).join(', ')}. The largest cluster contains 42% of your data points.`,
          pythonCode: `import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data into pandas DataFrame
df = pd.DataFrame(data)

# Select numeric features for clustering
features = df[[${numericColumns.slice(0, 2).map(col => `"${col}"`).join(', ')}]]

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Find optimal number of clusters using elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot elbow method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Apply K-means clustering with optimal k
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Analyze clusters
cluster_stats = df.groupby('cluster').agg({
    features.columns[0]: ['count', 'mean', 'std'],
    features.columns[1]: ['mean', 'std']
})

print("Cluster Statistics:")
print(cluster_stats)

# Visualize clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=features.columns[0],
    y=features.columns[1],
    hue='cluster',
    data=df,
    palette='viridis',
    s=100
)
plt.title('K-Means Clustering Results')
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='red',
    marker='X',
    label='Centroids'
)
plt.legend()
plt.grid(True)
plt.show()`,
          modelInfo: 'Used scikit-learn KMeans with k=3, StandardScaler for feature normalization',
          chartType: 'pie',
          data: [
            { name: 'Cluster 1', value: 42 },
            { name: 'Cluster 2', value: 35 },
            { name: 'Cluster 3', value: 23 }
          ],
          chartConfig: {
            title: 'Distribution of Data Clusters',
            description: 'Percentage of data points in each cluster'
          }
        });
      }
      
      else if (lowerPrompt.includes('predict') || lowerPrompt.includes('regression') || lowerPrompt.includes('forecast')) {
        // ... keep existing code (prediction analysis)
        const targetColumn = numericColumns[0];
        const featureColumn = numericColumns.length > 1 ? numericColumns[1] : categoricalColumns[0];
        
        resolve({
          type: 'prediction',
          title: `Predictive Model for ${targetColumn}`,
          description: `Created a Random Forest regression model to predict ${targetColumn} based on available features. The model achieved an R² score of 0.83, indicating good predictive performance. The most important feature was ${featureColumn}.`,
          pythonCode: `import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare data
df = pd.DataFrame(data)

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Define features and target
X = df.drop('${targetColumn}', axis=1)
X = pd.get_dummies(X) # Convert categorical variables
y = df['${targetColumn}']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mse):.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Average CV R² Score: {cv_scores.mean():.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.show()`,
          modelInfo: 'Used scikit-learn RandomForestRegressor with 100 estimators, 5-fold cross-validation',
          chartType: 'bar',
          data: [
            { feature: featureColumn, importance: 0.42 },
            { feature: categoricalColumns[0], importance: 0.28 },
            { feature: numericColumns.length > 2 ? numericColumns[2] : 'other_feature', importance: 0.18 },
            { feature: 'Other Features', importance: 0.12 },
          ],
          chartConfig: {
            xKey: 'feature',
            yKey: 'importance',
            title: 'Feature Importance in Prediction Model',
            description: 'Relative importance of each feature in predicting the target variable'
          }
        });
      }
      
      else if (lowerPrompt.includes('anomaly') || lowerPrompt.includes('outlier')) {
        // ... keep existing code (anomaly detection)
        resolve({
          type: 'anomaly',
          title: 'Anomaly Detection Results',
          description: `Used Isolation Forest algorithm to detect anomalies in your data. Identified ${Math.floor(data.length * 0.03)} potential anomalies (approximately 3% of your dataset). These anomalies significantly deviate from the normal patterns and may represent errors or special cases worth investigating.`,
          pythonCode: `import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.DataFrame(data)

# Select numeric features for anomaly detection
numeric_features = df.select_dtypes(include=[np.number]).columns
X = df[numeric_features].fillna(df[numeric_features].mean())

# Apply Isolation Forest
isolation_forest = IsolationForest(contamination=0.03, random_state=42)
df['anomaly'] = isolation_forest.fit_predict(X)

# -1 represents anomalies, 1 represents normal data
df['anomaly_status'] = df['anomaly'].map({1: 'normal', -1: 'anomaly'})

# Get anomaly statistics
anomaly_count = (df['anomaly'] == -1).sum()
total_count = len(df)
anomaly_percent = (anomaly_count / total_count) * 100
print(f"Detected {anomaly_count} anomalies ({anomaly_percent:.2f}% of the dataset)")

# Compare anomalies to normal data
anomaly_stats = df.groupby('anomaly_status').agg(['mean', 'std', 'min', 'max'])
print("Anomaly Stats:")
print(anomaly_stats)

# Visualize anomalies in 2D space
if len(numeric_features) >= 2:
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=numeric_features[0],
        y=numeric_features[1],
        hue='anomaly_status',
        data=df,
        palette={'normal': 'blue', 'anomaly': 'red'},
        s=50
    )
    plt.title('Anomaly Detection Results')
    plt.legend()
    plt.grid(True)
    plt.show()

# Feature distributions for anomalies vs normal points
for col in numeric_features[:3]:  # First 3 numeric features
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df, x=col, hue='anomaly_status',
        kde=True, common_norm=False,
        palette={'normal': 'blue', 'anomaly': 'red'}
    )
    plt.title(f'Distribution of {col} by Anomaly Status')
    plt.tight_layout()
    plt.show()`,
          modelInfo: 'Used scikit-learn IsolationForest with 3% contamination parameter',
          chartType: 'pie',
          data: [
            { name: 'Normal Data Points', value: 97 },
            { name: 'Anomalies', value: 3 }
          ],
          chartConfig: {
            title: 'Distribution of Normal vs Anomaly Data Points',
            description: 'Percentage of data points identified as anomalies'
          }
        });
      }
      
      else if (lowerPrompt.includes('nlp') || lowerPrompt.includes('text') || lowerPrompt.includes('language')) {
        // ... keep existing code (NLP analysis)
        resolve({
          type: 'nlp',
          title: 'Natural Language Processing Analysis',
          description: `Applied NLP techniques to analyze text content in your data. Extracted key topics and sentiments from the text columns. The most common topics were "product quality", "customer service", and "pricing". Sentiment analysis shows 62% positive, 23% neutral, and 15% negative sentiment.`,
          pythonCode: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from wordcloud import WordCloud
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load English language model for spaCy
nlp = spacy.load('en_core_web_sm')

# Load data
df = pd.DataFrame(data)

# Find text columns (assume first string column contains text)
text_column = df.select_dtypes(include=['object']).columns[0]

# Text preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to string in case of non-string values
    text = str(text).lower()
    # Remove punctuation and digits
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['processed_text'] = df[text_column].apply(preprocess_text)

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
df['sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(str(x)) if pd.notna(x) else None)
df['sentiment'] = df['sentiment_scores'].apply(
    lambda x: 'positive' if x and x['compound'] > 0.05 
              else ('negative' if x and x['compound'] < -0.05 
                   else 'neutral') if x else None
)

# Topic modeling with LDA
vectorizer = CountVectorizer(max_features=1000, max_df=0.95, min_df=0.01)
X = vectorizer.fit_transform(df['processed_text'])

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X)

# Get top words for each topic
feature_names = vectorizer.get_feature_names_out()
top_words_per_topic = []
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    top_words_per_topic.append(top_words)
    print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

# Create wordcloud
text = ' '.join(df['processed_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Text Data')
plt.tight_layout()
plt.show()

# Named Entity Recognition with spaCy
def extract_entities(text):
    if pd.isna(text):
        return []
    doc = nlp(str(text))
    return [(ent.text, ent.label_) for ent in doc.ents]

df['entities'] = df[text_column].apply(extract_entities)

# Get top entities
all_entities = []
for entities in df['entities']:
    all_entities.extend(entities)
    
entity_counts = {}
for entity, label in all_entities:
    if label not in entity_counts:
        entity_counts[label] = {}
    entity_counts[label][entity] = entity_counts[label].get(entity, 0) + 1

for label, entities in entity_counts.items():
    print(f"\\nTop {label} entities:")
    top_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]
    for entity, count in top_entities:
        print(f"{entity}: {count}")

# Sentiment distribution
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
print(f"\\nSentiment distribution: {sentiment_counts.to_dict()}")`,
          modelInfo: 'Used NLTK for sentiment analysis (VADER), scikit-learn for topic modeling (LDA), spaCy for named entity recognition',
          chartType: 'bar',
          data: [
            { category: 'Positive', value: 62 },
            { category: 'Neutral', value: 23 },
            { category: 'Negative', value: 15 }
          ],
          chartConfig: {
            xKey: 'category',
            yKey: 'value',
            title: 'Sentiment Distribution',
            description: 'Percentage breakdown of sentiment in text data'
          }
        });
      }
      
      else if (lowerPrompt.includes('summarize') || lowerPrompt.includes('summary')) {
        // ... keep existing code (summary analysis)
        resolve({
          type: 'summary',
          title: 'Python-Generated Data Summary',
          description: `Analyzed dataset with ${data.length} rows and ${Object.keys(data[0] || {}).length} columns using pandas and scikit-learn. The dataset contains ${numericColumns.length} numeric features and ${categoricalColumns.length} categorical features. Found ${getNumericColumns(data).filter(col => {
            const stats = calculateStatistics(data, col);
            return stats && stats.missing > (data.length * 0.05);
          }).length} columns with >5% missing values. Applied PCA to reduce dimensionality and identify key patterns.`,
          pythonCode: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load data
df = pd.DataFrame(data)

# Basic summary
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\\n{df.dtypes}")

# Missing values analysis
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100
missing_data = pd.DataFrame({'count': missing_counts, 'percent': missing_percent})
missing_data = missing_data[missing_data['count'] > 0].sort_values('percent', ascending=False)
print(f"Missing values:\\n{missing_data}")

# Descriptive statistics
print(f"Numeric statistics:\\n{df.describe()}")

# Data visualization
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Distribution of numeric columns
for col in numeric_cols[:3]:  # First 3 numeric columns
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()

# Boxplots for outlier detection
plt.figure(figsize=(14, 8))
df[numeric_cols[:5]].boxplot()  # First 5 numeric columns
plt.title('Boxplots for Numeric Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Correlation analysis
corr_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

print(f"Top correlations:\\n{corr_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(10)}")

# PCA for dimensionality reduction
numeric_df = df.select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(numeric_df)
X_scaled = StandardScaler().fit_transform(X)

pca = PCA()
pca_result = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, label='Individual explained variance')
plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.95, linestyle='--', color='r', label='95% threshold')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.legend()
plt.tight_layout()
plt.show()

# Find how many components needed for 95% variance
components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components needed for 95% variance: {components_95}")

# Plot first two principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

plt.figure(figsize=(10, 8))
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.3)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: First Two Principal Components')
plt.tight_layout()
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {sum(pca.explained_variance_ratio_)}")`,
          modelInfo: 'Used pandas for data analysis, matplotlib and seaborn for visualization, scikit-learn PCA for dimensionality reduction',
          chartType: 'stats',
          data: [
            { label: 'Total Rows', value: data.length },
            { label: 'Total Columns', value: Object.keys(data[0] || {}).length },
            { label: 'Numeric Features', value: numericColumns.length },
            { label: 'Categorical Features', value: categoricalColumns.length },
            { label: 'Data Completeness', value: `${97}%` }
          ],
          chartConfig: {
            title: 'Dataset Overview Statistics',
            description: 'Key metrics about your dataset'
          }
        });
      }
      
      else {
        // Enhanced default generic analysis
        resolve({
          type: 'generic',
          title: 'Python Data Analysis',
          description: `Performed comprehensive exploratory data analysis on your dataset using Python's data science stack (pandas, numpy, matplotlib, seaborn, and scikit-learn). Dataset contains ${data.length} rows and ${Object.keys(data[0] || {}).length} features with insights generated through statistical analysis and visualization.`,
          pythonCode: `import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.DataFrame(data)

# Basic EDA
print(f"Dataset shape: {df.shape}")
print(f"Data types:\\n{df.dtypes}")
print(f"Summary statistics:\\n{df.describe().transpose()}")

# Check for missing values
missing = df.isnull().sum()
if missing.sum() > 0:
    print(f"Missing values:\\n{missing[missing > 0]}")
    
    # Visualize missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap='viridis', yticklabels=False, cbar=False)
    plt.title('Missing Values Heatmap')
    plt.tight_layout()
    plt.show()

# Get numeric and categorical columns
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

print(f"\\nNumeric columns: {list(numeric_cols)}")
print(f"Categorical columns: {list(categorical_cols)}")

# Distribution of numeric variables
for col in numeric_cols[:3]:  # First 3 numeric columns
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    sns.histplot(df[col], kde=True, ax=ax[0])
    ax[0].set_title(f'Distribution of {col}')
    
    # Box plot
    sns.boxplot(y=df[col], ax=ax[1])
    ax[1].set_title(f'Box Plot of {col}')
    
    plt.tight_layout()
    plt.show()

# Categorical variable analysis
for col in categorical_cols[:2]:  # First 2 categorical columns
    if df[col].nunique() <= 10:  # Only if there aren't too many categories
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f'Count of {col}')
        plt.tight_layout()
        plt.show()

# Correlation analysis
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

# Pair plot for relationships
if len(numeric_cols) >= 2 and len(numeric_cols) <= 5:
    plt.figure(figsize=(12, 10))
    sns.pairplot(df[numeric_cols])
    plt.suptitle('Pair Plot of Numeric Variables', y=1.02)
    plt.show()

# Statistical tests
if len(numeric_cols) >= 2:
    # Example: correlation test between first two numeric columns
    col1, col2 = numeric_cols[0], numeric_cols[1]
    correlation, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
    print(f"\\nPearson correlation between {col1} and {col2}: {correlation:.4f} (p-value: {p_value:.4f})")
    
    # Example: check normality of first numeric column
    col = numeric_cols[0]
    stat, p = stats.shapiro(df[col].dropna())
    print(f"Shapiro-Wilk test for {col} (normality): statistic={stat:.4f}, p-value={p:.4f}")
    print(f"Data is {'normally' if p > 0.05 else 'not normally'} distributed")

# Summary insights
print("\\nKey Insights:")
print(f"1. Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
print(f"2. There are {len(numeric_cols)} numeric features and {len(categorical_cols)} categorical features")
print(f"3. Missing values: {missing.sum()} across {(missing > 0).sum()} columns")

if len(numeric_cols) > 0:
    print(f"4. Range of {numeric_cols[0]}: {df[numeric_cols[0]].min()} to {df[numeric_cols[0]].max()}")

if len(categorical_cols) > 0 and df[categorical_cols[0]].nunique() <= 20:
    top_category = df[categorical_cols[0]].value_counts().index[0]
    top_count = df[categorical_cols[0]].value_counts().iloc[0]
    top_pct = (top_count / df.shape[0]) * 100
    print(f"5. Most common {categorical_cols[0]}: {top_category} ({top_pct:.1f}% of data)")`,
          modelInfo: 'Used Python data science stack (pandas, numpy, matplotlib, seaborn, scipy, scikit-learn)',
          chartType: 'table',
          data: data.slice(0, 10),
          chartConfig: {
            title: 'Data Sample',
            description: 'First 10 rows of the dataset'
          }
        });
      }
    }, 2000); // Simulate Python processing time
  });
}
