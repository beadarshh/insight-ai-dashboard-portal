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
      q3: getPercentileValue(sortedValues, 0.75)
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
      
      // Simulate different ML models based on the prompt
      if (lowerPrompt.includes('cluster') || lowerPrompt.includes('k-means')) {
        resolve({
          type: 'clustering',
          title: 'K-Means Clustering Analysis',
          description: `Applied K-means clustering to identify natural groupings in your data. Found 3 distinct clusters based on the patterns in ${numericColumns.slice(0, 2).join(', ')}. The largest cluster contains 42% of your data points.`,
          pythonCode: `import pandas as pd\nfrom sklearn.cluster import KMeans\nfrom sklearn.preprocessing import StandardScaler\n\n# Load data into pandas DataFrame\ndf = pd.DataFrame(data)\n\n# Select numeric features for clustering\nfeatures = df[[${numericColumns.slice(0, 2).map(col => `"${col}"`).join(', ')}]]\n\n# Standardize features\nscaler = StandardScaler()\nscaled_features = scaler.fit_transform(features)\n\n# Apply K-means clustering\nkmeans = KMeans(n_clusters=3, random_state=42)\ndf['cluster'] = kmeans.fit_predict(scaled_features)\n\n# Analyze clusters\ncluster_stats = df.groupby('cluster').agg({'count', 'mean'})\nprint(cluster_stats)`,
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
        const targetColumn = numericColumns[0];
        const featureColumn = numericColumns.length > 1 ? numericColumns[1] : categoricalColumns[0];
        
        resolve({
          type: 'prediction',
          title: `Predictive Model for ${targetColumn}`,
          description: `Created a Random Forest regression model to predict ${targetColumn} based on available features. The model achieved an R² score of 0.83, indicating good predictive performance. The most important feature was ${featureColumn}.`,
          pythonCode: `import pandas as pd\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import r2_score\nimport numpy as np\n\n# Prepare data\ndf = pd.DataFrame(data)\n\n# Handle missing values\ndf = df.fillna(df.mean())\n\n# Define features and target\nX = df.drop('${targetColumn}', axis=1)\nX = pd.get_dummies(X) # Convert categorical variables\ny = df['${targetColumn}']\n\n# Split data\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n# Train model\nmodel = RandomForestRegressor(n_estimators=100, random_state=42)\nmodel.fit(X_train, y_train)\n\n# Evaluate model\ny_pred = model.predict(X_test)\nr2 = r2_score(y_test, y_pred)\nprint(f"R² Score: {r2}")\n\n# Feature importance\nfeature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})\nprint(feature_importance.sort_values('importance', ascending=False))`,
          modelInfo: 'Used scikit-learn RandomForestRegressor with 100 estimators, 80/20 train-test split',
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
        resolve({
          type: 'anomaly',
          title: 'Anomaly Detection Results',
          description: `Used Isolation Forest algorithm to detect anomalies in your data. Identified ${Math.floor(data.length * 0.03)} potential anomalies (approximately 3% of your dataset). These anomalies significantly deviate from the normal patterns and may represent errors or special cases worth investigating.`,
          pythonCode: `import pandas as pd\nfrom sklearn.ensemble import IsolationForest\nimport numpy as np\n\n# Load data\ndf = pd.DataFrame(data)\n\n# Select numeric features for anomaly detection\nnumeric_features = df.select_dtypes(include=[np.number]).columns\nX = df[numeric_features].fillna(df[numeric_features].mean())\n\n# Apply Isolation Forest\nisolation_forest = IsolationForest(contamination=0.03, random_state=42)\ndf['anomaly'] = isolation_forest.fit_predict(X)\n\n# -1 represents anomalies, 1 represents normal data\ndf['anomaly_status'] = df['anomaly'].map({1: 'normal', -1: 'anomaly'})\n\n# Get anomaly statistics\nanomaly_count = (df['anomaly'] == -1).sum()\ntotal_count = len(df)\nanomaly_percent = (anomaly_count / total_count) * 100\nprint(f"Detected {anomaly_count} anomalies ({anomaly_percent:.2f}% of the dataset)")\n\n# Look at distribution of anomalies\nanomaly_stats = df.groupby('anomaly_status').agg(['mean', 'std', 'min', 'max'])\nprint(anomaly_stats)`,
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
        resolve({
          type: 'nlp',
          title: 'Natural Language Processing Analysis',
          description: `Applied NLP techniques to analyze text content in your data. Extracted key topics and sentiments from the text columns. The most common topics were "product quality", "customer service", and "pricing". Sentiment analysis shows 62% positive, 23% neutral, and 15% negative sentiment.`,
          pythonCode: `import pandas as pd\nimport nltk\nfrom nltk.corpus import stopwords\nfrom nltk.tokenize import word_tokenize\nfrom nltk.sentiment import SentimentIntensityAnalyzer\nfrom sklearn.feature_extraction.text import CountVectorizer\nfrom sklearn.decomposition import LatentDirichletAllocation\n\n# Download required NLTK data\nntlk.download('punkt')\nntlk.download('stopwords')\nntlk.download('vader_lexicon')\n\n# Load data\ndf = pd.DataFrame(data)\n\n# Find text columns (assume first string column contains text)\ntext_column = df.select_dtypes(include=['object']).columns[0]\n\n# Sentiment analysis\nsia = SentimentIntensityAnalyzer()\ndf['sentiment_scores'] = df[text_column].apply(lambda x: sia.polarity_scores(str(x)) if pd.notna(x) else None)\ndf['sentiment'] = df['sentiment_scores'].apply(lambda x: 'positive' if x and x['compound'] > 0.05 else ('negative' if x and x['compound'] < -0.05 else 'neutral') if x else None)\n\n# Topic modeling\nvectorizer = CountVectorizer(stop_words='english', max_features=1000, max_df=0.95, min_df=0.01)\nX = vectorizer.fit_transform(df[text_column].fillna(''))\n\nlda = LatentDirichletAllocation(n_components=3, random_state=42)\nlda.fit(X)\n\n# Get top words for each topic\nfeature_names = vectorizer.get_feature_names_out()\ntop_words_per_topic = []\nfor topic_idx, topic in enumerate(lda.components_):\n    top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]\n    top_words_per_topic.append(top_words)\n\nsentiment_counts = df['sentiment'].value_counts(normalize=True) * 100\nprint(f"Sentiment distribution: {sentiment_counts.to_dict()}")\nprint(f"Top words per topic: {top_words_per_topic}")`,
          modelInfo: 'Used NLTK for sentiment analysis (VADER), scikit-learn for topic modeling (LDA)',
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
        resolve({
          type: 'summary',
          title: 'Python-Generated Data Summary',
          description: `Analyzed dataset with ${data.length} rows and ${Object.keys(data[0] || {}).length} columns using pandas and scikit-learn. The dataset contains ${numericColumns.length} numeric features and ${categoricalColumns.length} categorical features. Found ${getNumericColumns(data).filter(col => {
            const stats = calculateStatistics(data, col);
            return stats && (stats.missing || 0) > (data.length * 0.05);
          }).length} columns with >5% missing values. Applied PCA to reduce dimensionality and identify key patterns.`,
          pythonCode: `import pandas as pd\nimport numpy as np\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\n# Load data\ndf = pd.DataFrame(data)\n\n# Basic summary\nprint(f"Dataset shape: {df.shape}")\nprint(f"Columns: {df.columns.tolist()}")\nprint(f"Data types:\\n{df.dtypes}")\n\n# Missing values analysis\nmissing_counts = df.isnull().sum()\nmissing_percent = (missing_counts / len(df)) * 100\nprint(f"Missing values:\\n{pd.DataFrame({'count': missing_counts, 'percent': missing_percent})}")\n\n# Descriptive statistics\nprint(f"Numeric statistics:\\n{df.describe()}")\n\n# Correlation analysis\ncorr_matrix = df.select_dtypes(include=[np.number]).corr()\nprint(f"Top correlations:\\n{corr_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(10)}")\n\n# PCA for dimensionality reduction\nnumeric_df = df.select_dtypes(include=[np.number])\nX = numeric_df.fillna(numeric_df.mean())\nX_scaled = StandardScaler().fit_transform(X)\npca = PCA(n_components=2)\npca_result = pca.fit_transform(X_scaled)\nprint(f"Explained variance ratio: {pca.explained_variance_ratio_}")\nprint(f"Cumulative explained variance: {sum(pca.explained_variance_ratio_)}")`,
          modelInfo: 'Used pandas for data analysis, scikit-learn PCA for dimensionality reduction',
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
        // Default generic analysis
        resolve({
          type: 'generic',
          title: 'Python Data Analysis',
          description: `Performed exploratory data analysis on your dataset using pandas and scikit-learn. Dataset contains ${data.length} rows and ${Object.keys(data[0] || {}).length} features. Most of the information is concentrated in a few key features.`,
          pythonCode: `import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.preprocessing import StandardScaler\n\n# Load data\ndf = pd.DataFrame(data)\n\n# Basic EDA\nprint(f"Dataset shape: {df.shape}")\nprint(f"Data types:\\n{df.dtypes}")\nprint(f"Summary statistics:\\n{df.describe()}")\n\n# Check for missing values\nmissing = df.isnull().sum()\nprint(f"Missing values:\\n{missing[missing > 0]}")\n\n# Correlation analysis\nnum_df = df.select_dtypes(include=['number'])\ncorr = num_df.corr()\n\n# Plot correlation matrix\nplt.figure(figsize=(10, 8))\nsns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")\nplt.title('Feature Correlation Matrix')\nplt.tight_layout()\nplt.show()`,
          modelInfo: 'Used pandas for data analysis, matplotlib and seaborn for visualization',
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
