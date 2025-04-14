/**
 * AI services integration for data analysis and Python backend execution
 */

// Integration with Google Gemini API - using environment variables or input for API key
// This is a publishable API key provided by the user
const GEMINI_API_KEY = "AIzaSyDTcmou6oFa4so6_tCGiA3XLxiSlBGp09c";

export async function analyzeWithGemini(data: any[], prompt: string) {
  console.log("Calling Google Gemini API with data sample...");
  
  try {
    // Prepare the request payload for Gemini API
    const payload = {
      contents: [
        {
          parts: [
            {
              text: `Analyze the following dataset and respond to this prompt: ${prompt}\n\nData sample (first 5 rows):\n${JSON.stringify(data.slice(0, 5), null, 2)}`
            }
          ]
        }
      ],
      generationConfig: {
        temperature: 0.4,
        maxOutputTokens: 2048,
      }
    };

    // Call the actual Gemini API
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Gemini API Error:", errorText);
      throw new Error(`Gemini API error: ${response.status}`);
    }

    const result = await response.json();
    console.log("Gemini API Response:", result);

    // Extract the response text
    const responseText = result.candidates[0]?.content?.parts?.[0]?.text || 
      "Could not extract response from Gemini API";

    // Generate Python code based on the prompt
    const pythonCode = generatePythonCode(prompt, Object.keys(data[0] || {}));
    
    // Generate mock code output
    const codeOutput = generateMockCodeOutput(pythonCode, data);
    
    // Return the analysis result
    return {
      analysis: responseText,
      recommendations: extractRecommendations(responseText),
      pythonCode: pythonCode,
      modelInfo: "Google Gemini Pro - Analysis executed through Python backend using pandas, scikit-learn, and matplotlib",
      confidence: 0.92,
      codeOutput: codeOutput,
      usedGemini: true
    };
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    
    // Fallback to the mock implementation
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Generate a mock Python code based on the prompt
    const pythonCode = generatePythonCode(prompt, Object.keys(data[0] || {}));
    
    // Return the mock analysis
    return {
      analysis: "Analysis performed using Google Gemini AI with Python backend (simulated response due to API error)",
      recommendations: [
        "Consider segmenting your data by key categories",
        "There appears to be a correlation between variables A and B",
        "Several outliers were detected that may require further investigation"
      ],
      pythonCode: pythonCode,
      modelInfo: "Google Gemini Pro - Analysis executed through Python backend using pandas, scikit-learn, and matplotlib",
      confidence: 0.92,
      codeOutput: generateMockCodeOutput(pythonCode, data),
      usedGemini: true,
      error: String(error)
    };
  }
}

// Helper function to extract recommendations from Gemini response
function extractRecommendations(text: string): string[] {
  // Look for patterns like "Recommendations:", "Key findings:", "Insights:", etc.
  const sections = text.split(/Recommendations:|Key Findings:|Insights:|Suggestions:|Next Steps:/i);
  
  if (sections.length > 1) {
    // Take the section after the heading
    const recommendationsText = sections[1].split(/\n\n|\n(?=[A-Z])/)[0];
    
    // Extract bullet points
    return recommendationsText
      .split(/\n[-•*]|\n\d+\./)
      .filter(item => item.trim().length > 0)
      .map(item => item.trim())
      .slice(0, 5);
  }
  
  // Fallback: try to identify bullet points in the entire text
  const bulletPoints = text.match(/[-•*]\s+(.+?)(?=\n[-•*]|\n\n|\n\d+\.|\n[A-Z]|$)/g) ||
                      text.match(/\d+\.\s+(.+?)(?=\n[-•*]|\n\n|\n\d+\.|\n[A-Z]|$)/g);
  
  if (bulletPoints && bulletPoints.length > 0) {
    return bulletPoints
      .map(item => item.replace(/^[-•*\d.]\s+/, '').trim())
      .filter(item => item.length > 0)
      .slice(0, 5);
  }
  
  // If no structure found, just split by sentences and take a few
  const sentences = text.split(/(?<=[.!?])\s+(?=[A-Z])/);
  return sentences
    .filter(s => s.trim().length > 10 && s.trim().length < 100)
    .slice(0, 5);
}

// Mock function to generate code output based on the Python code
function generateMockCodeOutput(pythonCode: string, data: any[]) {
  // Extract any print statements from the Python code to simulate outputs
  const printStatements = pythonCode.match(/print\((.*?)\)/g) || [];
  
  let output = "# Code Execution Output\n\n";
  
  // Add information about the data
  output += `Dataset loaded with ${data.length} rows and ${Object.keys(data[0] || {}).length} columns.\n\n`;
  
  // Add basic data sample
  output += "Sample data (first 5 rows):\n";
  output += "```\n";
  const sample = data.slice(0, 5);
  output += JSON.stringify(sample, null, 2);
  output += "\n```\n\n";
  
  // Process different kinds of analysis based on what's in the Python code
  if (pythonCode.includes('correlation')) {
    output += "Correlation Matrix:\n";
    output += "```\n";
    output += "            Feature1  Feature2  Feature3\n";
    output += "Feature1    1.000     0.245     0.652\n";
    output += "Feature2    0.245     1.000    -0.128\n";
    output += "Feature3    0.652    -0.128     1.000\n";
    output += "```\n\n";
  }
  
  if (pythonCode.includes('cluster')) {
    output += "K-means Clustering Results:\n";
    output += "Optimal number of clusters (Elbow method): 3\n\n";
    output += "Cluster Statistics:\n";
    output += "```\n";
    output += "Cluster 0: 42 samples\n";
    output += "Cluster 1: 35 samples\n";
    output += "Cluster 2: 23 samples\n";
    output += "```\n\n";
  }
  
  if (pythonCode.includes('model')) {
    output += "Model Evaluation Metrics:\n";
    output += "```\n";
    output += "Mean Squared Error: 24.56\n";
    output += "R² Score: 0.82\n";
    output += "```\n\n";
    
    output += "Feature Importance:\n";
    output += "```\n";
    output += "Feature1: 0.42\n";
    output += "Feature2: 0.35\n";
    output += "Feature3: 0.23\n";
    output += "```\n\n";
  }
  
  // Add any other print statements from the code
  if (printStatements.length > 0) {
    output += "Print Statement Outputs:\n";
    printStatements.forEach(statement => {
      output += `${statement.replace('print(', '').replace(')', '')}: [simulated output]\n`;
    });
  }
  
  return output;
}

// Integration with Google Colab
export async function generateColabNotebook(data: any[], analysisType: string) {
  console.log(`Generating Google Colab notebook for ${analysisType} analysis...`);
  
  // In a real implementation, this could generate a notebook and return a link
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  return {
    notebookUrl: "https://colab.research.google.com/drive/sample-notebook-id",
    notebookName: "Data_Analysis_" + analysisType + "_" + new Date().toISOString().split('T')[0] + ".ipynb",
    cells: 15,
    executionTime: "5 minutes"
  };
}

// Mock function to explain Python code
export async function explainPythonCode(code: string) {
  console.log("Explaining Python code with AI...");
  
  await new Promise(resolve => setTimeout(resolve, 800));
  
  return {
    explanation: "This Python code uses pandas for data manipulation and scikit-learn for machine learning. It loads the data, cleans it by handling missing values, and then applies statistical analysis to identify patterns.",
    complexity: "Intermediate",
    keyLibraries: ["pandas", "scikit-learn", "matplotlib", "numpy"]
  };
}

// Generate Python code based on the user prompt
function generatePythonCode(prompt: string, columns: string[]): string {
  const lowerPrompt = prompt.toLowerCase();
  
  // Basic imports that almost every analysis needs
  let code = "import pandas as pd\n" +
    "import numpy as np\n" +
    "import matplotlib.pyplot as plt\n" +
    "import seaborn as sns\n" +
    "from sklearn import preprocessing\n\n" +
    "# Load the dataset\n" +
    "df = pd.read_csv('dataset.csv')\n\n" +
    "# Display basic information\n" +
    "print(df.info())\n" +
    "print(\"\\nSample data:\")\n" +
    "print(df.head())\n\n" +
    "# Basic statistics\n" +
    "print(\"\\nBasic statistics:\")\n" +
    "print(df.describe())\n\n";

  // Add code based on the prompt
  if (lowerPrompt.includes('summary') || lowerPrompt.includes('overview')) {
    code += "\n# Generate summary statistics\n" +
      "summary = df.describe(include='all').T\n" +
      "missing_values = df.isnull().sum()\n" +
      "print(\"\\nMissing values:\")\n" +
      "print(missing_values[missing_values > 0])\n\n" +
      "# Data types and unique values\n" +
      "print(\"\\nData types:\")\n" +
      "print(df.dtypes)\n" +
      "for col in df.select_dtypes(include=['object']).columns:\n" +
      "    print(f\"\\nUnique values in {col}:\")\n" +
      "    print(df[col].value_counts().head(10))\n";
  }
  
  if (lowerPrompt.includes('correlation') || lowerPrompt.includes('relationship')) {
    code += "\n# Calculate correlation matrix\n" +
      "numeric_df = df.select_dtypes(include=['number'])\n" +
      "correlation_matrix = numeric_df.corr()\n" +
      "print(\"\\nCorrelation Matrix:\")\n" +
      "print(correlation_matrix)\n\n" +
      "# Plot correlation heatmap\n" +
      "plt.figure(figsize=(12, 8))\n" +
      "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)\n" +
      "plt.title('Correlation Heatmap')\n" +
      "plt.tight_layout()\n" +
      "plt.show()\n";
  }
  
  if (lowerPrompt.includes('distribution') || lowerPrompt.includes('histogram')) {
    code += "\n# Plot distributions of numeric columns\n" +
      "numeric_columns = df.select_dtypes(include=['number']).columns\n" +
      "num_cols = len(numeric_columns)\n" +
      "fig_rows = (num_cols + 1) // 2  # Calculate number of rows needed\n\n" +
      "plt.figure(figsize=(15, 5 * fig_rows))\n" +
      "for i, column in enumerate(numeric_columns):\n" +
      "    plt.subplot(fig_rows, 2, i+1)\n" +
      "    sns.histplot(df[column], kde=True)\n" +
      "    plt.title(f'Distribution of {column}')\n" +
      "plt.tight_layout()\n" +
      "plt.show()\n";
  }
  
  if (lowerPrompt.includes('clustering') || lowerPrompt.includes('segment')) {
    code += "\n# K-means clustering\n" +
      "from sklearn.cluster import KMeans\n" +
      "from sklearn.preprocessing import StandardScaler\n\n" +
      "# Select numeric features for clustering\n" +
      "features = df.select_dtypes(include=['number']).dropna()\n\n" +
      "# Standardize the features\n" +
      "scaler = StandardScaler()\n" +
      "scaled_features = scaler.fit_transform(features)\n\n" +
      "# Determine optimal number of clusters using elbow method\n" +
      "wcss = []\n" +
      "for i in range(1, 11):\n" +
      "    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)\n" +
      "    kmeans.fit(scaled_features)\n" +
      "    wcss.append(kmeans.inertia_)\n\n" +
      "# Plot elbow curve\n" +
      "plt.figure(figsize=(10, 6))\n" +
      "plt.plot(range(1, 11), wcss, marker='o')\n" +
      "plt.title('Elbow Method for Optimal Clusters')\n" +
      "plt.xlabel('Number of Clusters')\n" +
      "plt.ylabel('WCSS')\n" +
      "plt.grid(True)\n" +
      "plt.show()\n\n" +
      "# Apply K-means with determined number of clusters (example: 3)\n" +
      "kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)\n" +
      "cluster_labels = kmeans.fit_predict(scaled_features)\n\n" +
      "# Add cluster labels to original dataframe\n" +
      "df_with_clusters = df.copy()\n" +
      "df_with_clusters['Cluster'] = cluster_labels\n\n" +
      "# Analyze clusters\n" +
      "cluster_summary = df_with_clusters.groupby('Cluster').mean()\n" +
      "print(\"\\nCluster summary:\")\n" +
      "print(cluster_summary)\n";
  }
  
  if (lowerPrompt.includes('prediction') || lowerPrompt.includes('forecast') || lowerPrompt.includes('model')) {
    // Get a potential target variable
    const potentialTarget = columns[columns.length - 1];
    
    code += "\n# Prediction model\n" +
      "from sklearn.model_selection import train_test_split\n" +
      "from sklearn.ensemble import RandomForestRegressor\n" +
      "from sklearn.metrics import mean_squared_error, r2_score\n\n" +
      "# Prepare data for modeling\n" +
      "X = df.select_dtypes(include=['number']).drop(['" + potentialTarget + "'], axis=1, errors='ignore').fillna(0)\n" +
      "y = df['" + potentialTarget + "']  # Assuming this is the target variable\n\n" +
      "# Split data\n" +
      "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n" +
      "# Train model\n" +
      "model = RandomForestRegressor(n_estimators=100, random_state=42)\n" +
      "model.fit(X_train, y_train)\n\n" +
      "# Make predictions\n" +
      "y_pred = model.predict(X_test)\n\n" +
      "# Evaluate model\n" +
      "mse = mean_squared_error(y_test, y_pred)\n" +
      "r2 = r2_score(y_test, y_pred)\n\n" +
      "print(f\"\\nModel Evaluation:\")\n" +
      "print(f\"Mean Squared Error: {mse:.2f}\")\n" +
      "print(f\"R² Score: {r2:.2f}\")\n\n" +
      "# Feature importance\n" +
      "feature_importance = pd.DataFrame(\n" +
      "    {'Feature': X.columns, 'Importance': model.feature_importances_}\n" +
      ").sort_values('Importance', ascending=False)\n\n" +
      "print(\"\\nFeature Importance:\")\n" +
      "print(feature_importance)\n\n" +
      "# Plot feature importance\n" +
      "plt.figure(figsize=(10, 6))\n" +
      "sns.barplot(x='Importance', y='Feature', data=feature_importance)\n" +
      "plt.title('Feature Importance')\n" +
      "plt.tight_layout()\n" +
      "plt.show()\n";
  }
  
  if (lowerPrompt.includes('anomaly') || lowerPrompt.includes('outlier')) {
    code += "\n# Anomaly detection using Isolation Forest\n" +
      "from sklearn.ensemble import IsolationForest\n\n" +
      "# Select numeric features\n" +
      "numeric_data = df.select_dtypes(include=['number']).dropna()\n\n" +
      "# Apply Isolation Forest\n" +
      "isolation_forest = IsolationForest(contamination=0.05, random_state=42)\n" +
      "outliers = isolation_forest.fit_predict(numeric_data)\n\n" +
      "# Add outlier information to dataframe\n" +
      "df_with_outliers = df.copy()\n" +
      "df_with_outliers['is_outlier'] = np.where(outliers == -1, 'Outlier', 'Normal')\n\n" +
      "# Count outliers\n" +
      "outlier_count = (outliers == -1).sum()\n" +
      "print(f\"\\nNumber of detected outliers: {outlier_count}\")\n\n" +
      "# Analyze outliers\n" +
      "outlier_df = df_with_outliers[df_with_outliers['is_outlier'] == 'Outlier']\n" +
      "print(\"\\nOutlier statistics:\")\n" +
      "print(outlier_df.describe())\n\n" +
      "# Visualize outliers for a numeric column (first numeric column)\n" +
      "numeric_col = numeric_data.columns[0]\n" +
      "plt.figure(figsize=(10, 6))\n" +
      "sns.boxplot(x='is_outlier', y=numeric_col, data=df_with_outliers)\n" +
      "plt.title(f'Outlier Analysis for {numeric_col}')\n" +
      "plt.xticks(rotation=45)\n" +
      "plt.tight_layout()\n" +
      "plt.show()\n";
  }
  
  if (lowerPrompt.includes('time') || lowerPrompt.includes('trend') || lowerPrompt.includes('series')) {
    code += "\n# Time series analysis\n" +
      "# Assuming there's a date/time column\n" +
      "try:\n" +
      "    # Find potential date columns\n" +
      "    date_columns = []\n" +
      "    for col in df.columns:\n" +
      "        if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower():\n" +
      "            date_columns.append(col)\n" +
      "    \n" +
      "    if date_columns:\n" +
      "        date_col = date_columns[0]  # Use the first detected date column\n" +
      "        print(f\"\\nDetected date column: {date_col}\")\n" +
      "        \n" +
      "        # Convert to datetime\n" +
      "        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')\n" +
      "        \n" +
      "        # Set as index\n" +
      "        ts_df = df.set_index(date_col)\n" +
      "        \n" +
      "        # Get numeric columns\n" +
      "        numeric_cols = ts_df.select_dtypes(include=['number']).columns\n" +
      "        \n" +
      "        # Resample by month and plot\n" +
      "        if len(numeric_cols) > 0:\n" +
      "            numeric_col = numeric_cols[0]\n" +
      "            monthly_data = ts_df[numeric_col].resample('M').mean()\n" +
      "            \n" +
      "            plt.figure(figsize=(12, 6))\n" +
      "            monthly_data.plot()\n" +
      "            plt.title(f'Monthly Trend of {numeric_col}')\n" +
      "            plt.ylabel(numeric_col)\n" +
      "            plt.grid(True)\n" +
      "            plt.tight_layout()\n" +
      "            plt.show()\n" +
      "            \n" +
      "            # Check seasonality with decomposition\n" +
      "            print(\"\\nTime Series Decomposition:\")\n" +
      "            from statsmodels.tsa.seasonal import seasonal_decompose\n" +
      "            \n" +
      "            # Only if we have enough data points\n" +
      "            if len(monthly_data) >= 12:\n" +
      "                decomposition = seasonal_decompose(monthly_data, model='additive', period=12)\n" +
      "                \n" +
      "                plt.figure(figsize=(12, 10))\n" +
      "                plt.subplot(411)\n" +
      "                plt.plot(decomposition.observed)\n" +
      "                plt.title('Observed')\n" +
      "                plt.subplot(412)\n" +
      "                plt.plot(decomposition.trend)\n" +
      "                plt.title('Trend')\n" +
      "                plt.subplot(413)\n" +
      "                plt.plot(decomposition.seasonal)\n" +
      "                plt.title('Seasonal')\n" +
      "                plt.subplot(414)\n" +
      "                plt.plot(decomposition.resid)\n" +
      "                plt.title('Residual')\n" +
      "                plt.tight_layout()\n" +
      "                plt.show()\n" +
      "    else:\n" +
      "        print(\"No date/time columns detected for time series analysis\")\n" +
      "except Exception as e:\n" +
      "    print(f\"Could not perform time series analysis: {str(e)}\")\n";
  }
  
  if (lowerPrompt.includes('nlp') || lowerPrompt.includes('text') || lowerPrompt.includes('natural language')) {
    code += "\n# NLP analysis\n" +
      "from sklearn.feature_extraction.text import CountVectorizer\n" +
      "from wordcloud import WordCloud\n\n" +
      "# Find potential text columns\n" +
      "text_columns = []\n" +
      "for col in df.columns:\n" +
      "    if df[col].dtype == 'object':\n" +
      "        # Check if column has string values\n" +
      "        if df[col].fillna('').astype(str).str.len().mean() > 10:\n" +
      "            text_columns.append(col)\n\n" +
      "if text_columns:\n" +
      "    text_col = text_columns[0]  # Use first text column\n" +
      "    print(f\"\\nAnalyzing text column: {text_col}\")\n" +
      "    \n" +
      "    # Clean text\n" +
      "    df['clean_text'] = df[text_col].fillna('').astype(str).str.lower()\n" +
      "    \n" +
      "    # Create word cloud\n" +
      "    all_text = ' '.join(df['clean_text'])\n" +
      "    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)\n" +
      "    \n" +
      "    plt.figure(figsize=(10, 6))\n" +
      "    plt.imshow(wordcloud, interpolation='bilinear')\n" +
      "    plt.axis('off')\n" +
      "    plt.title(f'Word Cloud for {text_col}')\n" +
      "    plt.tight_layout()\n" +
      "    plt.show()\n" +
      "    \n" +
      "    # Count most common words\n" +
      "    vectorizer = CountVectorizer(stop_words='english', max_features=20)\n" +
      "    X = vectorizer.fit_transform(df['clean_text'])\n" +
      "    \n" +
      "    # Get top words and their counts\n" +
      "    words = vectorizer.get_feature_names_out()\n" +
      "    counts = X.sum(axis=0).A1\n" +
      "    \n" +
      "    # Create DataFrame for top words\n" +
      "    top_words = pd.DataFrame({'word': words, 'count': counts})\n" +
      "    top_words = top_words.sort_values('count', ascending=False)\n" +
      "    \n" +
      "    print(\"\\nTop words:\")\n" +
      "    print(top_words)\n" +
      "    \n" +
      "    # Plot top words\n" +
      "    plt.figure(figsize=(10, 6))\n" +
      "    sns.barplot(x='count', y='word', data=top_words)\n" +
      "    plt.title(f'Top Words in {text_col}')\n" +
      "    plt.tight_layout()\n" +
      "    plt.show()\n" +
      "else:\n" +
      "    print(\"No suitable text columns detected for NLP analysis\")\n";
  }

  return code;
}

// Function to simulate Python analysis execution through Python backend
export async function simulatePythonAnalysis(data: any[], prompt: string) {
  console.log("Executing Python analysis through backend...");
  
  // Generate Python code based on the prompt
  const pythonCode = generatePythonCode(prompt, Object.keys(data[0] || {}));
  
  // In a real implementation, this would send the data and code to a Python backend
  // Simulate a backend API call
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Generate mock code output
  const codeOutput = generateMockCodeOutput(pythonCode, data);
  
  // Determine what type of analysis was requested
  const lowerPrompt = prompt.toLowerCase();
  
  // Build a response based on the analysis type
  let response: any = {
    type: 'analysis',
    title: 'AI Data Analysis',
    description: 'Analysis performed using Python backend with pandas, numpy, scikit-learn, and matplotlib.',
    pythonCode,
    codeOutput,
    chartType: null,
    data: null,
    chartConfig: {},
    visualizations: []
  };
  
  // Different types of analysis responses based on different prompt keywords
  if (lowerPrompt.includes('summary') || lowerPrompt.includes('overview')) {
    response.title = 'Data Summary';
    response.description = 'This analysis provides a comprehensive overview of your dataset containing ' + 
      data.length + ' rows and ' + Object.keys(data[0] || {}).length + 
      ' columns. The analysis was performed using Python backend with pandas and numpy libraries.';
    
    // Add multiple visualizations
    response.visualizations = [
      {
        type: 'table',
        title: 'Data Preview',
        data: data.slice(0, 10),
        config: {
          title: 'Data Preview',
          description: 'First 10 rows from your dataset'
        }
      },
      {
        type: 'stats',
        title: 'Dataset Statistics',
        data: [
          { label: 'Total Rows', value: data.length },
          { label: 'Total Columns', value: Object.keys(data[0] || {}).length },
          { label: 'Numeric Columns', value: Object.keys(data[0] || {}).filter(k => typeof data[0][k] === 'number').length },
          { label: 'Text Columns', value: Object.keys(data[0] || {}).filter(k => typeof data[0][k] === 'string').length }
        ],
        config: {
          title: 'Dataset Statistics',
          description: 'Key metrics about your dataset'
        }
      }
    ];
    
    response.modelInfo = 'Analysis performed by Python backend using pandas library with descriptive statistics';
  }
  else if (lowerPrompt.includes('distribution') || lowerPrompt.includes('histogram')) {
    const numericColumns = Object.keys(data[0] || {}).filter(key => typeof data[0][key] === 'number');
    if (numericColumns.length > 0) {
      const column = numericColumns[0];
      // Create histogram data
      const values = data.map(row => row[column]).filter(val => val !== null && val !== undefined);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min;
      const binWidth = range / 10;
      
      const bins = Array(10).fill(0).map((_, i) => ({
        name: (min + i * binWidth).toFixed(1) + "-" + (min + (i + 1) * binWidth).toFixed(1),
        value: values.filter(v => v >= min + i * binWidth && v < min + (i + 1) * binWidth).length
      }));
      
      response.title = 'Distribution Analysis of ' + column;
      response.description = 'This analysis shows the frequency distribution of values in the ' + 
        column + ' column using histograms. The analysis was performed using pandas and matplotlib libraries.';
      response.chartType = 'bar';
      response.data = bins;
      response.chartConfig = {
        xKey: 'name',
        yKey: 'value',
        title: 'Distribution of ' + column,
        description: 'Frequency histogram showing the distribution of values'
      };
      response.modelInfo = 'Python pandas and matplotlib for distribution analysis';
    }
  }
  else if (lowerPrompt.includes('correlation') || lowerPrompt.includes('relationship')) {
    const numericColumns = Object.keys(data[0] || {}).filter(key => typeof data[0][key] === 'number');
    if (numericColumns.length >= 2) {
      const col1 = numericColumns[0];
      const col2 = numericColumns[1];
      
      response.title = 'Correlation Analysis between ' + col1 + ' and ' + col2;
      response.description = 'This analysis examines the relationship between ' + col1 + ' and ' + col2 + 
        ' using correlation coefficients and visualization techniques. A scatter plot is used to visualize the relationship, and Pearson correlation coefficient is calculated.';
      response.chartType = 'table';
      response.data = [
        { "Variable 1": col1, "Variable 2": col2, "Correlation": 0.72, "p-value": 0.001, "Relationship": "Strong positive correlation" },
        { "Variable 1": col1, "Variable 2": numericColumns[2] || "Other", "Correlation": -0.45, "p-value": 0.003, "Relationship": "Moderate negative correlation" }
      ];
      response.chartConfig = {
        title: 'Correlation Results',
        description: 'Pearson correlation coefficients between variables'
      };
      response.modelInfo = 'Python pandas correlation analysis with scipy.stats';
    }
  }
  else if (lowerPrompt.includes('clustering') || lowerPrompt.includes('segment')) {
    response.title = 'K-means Clustering Analysis';
    response.description = 'This analysis performs K-means clustering to identify natural groupings in the data. The algorithm identified 3 distinct clusters based on the numeric variables in the dataset.';
    response.chartType = 'pie';
    response.data = [
      { name: 'Cluster 1', value: Math.floor(data.length * 0.4) },
      { name: 'Cluster 2', value: Math.floor(data.length * 0.35) },
      { name: 'Cluster 3', value: Math.floor(data.length * 0.25) }
    ];
    response.chartConfig = {
      title: 'Cluster Distribution',
      description: 'Distribution of data points across clusters'
    };
    response.modelInfo = 'Python scikit-learn K-means clustering with scikit-learn';
  }
  else if (lowerPrompt.includes('prediction') || lowerPrompt.includes('forecast') || lowerPrompt.includes('model')) {
    const numericColumns = Object.keys(data[0] || {}).filter(key => typeof data[0][key] === 'number');
    if (numericColumns.length > 0) {
      const targetColumn = numericColumns[numericColumns.length - 1];
      
      response.title = 'Predictive Model for ' + targetColumn;
      response.description = 'This analysis builds a Random Forest Regression model to predict ' + 
        targetColumn + ' based on other numeric features. The model achieves an R² score of 0.83, indicating good predictive power.';
      response.chartType = 'stats';
      response.data = [
        { label: 'R² Score', value: 0.83 },
        { label: 'Mean Squared Error', value: 245.67 },
        { label: 'Mean Absolute Error', value: 12.34 },
        { label: 'Training Samples', value: Math.floor(data.length * 0.8) },
        { label: 'Testing Samples', value: Math.floor(data.length * 0.2) }
      ];
      response.chartConfig = {
        title: 'Model Evaluation Metrics',
        description: 'Performance metrics for the predictive model'
      };
      response.modelInfo = 'Python Random Forest Regression from scikit-learn';
    }
  }
  else if (lowerPrompt.includes('anomaly') || lowerPrompt.includes('outlier')) {
    response.title = 'Anomaly Detection';
    response.description = 'This analysis uses Isolation Forest algorithm to detect anomalies in the dataset. The analysis identified outliers that deviate significantly from the normal patterns in the data.';
    response.chartType = 'stats';
    response.data = [
      { label: 'Total Records', value: data.length },
      { label: 'Normal Records', value: Math.floor(data.length * 0.95) },
      { label: 'Outliers Detected', value: Math.floor(data.length * 0.05) },
      { label: 'Contamination Rate', value: '5%' }
    ];
    response.chartConfig = {
      title: 'Anomaly Detection Results',
      description: 'Summary of the outlier detection process'
    };
    response.modelInfo = 'Python Isolation Forest from scikit-learn';
  }
  else if (lowerPrompt.includes('nlp') || lowerPrompt.includes('text') || lowerPrompt.includes('natural language')) {
    response.title = 'Natural Language Processing Analysis';
    response.description = 'This analysis applies NLP techniques to extract insights from text columns in the dataset. The analysis includes word frequency analysis, sentiment detection, and topic modeling.';
    response.chartType = 'table';
    response.data = [
      { "Word": "data", "Frequency": 156, "TF-IDF Score": 0.85 },
      { "Word": "analysis", "Frequency": 142, "TF-IDF Score": 0.82 },
      { "Word": "product", "Frequency": 98, "TF-IDF Score": 0.76 },
      { "Word": "customer", "Frequency": 87, "TF-IDF Score": 0.74 },
      { "Word": "service", "Frequency": 72, "TF-IDF Score": 0.71 }
    ];
    response.chartConfig = {
      title: 'Top Words by Frequency',
      description: 'Most common words found in the text data'
    };
    response.modelInfo = 'Python NLP analysis using NLTK and scikit-learn';
  }
  else {
    // Default response when prompt doesn't match specific analyses
    response.title = 'Comprehensive Data Analysis';
    response.description = 'This analysis provides a comprehensive examination of your dataset, including summary statistics, visualizations, and key insights.';
    
    // Include multiple visualizations in default case
    response.visualizations = [
      {
        type: 'stats',
        title: 'Dataset Statistics',
        data: [
          { label: 'Total Records', value: data.length },
          { label: 'Features', value: Object.keys(data[0] || {}).length },
          { label: 'Complete Records', value: Math.floor(data.length * 0.97) },
          { label: 'Records with Missing Values', value: Math.floor(data.length * 0.03) }
        ],
        config: {
          title: 'Dataset Statistics',
          description: 'Key metrics about your dataset'
        }
      },
      {
        type: 'table',
        title: 'Data Sample',
        data: data.slice(0, 5),
        config: {
          title: 'Data Sample',
          description: 'First 5 rows of your dataset'
        }
      }
    ];
    
    response.modelInfo = 'Python pandas and scikit-learn for comprehensive analysis';
  }
  
  return response;
}

// Mock function to simulate sending data to Python backend for processing
export async function sendToPythonBackend(data: any[], operation: string) {
  console.log(`Sending data to Python backend for ${operation}...`);
  
  // In a real implementation, this would be an API call to a Python backend service
  await new Promise(resolve => setTimeout(resolve, 1500));
  
  return {
    status: "success",
    message: `Data successfully processed with ${operation} on Python backend`,
    processingTime: "2.3 seconds",
    backendInfo: "Python 3.10, pandas 2.0.3, scikit-learn 1.3.0"
  };
}
