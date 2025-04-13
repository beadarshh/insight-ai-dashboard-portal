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
