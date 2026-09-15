import pandas as pd
import numpy as np

# Function to check non-empty values
def get_first_non_empty(row):
    import pandas as pd
    for col in ['pathJADNifti', 'pathKWLNifti', 'pathJADMLNifti', 'pathVerifiedMLNifti']:
        if pd.notna(row[col]) and row[col] != '':
            return row[col]
    return None  # Default to empty if all are empty
    
def datasetgenerator(df, require):
    """
    Generate a filtered and annotated dataset from a given DataFrame.
    
    This function takes an input DataFrame (df) along with two lists of 
    required columns/paths (requireMAN for manual (man) dataset, and 
    requireML for machine-learning (ML) dataset). It does the following:
    
    1. Filters out rows missing critical DICOM paths (e.g., vibe2min, DCE).
    2. Excludes rows with invalid dynamic series.
    3. Builds a 'pathmask' column that prioritizes different manual mask paths.
    4. Labels each subject as either 'man' or 'ML' based on presence of a mask.
    5. Applies additional filtering to each subset (man vs. ML) based on
       required columns in requireMAN and requireML.
    6. Sorts, cleans up, and returns the final DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame containing imaging paths, subject IDs, etc.
    requireMAN : list of str
        Columns/paths that must be non-null for manual (man) dataset entries.
    requireML : list of str
        Columns/paths that must be non-null for machine-learning (ML) dataset entries.
        
    Returns
    -------
    pd.DataFrame
        A cleaned, filtered, and annotated DataFrame.
    """
    
    # Make a copy of the original df to avoid mutating it
    df = df.copy()
    
    # Restrict the DataFrame to only these relevant columns
    # (i.e., dropping any columns not in fntab)
    fntab = [
        'subj', 'pathvibe2minDicom', 'pathT2Dicom', 'pathADCDicom',
        'pathJADNifti', 'pathKWLNifti', 'pathVerifiedMLNifti', 'pathJADMLNifti'
    ]
    df = df[fntab]
            
    # Construct a single 'pathmask' column prioritizing multiple possible 
    #    manual mask path columns (JAD, KWL, VerifiedML, or JADML).
    #    The first non-null path in that priority order is used.
    df['pathmask'] = df.apply(get_first_non_empty, axis=1)
    
    # Assign a dataset label: 'man' if a manual mask path exists, else 'ML'
    df["dataset"] = 'ML'
    df.loc[pd.notnull(df.pathmask), 'dataset'] = 'man'    

    # Filter data set
    df = df.dropna(subset=require, axis=0)
        
    # Sort the final DataFrame by subject ID
    df = df.sort_values(by='subj').reset_index(drop=True)
    
    # Replace any remaining np.nan values with None (useful if exporting to certain formats)
    df = df.replace({np.nan: None})

    print(f"{len(df)} datasett tilfredsstiller betingelsene")
    
    return df
