import pandas as pd
import logging
from pathlib import Path

class DatabaseManager:
    def __init__(self):
        self.df = None
        self.current_file = None
        self.database_directory = None

    def load_database(self, file_path: str) -> bool:
        """Load database from CSV file."""
        try:
            self.df = pd.read_csv(file_path)
            self.current_file = file_path
            # Store the directory where the database file is located
            self.database_directory = Path(file_path).parent

            # Ensure required columns exist
            required_columns = ['pbr_location', 'processed', 'false_positive', 'user']  # Add 'user' to required columns
            for col in required_columns:
                if col not in self.df.columns:
                    if col == 'user':
                        self.df[col] = pd.Series(dtype=str)  # String type for user column
                    else:
                        self.df[col] = pd.Series(dtype=bool)

            # Convert string values to proper boolean
            bool_columns = ['processed', 'false_positive']
            for col in bool_columns:
                # First convert column to string to handle mixed types
                self.df[col] = self.df[col].astype(str)
                # Map various string representations to boolean
                self.df[col] = self.df[col].map({
                    'True': True, 'true': True, '1': True, 
                    'False': False, 'false': False, '0': False,
                    'nan': False, 'None': False, '': False
                })
                # Handle any remaining values as False
                self.df[col] = self.df[col].fillna(False).astype(bool)

            logging.info(f"Loaded database with {len(self.df)} entries")
            return True

        except Exception as e:
            logging.error(f"Error loading database: {str(e)}", exc_info=True)
            self.df = None
            return False

    def get_full_file_path(self, relative_path: str) -> str:
        """
        Get the full path for a file relative to the database directory.
        
        Args:
            relative_path: The relative path from the database
            
        Returns:
            str: The full absolute path to the file
        """
        if self.database_directory is None:
            return relative_path
        
        # Convert to Path object and resolve relative to database directory
        full_path = self.database_directory / relative_path
        return str(full_path)

    def get_next_unprocessed(self):
        """Get the next unprocessed PBR entry."""
        try:
            if self.df is None or len(self.df) == 0:
                return None

            # Get first unprocessed and non-false-positive entry
            mask = (~self.df['processed'].astype(bool)) & (~self.df['false_positive'].astype(bool))
            unprocessed = self.df[mask]

            if len(unprocessed) == 0:
                return None

            return unprocessed.iloc[0].to_dict()

        except Exception as e:
            logging.error(f"Error getting next unprocessed: {str(e)}", exc_info=True)
            return None

    def mark_false_positive(self, pbr_name: str):
        """Mark a PBR as false positive."""
        try:
            if self.df is None:
                return False

            # Find the row with matching pbr_name
            mask = self.df['pbr_location'].str.contains(pbr_name, na=False)
            if not mask.any():
                logging.warning(f"PBR {pbr_name} not found in database")
                return False

            # Update the false_positive column
            self.df.loc[mask, 'false_positive'] = True
            
            # Save changes to file
            if self.current_file:
                self.df.to_csv(self.current_file, index=False)
            return True

        except Exception as e:
            logging.error(f"Error marking false positive: {str(e)}", exc_info=True)
            return False

    def update_entry(self, pbr_name: str, **kwargs):
        """Update database entry with processing results."""
        try:
            if self.df is None:
                return False

            # Find the row with matching pbr_name
            mask = self.df['pbr_location'].str.contains(pbr_name, na=False)
            if not mask.any():
                logging.warning(f"PBR {pbr_name} not found in database")
                return False

            # Update each field
            for key, value in kwargs.items():
                if key not in self.df.columns:
                    # Initialize new columns as string type if it's the user column
                    if key == 'user':
                        self.df[key] = pd.Series(dtype=str)
                    else:
                        self.df[key] = None
                
                # Process the value based on its type or column
                if key == 'user':
                    processed_value = str(value)  # Ensure user is stored as string
                elif isinstance(value, (int, float)):
                    processed_value = value
                else:
                    processed_value = str(value)

                # Update the value
                self.df.loc[mask, key] = processed_value

            # Save changes to file
            if self.current_file:
                self.df.to_csv(self.current_file, index=False)
            return True

        except Exception as e:
            logging.error(f"Error updating entry: {str(e)}", exc_info=True)
            return False
