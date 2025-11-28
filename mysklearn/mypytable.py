from mysklearn import myutils

import copy
import csv

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        if not self.column_names:
            print("Empty table")
            return
            
        # Calculate column widths
        col_widths = []
        for i, col_name in enumerate(self.column_names):
            max_width = len(str(col_name))
            for row in self.data:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)  # Add padding
        
        # Print header
        header_line = "|"
        for i, col_name in enumerate(self.column_names):
            header_line += f" {str(col_name):<{col_widths[i]-1}}|"
        print(header_line)
        
        # Print separator
        separator = "+"
        for width in col_widths:
            separator += "-" * width + "+"
        print(separator)
        
        # Print data rows
        for row in self.data:
            row_line = "|"
            for i in range(len(self.column_names)):
                value = str(row[i]) if i < len(row) else ""
                row_line += f" {value:<{col_widths[i]-1}}|"
            print(row_line)

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        if not self.data:
            return 0, len(self.column_names)
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        # Find column index
        if isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError(f"Column '{col_identifier}' not found")
            col_index = self.column_names.index(col_identifier)
        elif isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError(f"Column index {col_identifier} out of range")
            col_index = col_identifier
        else:
            raise ValueError("Column identifier must be string or int")

        # Extract column data
        column_data = []
        for row in self.data:
            if col_index < len(row):
                value = row[col_index]
                if include_missing_values or value != "NA":
                    column_data.append(value)
            elif include_missing_values:
                column_data.append("NA")

        return column_data

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for i, row in enumerate(self.data):
            for j, value in enumerate(row):
                try:
                    # Try to convert to float first
                    numeric_value = float(value)
                    # If it's a whole number, convert to int
                    if numeric_value.is_integer():
                        self.data[i][j] = int(numeric_value)
                    else:
                        self.data[i][j] = numeric_value
                except (ValueError, TypeError):
                    # Leave as is if can't convert
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        # Sort in descending order so we remove from the end first
        # This prevents index shifting issues
        sorted_indexes = sorted(row_indexes_to_drop, reverse=True)
        
        for index in sorted_indexes:
            if 0 <= index < len(self.data):
                del self.data[index]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            self.column_names = next(csv_reader)  # First row is header
            self.data = []
            for row in csv_reader:
                self.data.append(row)
        
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(self.column_names)  # Write header
            csv_writer.writerows(self.data)  # Write data rows

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        # Get column indexes
        key_indexes = []
        for col_name in key_column_names:
            if col_name not in self.column_names:
                raise ValueError(f"Column '{col_name}' not found")
            key_indexes.append(self.column_names.index(col_name))

        seen_keys = set()
        duplicate_indexes = []

        for i, row in enumerate(self.data):
            # Create key tuple from specified columns
            key = tuple(row[j] if j < len(row) else "NA" for j in key_indexes)
            
            if key in seen_keys:
                duplicate_indexes.append(i)
            else:
                seen_keys.add(key)

        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        rows_to_remove = []
        for i, row in enumerate(self.data):
            if "NA" in row:
                rows_to_remove.append(i)
        
        # Remove rows in reverse order to avoid index shifting
        self.drop_rows(rows_to_remove)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        if col_name not in self.column_names:
            raise ValueError(f"Column '{col_name}' not found")
        
        col_index = self.column_names.index(col_name)
        
        # Calculate average of non-missing numeric values
        numeric_values = []
        for row in self.data:
            if col_index < len(row) and row[col_index] != "NA":
                try:
                    numeric_values.append(float(row[col_index]))
                except (ValueError, TypeError):
                    pass  # Skip non-numeric values
        
        if not numeric_values:
            return  # No numeric values to compute average
        
        average = sum(numeric_values) / len(numeric_values)
        
        # Replace missing values with average
        for row in self.data:
            if col_index < len(row) and row[col_index] == "NA":
                row[col_index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_table = MyPyTable(["attribute", "min", "max", "mid", "avg", "median"])
        
        for col_name in col_names:
            if col_name not in self.column_names:
                continue
                
            # Get numeric values (excluding missing values)
            column_data = self.get_column(col_name, include_missing_values=False)
            numeric_values = []
            for value in column_data:
                try:
                    numeric_values.append(float(value))
                except (ValueError, TypeError):
                    pass
            
            if not numeric_values:
                continue  # Skip if no numeric values
            
            # Calculate statistics
            numeric_values.sort()  # For median calculation
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            mid_val = (min_val + max_val) / 2
            avg_val = sum(numeric_values) / len(numeric_values)
            
            # Calculate median
            n = len(numeric_values)
            if n % 2 == 0:
                median_val = (numeric_values[n//2 - 1] + numeric_values[n//2]) / 2
            else:
                median_val = numeric_values[n//2]
            
            # Add row to summary table
            summary_table.data.append([col_name, min_val, max_val, mid_val, avg_val, median_val])
        
        return summary_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # Get key column indexes for both tables
        self_key_indexes = []
        other_key_indexes = []
        
        for col_name in key_column_names:
            if col_name not in self.column_names or col_name not in other_table.column_names:
                raise ValueError(f"Key column '{col_name}' not found in both tables")
            self_key_indexes.append(self.column_names.index(col_name))
            other_key_indexes.append(other_table.column_names.index(col_name))

        # Create new column names (self columns + other columns excluding key columns)
        new_column_names = self.column_names[:]
        for i, col_name in enumerate(other_table.column_names):
            if i not in other_key_indexes:
                new_column_names.append(col_name)

        # Create hash table for other_table rows
        other_hash = {}
        for row in other_table.data:
            key = tuple(row[i] if i < len(row) else "NA" for i in other_key_indexes)
            if key not in other_hash:
                other_hash[key] = []
            other_hash[key].append(row)

        # Perform inner join
        joined_data = []
        for self_row in self.data:
            self_key = tuple(self_row[i] if i < len(self_row) else "NA" for i in self_key_indexes)
            
            if self_key in other_hash:
                for other_row in other_hash[self_key]:
                    # Combine rows: self_row + other_row (excluding key columns from other)
                    new_row = self_row[:]
                    for i, value in enumerate(other_row):
                        if i not in other_key_indexes:
                            new_row.append(value)
                    joined_data.append(new_row)

        return MyPyTable(new_column_names, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        # Get key column indexes for both tables
        self_key_indexes = []
        other_key_indexes = []
        
        for col_name in key_column_names:
            if col_name not in self.column_names or col_name not in other_table.column_names:
                raise ValueError(f"Key column '{col_name}' not found in both tables")
            self_key_indexes.append(self.column_names.index(col_name))
            other_key_indexes.append(other_table.column_names.index(col_name))

        # Create new column names
        new_column_names = self.column_names[:]
        other_non_key_cols = []
        for i, col_name in enumerate(other_table.column_names):
            if i not in other_key_indexes:
                new_column_names.append(col_name)
                other_non_key_cols.append(i)

        # Create hash tables for both tables
        self_hash = {}
        other_hash = {}
        
        for row in self.data:
            key = tuple(row[i] if i < len(row) else "NA" for i in self_key_indexes)
            if key not in self_hash:
                self_hash[key] = []
            self_hash[key].append(row)

        for row in other_table.data:
            key = tuple(row[i] if i < len(row) else "NA" for i in other_key_indexes)
            if key not in other_hash:
                other_hash[key] = []
            other_hash[key].append(row)

        # Get all unique keys
        all_keys = set(self_hash.keys()) | set(other_hash.keys())
        
        joined_data = []
        for key in all_keys:
            self_rows = self_hash.get(key, [])
            other_rows = other_hash.get(key, [])
            
            if self_rows and other_rows:
                # Inner join case
                for self_row in self_rows:
                    for other_row in other_rows:
                        new_row = self_row[:]
                        for i in other_non_key_cols:
                            if i < len(other_row):
                                new_row.append(other_row[i])
                            else:
                                new_row.append("NA")
                        joined_data.append(new_row)
            elif self_rows:
                # Left join case
                for self_row in self_rows:
                    new_row = self_row[:]
                    for _ in other_non_key_cols:
                        new_row.append("NA")
                    joined_data.append(new_row)
            elif other_rows:
                # Right join case
                for other_row in other_rows:
                    new_row = ["NA"] * len(self.column_names)
                    # Fill in key values
                    for i, key_val in enumerate(key):
                        if i < len(self_key_indexes):
                            new_row[self_key_indexes[i]] = key_val
                    # Add other table's non-key values
                    for i in other_non_key_cols:
                        if i < len(other_row):
                            new_row.append(other_row[i])
                        else:
                            new_row.append("NA")
                    joined_data.append(new_row)

        return MyPyTable(new_column_names, joined_data)