import csv
import io

class CSVCreator:
    @staticmethod
    def export_dict_to_csv(data: dict[str, list[str]]) -> bytes:
        ## Converts a dictionary to a CSV file with headers in first row and values in second row.
        if not data or not all(isinstance(v, list) for v in data.values()):
            raise ValueError("Input error: Can't make CSV with data passed in. Input data must be a dictionary with list values")

        ## Verify all lists have the same length
        list_lengths = {len(v) for v in data.values()}
        if len(list_lengths) != 1:
            raise ValueError("Input Error: Can't make a CSV because the list of values are of different sizes")

        ## Use simple built in CSV creation.
        ## If we need .xlsx or something we can swap this out
        output = io.StringIO()
        writer = csv.writer(output)

        ## Write headers (dictionary keys == column names)
        writer.writerow(data.keys())

        ## Write data rows (each item will be its own row)
        num_rows = len(next(iter(data.values())))
        for i in range(num_rows):
            row = [data[key][i] for key in data.keys()]
            writer.writerow(row)

        ## Get the csv value and encode to bytes
        ## Return value is data
        ## I think sending the result as data makes it more compatible with sockets, but
        ## If not we can remove this encoding step
        csv_bytes = output.getvalue().encode('utf-8')

        ## Clean up
        output.close()

        return csv_bytes
