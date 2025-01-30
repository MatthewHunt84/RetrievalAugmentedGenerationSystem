import csv
import io

class CSVCreator:
    @staticmethod
    def export_dict_to_csv(data: dict[str, list[str]]) -> bytes:
        ## Converts a dictionary to a CSV file with headers in first row and values in second row.
        if not data or not all(isinstance(v, list) for v in data.values()):
            raise ValueError("Data must be a dictionary with list values")

        # Verify all lists have the same length
        list_lengths = {len(v) for v in data.values()}
        if len(list_lengths) != 1:
            raise ValueError("All value lists must have the same length")

        output = io.StringIO()
        writer = csv.writer(output)

        ## Write headers (dictionary keys)
        writer.writerow(data.keys())

        # Write data rows
        num_rows = len(next(iter(data.values())))
        for i in range(num_rows):
            row = [data[key][i] for key in data.keys()]
            writer.writerow(row)

        ## Get the csv value and encode to bytes
        ## Return value is data
        ## I *think* makes it more compatible with sockets, but if not we can remove this encoding step
        csv_bytes = output.getvalue().encode('utf-8')

        ## Clean up
        output.close()

        return csv_bytes

# class CSVCreator:
#     @staticmethod
#     def export_dict_to_csv(self, data: dict[str, str | bool]) -> bytes:
#
#         ## Converts a dictionary to a CSV file with headers in first row and values in second row.
#         ## Return value is data (which I think makes it more compatible with sockets, if not we can just return the output.getValue()
#         output = io.StringIO()
#         writer = csv.writer(output)
#         writer.writerow(data.keys())
#         writer.writerow(data.values())
#
#         ## Get the string value and encode to bytes
#         csv_bytes = output.getvalue().encode('utf-8')
#
#         ## Clean up
#         output.close()
#
#         return csv_bytes