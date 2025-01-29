import csv
import io

class CSVCreator:
    @staticmethod
    def export_dict_to_csv(self, data: dict[str, str | bool]) -> bytes:

        ## Converts a dictionary to a CSV file with headers in first row and values in second row.
        ## Return value is data (which I think makes it more compatible with sockets, if not we can just return the output.getValue()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(data.keys())
        writer.writerow(data.values())

        ## Get the string value and encode to bytes
        csv_bytes = output.getvalue().encode('utf-8')

        ## Clean up
        output.close()

        return csv_bytes