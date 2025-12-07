import sys                   # sys module gives access to traceback details
from src.logger import logging   # import custom logging from your project


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information:
    - file name
    - line number
    - original error message
    """

    _, _, exc_tb = error_detail.exc_info()  
    # exc_info() gives (exception_type, exception_value, traceback)
    # We only need traceback (exc_tb)

    file_name = exc_tb.tb_frame.f_code.co_filename  
    # tb_frame gives the frame where error occurred
    # f_code.co_filename gives the exact file name of the error

    line_number = exc_tb.tb_lineno  
    # tb_lineno gives the line number where the error occurred

    # Format and return a detailed error message
    return f"Error occurred in file: {file_name} at line: {line_number} with message: {str(error)}"


class CustomException(Exception):
    """
    Custom Exception that attaches extra debugging information
    like file name, line number, and original error message.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  
        # Initialize base Exception class

        # Create detailed error message using helper function
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        # When exception is printed, return the detailed message
        return self.error_message


# Only runs when this file is executed directly
if __name__ == "__main__":
    try:
        a = 1 / 0        # This will cause ZeroDivisionError
    except Exception as e:
        logging.info("Divided by zero error caught")  
        # Log custom info message

        raise CustomException(e, sys)  
        # Raise your custom exception with full details
        

    
    