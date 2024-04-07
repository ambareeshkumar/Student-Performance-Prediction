from src.logger import logging
import sys


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = f'Error occured in the file {file_name} on line number {exc_tb.tb_lineno} error message {str(error)}'
    return error_msg


class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)
        self.error_msg = error_message_detail(error_msg, error_detail=error_detail)

    def __str__(self):
        return self.error_msg
