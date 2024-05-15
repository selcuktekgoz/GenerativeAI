import logging

from rich.logging import RichHandler

lformat = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level="INFO",
    format=lformat,
    datefmt="[%X]",
    handlers=[RichHandler(show_time=False, show_level=False)],
)


def linfo(msg):
    logging.info(msg)


def lerror(msg):
    logging.error(msg)


if __name__ == "__main__":
    linfo("log info message")
    lerror("log error message")
