#!/usr/bin/env python

import logging
import sys

EXIT_SIG = -1

format_str = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s : '
format_str += '%(message)s'

stdout_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(
    'SDBI.log', mode='a', encoding="utf-8", delay=False)

logging.basicConfig(format=format_str, level=logging.INFO, 
    handlers=[file_handler, stdout_handler])
