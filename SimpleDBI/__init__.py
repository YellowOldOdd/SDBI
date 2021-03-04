#!/usr/bin/env python

import logging

format_str = '%(filename)s[line:%(lineno)d] - %(levelname)s : '
format_str += '%(message)s' 
logging.basicConfig(format=format_str, level=logging.INFO, )

EXIT_SIG = -1